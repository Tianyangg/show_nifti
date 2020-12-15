import numpy as np
import gui
import nibabel as nib
import os
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, color, io, img_as_float
from matplotlib.backends.backend_pdf import PdfPages



def load_img(im_path, mask=False):
    img = sitk.ReadImage(im_path)
    return img


def resampleRAI(img):
    # get image data
    image_out = sitk.GetImageFromArray(sitk.GetArrayFromImage(img))

    # setup other image characteristics
    image_out.SetOrigin(img.GetOrigin())
    image_out.SetSpacing(img.GetSpacing())
    # set to RAI
    image_out.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    return image_out



def int2binary(mask_slice):
    color_dict = {
        '0': [0, 0, 0],
        '1': [0, 0, 1],
        '2': [0, 1, 0],
        '3': [1, 0, 0],
        '4': [0, 1, 1],
        '5': [1, 0, 1],
        '6': [1, 1, 0],
        '7': [1, 1, 1]
    }
    mask_slice_rgb = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3))
    unique_label = np.unique(mask_slice)
    for l in unique_label:
        mask_slice_rgb[mask_slice == l] = color_dict[str(l % 7)]

    return mask_slice_rgb


def combine_img_mask(img_slice, mask_slice, alpha=0.4):
    img_slice = (img_slice.astype(float) - img_slice.min())/(img_slice.max() - img_slice.min())
    img_color = np.dstack((img_slice, img_slice, img_slice))
    mask_slice = int2binary(mask_slice)
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(mask_slice)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def slice_volume(img, slice_ind, plane, label=None, level_window=None):
    """
    show image using pyplot
    :param img: simpleitk image style
    :param plane:
    :return:
    """
    if label is not None:
        img_arr = sitk.GetArrayFromImage(img)
        label_arr = sitk.GetArrayFromImage(label)
        if level_window is not None:
            # calulate lower and upper
            level = level_window[0]
            window = level_window[1]
            upper = level + window / 2
            lower = level - window / 2
            img_arr[img_arr < lower] = lower
            img_arr[img_arr > upper] = upper
        if plane == 'axial':
            nda_slice = img_arr[slice_ind, :, :]
            label_slice = label_arr[slice_ind, :, :]
            nda_slice = np.flip(np.flip(nda_slice), axis=1)
            label_slice = np.flip(np.flip(label_slice), axis=1)
            nda_slice = combine_img_mask(nda_slice, label_slice)
        elif plane == 'coronal':
            nda_slice = img_arr[:, slice_ind, :]
            label_slice = label_arr[:, slice_ind, :]
            nda_slice = np.flip(np.flip(nda_slice), axis=1)
            label_slice = np.flip(np.flip(label_slice), axis=1)
            nda_slice = combine_img_mask(nda_slice, label_slice)
        else:
            nda_slice = img_arr[:, :, slice_ind]
            label_slice = label_arr[:, :, slice_ind]
            nda_slice = np.flip(nda_slice)
            label_slice = np.flip(label_slice)
            nda_slice = combine_img_mask(nda_slice, label_slice)

        # plt.imshow(nda_slice)
        # plt.show()
    else:
        # readin rai gives z y x
        nda = sitk.GetArrayFromImage(img).astype(float)
        if level_window is not None:
            # calulate lower and upper
            level = level_window[0]
            window = level_window[1]
            upper = level + window/2
            lower = level - window/2
            nda[nda < lower] = lower
            nda[nda > upper] = upper
        # nda = (nda - nda.min())/(nda.max() - nda.min())
        # print(nda)
        if plane == 'axial':
            nda_slice = nda[slice_ind, :, :]
            nda_slice = np.flip(np.flip(nda_slice), axis=1)
        elif plane == 'coronal':
            nda_slice = nda[:, slice_ind, :]
            nda_slice = np.flip(np.flip(nda_slice), axis=1)
        else:
            nda_slice = nda[:, :, slice_ind]
            nda_slice = np.flip(nda_slice)

        # plt.imshow(nda_slice, cmap=plt.cm.gray)
        # plt.show()

    return nda_slice


def make_isotropic(image, interpolator=sitk.sitkNearestNeighbor):
    '''
    Resample an image to isotropic pixels (using smallest spacing from original) and save to file. Many file formats
    (jpg, png,...) expect the pixels to be isotropic. By default the function uses a linear interpolator. For
    label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
    '''
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing]*image.GetDimension()
    new_size = [int(round(osz*ospc/min_spacing)) for osz, ospc in zip(original_size, original_spacing)]
    new_img = sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                         image.GetPixelID())
    new_img.SetSpacing(new_spacing)
    return new_img


def show_image(img_path, core_ind=None, mask_list=None):
    """

    :param img_path:
    :param core_ind: the core index slice [x, y, z]  -> to numpy: core_ind[2], core_ind[1], core_ind[0]
    :param mask_list:
    :return:
    """
    img = load_img(img_path)  # load image in
    if core_ind is not None:
        pass
    else:
        core_ind = np.array(img.GetSize())/2
        core_ind = core_ind.astype(int)  # x, y, z
    original_spacing = img.GetSpacing()

    # resample2rai and isotropic
    img = resampleRAI(img)
    img = make_isotropic(img, interpolator=sitk.sitkNearestNeighbor)
    new_spacing = img.GetSpacing()
    core_ind_newSpacing = np.array(np.divide(original_spacing, new_spacing) * core_ind).astype(int)

    a_slice_list = []
    c_slice_list = []
    s_slice_list = []
    img_a_slice = slice_volume(img, core_ind_newSpacing[2], 'axial', level_window=[-600, 1300])
    img_c_slice = slice_volume(img, core_ind_newSpacing[1], 'coronal', level_window=[-600, 1300])
    img_s_slice = slice_volume(img, core_ind_newSpacing[0], 'sagittal', level_window=[-600, 1300])

    if mask_list is not None:
        for mask_path in mask_list:
            label = load_img(mask_path)
            label = resampleRAI(label)
            label = make_isotropic(label, interpolator=sitk.sitkNearestNeighbor)

            a_slice = slice_volume(img, core_ind_newSpacing[2], 'axial', label=label, level_window=[-600, 1300])
            c_slice = slice_volume(img, core_ind_newSpacing[1], 'coronal', label=label, level_window=[-600, 1300])
            s_slice = slice_volume(img, core_ind_newSpacing[0], 'sagittal', label=label, level_window=[-600, 1300])
            a_slice_list.append(a_slice)
            c_slice_list.append(c_slice)
            s_slice_list.append(s_slice)

    return img_a_slice, a_slice_list, img_c_slice, c_slice_list, img_s_slice, s_slice_list



def save_figs(img_list, mask_list=None):
    pp = PdfPages('./foo.pdf')
    for ind, img in enumerate(img_list):
        img_fig = plt.figure()
        plt.imshow(img, cmap=plt.cm.gray)
        plt.axis('off')
        pp.savefig(img_fig)
        if mask_list is not None:
            img_mask = plt.figure()
            plt.imshow(mask_list[ind])
            plt.axis('off')
            pp.savefig(img_mask)

    pp.close()

def main():
    im_rootpath = "./data"
    subjects = os.listdir(im_rootpath)
    for sub in subjects:
        img_a_slice, a_slice_list, img_c_slice, c_slice_list, img_s_slice, s_slice_list = \
            show_image(img_path=os.path.join(im_rootpath, sub, 'study_0255.nii.gz'), mask_list=[os.path.join(im_rootpath, sub, 'study_0255_mask.nii.gz')])
        save_figs([img_a_slice], a_slice_list)
        # core_ind = np.array([250, 200, 14])
        # img = load_img(os.path.join(im_rootpath, sub, 'study_0255.nii.gz'))
        # label = load_img(os.path.join(im_rootpath, sub, 'study_0255_mask.nii.gz'))
        # original_spacing = img.GetSpacing()
        # img = resampleRAI(img)
        # img = make_isotropic(img, interpolator=sitk.sitkNearestNeighbor)
        # label = resampleRAI(label)
        # label = make_isotropic(label, interpolator=sitk.sitkNearestNeighbor)
        # new_spacing = img.GetSpacing()
        # core_ind_newSpacing = np.array(np.divide(original_spacing, new_spacing) * core_ind).astype(int)
        # a_slice = slice_volume(img, core_ind_newSpacing[2], 'axial', label=label, level_window=[-600, 1300])
        # c_slice = slice_volume(img, core_ind_newSpacing[1], 'coronal', label=label, level_window=[-600, 1300])
        # s_slice = slice_volume(img, core_ind_newSpacing[0], 'sagittal', label=label, level_window=[-600, 1300])

    return 0


if __name__ == "__main__":
    main()
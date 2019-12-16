import os
import progressbar
import numpy as np
import SimpleITK as sitk
import random
import pprint
import scipy.ndimage as snd
from utils import adjust_center_for_boundaries, extract_patch, uniform_sample
from utils import resample

def sample_patches(lid, imgpath, lblpath):
    '''
    Extract Patches

    Parameters:
        imgpath - path of the image

        lblpath - path of the corressponding segmented image

        lid - 

    '''

    #Image

    #Read the image and return an object (generator comprehension), Here 'image' is a sitk's object
    image = sitk.ReadImage(imgpath)
    image = resample(image, (1.0, 1.0, 1.0), interpolator = sitk.sitkLinear)
    #get the pixel values from the image
    img_arr = sitk.GetArrayFromImage(image)
    #Zooms the image according to the zoom
    img_arr = snd.zoom(img_arr, zoom = (0.5, 0.5, 0.5), order = 1)
    #pixel values are clipped b/w (-100, 400) and converted to float32
    img_arr = np.float32(np.clip(img_arr, -100, 400))   #Why clipping b/w (-100,400)?
    #Some kind of data preprocessing
    img_arr = np.uint8(255*(img_arr + 100)/(500))
    #Pad the image
    img_arr = np.pad(img_arr, ((100,100),(100,100),(100,100)), mode = 'constant')

    #Label    
    label = sitk.ReadImage(lblpath)
    label = resample(label, (1.0, 1.0, 1.0), interpolator = sitk.sitkNearestNeighbor)
    lbl_arr = sitk.GetArrayFromImage(label)
    lbl_arr[lbl_arr == 2] = 1
    lbl_arr = np.uint8(snd.zoom(lbl_arr, zoom = (0.5, 0.5, 0.5), order = 0))
    #Copies the content of 'lbl_arr' and adds '1' to it
    lbl_arr_cp = lbl_arr.copy() + 1
    lbl_arr = np.pad(lbl_arr, ((100,100),(100,100),(100,100)), mode = 'constant')
    lbl_arr_cp = np.pad(lbl_arr_cp, ((100,100),(100,100),(100,100)), mode = 'constant')
    lbl_arr_cp -= 1
    
    #Getting the crops for liver class and non-liver class
    class1_locs = uniform_sample(lbl_arr_cp == 0, 50)
    class2_locs = uniform_sample(lbl_arr_cp == 1, 50)
#     print(' class 1, class 2 :', len(class1_locs), len(class2_locs))
    locs = class1_locs[:5] + class2_locs[:45]
    random.shuffle(locs)
    
    patch_size, lbl_size = [116, 132, 132], [28, 44, 44]
    liver_pixel_count = {}
    for idx, l in enumerate(locs):
        l = adjust_center_for_boundaries(l, patch_size, img_arr.shape)
        img_patch = extract_patch(img_arr, l, patch_size)
        lbl_patch = extract_patch(lbl_arr, l, lbl_size)
        liver_pixel_count[idx] = np.sum(lbl_patch)
        save_dir = './data/train'
        inppname = 'img' + str(lid) + '_input'+str(idx)+'.npy'
        tgtpname = 'img' + str(lid) + '_label'+str(idx)+'.npy'
        np.save(os.path.join(save_dir, inppname), img_patch)
        np.save(os.path.join(save_dir, tgtpname), lbl_patch)




litsids = [1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 19, 20, 22, 25, 26, 27, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 48, 49, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 83, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 102, 103, 104, 105, 107, 108, 111, 113, 116, 117, 119, 120, 122, 124, 126, 128, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]
img_lbl_pairs = []
data_root = '/mnt/data/LiverCT/Parenchyma/LITS/train'

for lid in litsids:
    imgpath = os.path.join(data_root, 'volume-'+str(lid)+'.nii')
    lblpath = os.path.join(data_root, 'segmentation-'+str(lid)+'.nii')
    img_lbl_pairs.append([lid, imgpath, lblpath])

# bar = progressbar.ProgressBar(len(img_lbl_pairs)).start()
# patch_save_root = './data'
for step, [lid, imgpath, lblpath] in enumerate(img_lbl_pairs):
    sample_patches(lid, imgpath, lblpath)
#    bar.update(step+1)


"""
code for parallelization
import multiprocessing as mp
litsids = [102, 103, 104, 105, 107, 108]
img_lbl_pairs = []
data_root = '/mnt/data/LiverCT/Parenchyma/LITS'
for lid in listids:
    img_lbl_pairs.append()
pool = mp.Pool(3)
"""


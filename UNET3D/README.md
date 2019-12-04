## Training a 3D UNET for liver segmentation

This repository contains code to train a unet3d for liver segmentation. 

## Requirements : 
1. Pytorch >= 1.1
2. SimpleITK >= 1.2
3. Python >= 3.5


## Steps to train your own model : 
#### Extract patches for unet3d training 
1. Create `models` and `data` directories in the training folder 
2. Set `data_root` in patch_extract.py to your location of LITS training data and run `python3 patch_extract.py`
3. Set `gpu_ids`  in `train.py` and run `python3 train.py` to train the 3d unet for liver segmentation

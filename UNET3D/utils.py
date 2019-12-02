import numpy as np
import SimpleITK as sitk

def uniform_sample(bool_arr, max_locs):
	shape = bool_arr.shape
	if np.sum(bool_arr)/np.prod(shape) < 5e-5:
		return []
	locs = []
	while True:
		rx = np.random.randint(shape[0])
		ry = np.random.randint(shape[1])
		rz = np.random.randint(shape[2])
		if bool_arr[rx][ry][rz]:
			locs.append([rx,ry,rz])
		if len(locs) == max_locs:
			break
	return locs


def adjust_center_for_boundaries( center, patch_size, shape):
    cx, cy, cz = center
    px, py, pz = patch_size
    sx, sy, sz = shape
    _px, px_, _py, py_, _pz, pz_ = px//2, px - px//2, py//2, py - py//2, pz//2, pz - pz//2 
    if cx - _px < 0:
        cx = 0 + _px
    elif cx + px_ > sx:
        cx = sx - px_

    if cy - _py < 0:
        cy = 0 + _py
    elif cy + py_ > sy:
        cy = sy - py_

    if cz - _pz < 0:
        cz = 0 + _pz
    elif cz + pz_ > sz:
        cz = sz - pz_

    return cx, cy, cz


def extract_patch(image, loc, patch_size):
    px, py, pz = patch_size
    _px, px_, _py, py_, _pz, pz_ = px//2, px - px//2, py//2, py - py//2, pz//2, pz - pz//2    
    lx, ly, lz = loc
    image_patch = image[lx-_px:lx+px_,ly-_py:ly+py_,lz-_pz:lz+pz_]
    return image_patch


def resample(
    image, output_spacing, interpolator=sitk.sitkLinear, default_value=0
):
    """
    image : sitk Image
    """
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_size = [0.0, 0.0, 0.0]

    output_origin = image.GetOrigin()
    output_direction = image.GetDirection()

    output_size[0] = int(
        input_size[0] * input_spacing[0] / output_spacing[0] + 0.5
    )
    output_size[1] = int(
        input_size[1] * input_spacing[1] / output_spacing[1] + 0.5
    )
    output_size[2] = int(
        input_size[2] * input_spacing[2] / output_spacing[2] + 0.5
    )

    output_size = tuple(output_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputOrigin(output_origin)
    resampler.SetOutputDirection(output_direction)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    image = resampler.Execute(image)
    return image
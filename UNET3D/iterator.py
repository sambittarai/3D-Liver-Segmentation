import numpy as np
import math

class BasicVolumeIterator(object):
	"""
	This class helps you to get patches in raster scan order from
	a bigger 3D array. 
	The array dimensions have to be divisible by the patch dimensions
	"""
	def __init__(self, vol_array, patch_size, stride_size):
		self.vol_array = vol_array
		self.patch_size = patch_size
		self.stride_size = stride_size
		self.coords = [0,0,0]
		self.iter_over = False

	def get_patch(self):
		c = self.coords
		ps = self.patch_size
		patch = self.vol_array[	c[0]:c[0] + ps[0],
								c[1]:c[1] + ps[1],
								c[2]:c[2] + ps[2]]
		return patch

	def set_patch(self, patch):
		c = self.coords
		ps = self.patch_size
		self.vol_array[	c[0]:c[0] + ps[0],
						c[1]:c[1] + ps[1],
						c[2]:c[2] + ps[2]] = patch

	def move_coords(self):
		max_x, max_y, max_z = self.vol_array.shape
		ps = self.patch_size
		c = self.coords
		ss = self.stride_size
		c[2] = c[2] + ss[2]	# current patch has already been extracted
		# analyze correct coordinates for next patch
		if c[2] + ps[2] > max_z:
			c[2] = 0
			c[1] = c[1] + ss[1]
		if c[1] + ps[1] > max_y:
			c[1] = 0
			c[0] = c[0] + ss[0]
		if c[0] + ps[0] > max_x:
			self.iter_over = True

	def get_num_patches_across_dim(self):
		vol_shape = self.vol_array.shape
		n0 = math.floor((vol_shape[0] - self.patch_size[0])/self.stride_size[0] + 1)
		n1 = math.floor((vol_shape[1] - self.patch_size[1])/self.stride_size[1] + 1)
		n2 = math.floor((vol_shape[2] - self.patch_size[2])/self.stride_size[2] + 1)
		return [n0,n1,n2]

	def get_num_patches(self):
		return np.prod(self.get_num_patches_across_dim())

	def is_not_over(self):
		return (not self.iter_over)
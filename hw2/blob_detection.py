from operator import mod
import os

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use same padding (mode = 'reflect'). Refer docs for further info.
from common import find_maxima_for_cell, read_img, save_img

from common import (find_maxima, read_img, visualize_maxima,
					visualize_scale_space)
import matplotlib.pyplot as plt


def gaussian_filter(image, sigma):
	"""
	Given an image, apply a Gaussian filter with the input kernel size
	and standard deviation

	Input
	  image: image of size HxW
	  sigma: scalar standard deviation of Gaussian Kernel

	Output
	  Gaussian filtered image of size HxW
	"""
	H, W = image.shape
	# -- good heuristic way of setting kernel size
	kernel_size = int(2 * np.ceil(2 * sigma) + 1)
	# Ensure that the kernel size isn't too big and is odd
	kernel_size = min(kernel_size, min(H, W) // 2)
	if kernel_size % 2 == 0:
		kernel_size = kernel_size + 1
	# TODO implement gaussian filtering of size kernel_size x kernel_size
	# Similar to Corner detection, use scipy's convolution function.
	# Again, be consistent with the settings (mode = 'reflect').
	
	kernel = np.zeros(shape=(kernel_size,kernel_size),dtype=np.float)
	radius = kernel_size // 2
	for i in range(-radius, radius + 1):
		for j in range(-radius, radius + 1):
			v = 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-1.0 / (2 * sigma ** 2) * (i ** 2 + j ** 2))
			kernel[i + radius, j + radius] = v
	kernel = kernel / np.sum(kernel)
	output = scipy.ndimage.convolve(image, kernel, mode = 'reflect')
	return output


def main():

	image = read_img('polka.png')
	# import pdb; pdb.set_trace()
	# Create directory for polka_detections
	if not os.path.exists("./polka_detections"):
		os.makedirs("./polka_detections")

   	# -- TODO Task 3: LoG Filter vs. DoG Filter
	# (a)
	kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
	kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
							[0, 2, 3, 5, 5, 5, 3, 2, 0],
							[3, 3, 5, 3, 0, 3, 5, 3, 3],
							[2, 5, 3, -12, -23, -12, 3, 5, 2],
							[2, 5, 0, -23, -40, -23, 0, 5, 2],
							[2, 5, 3, -12, -23, -12, 3, 5, 2],
							[3, 3, 5, 3, 0, 3, 5, 3, 3],
							[0, 2, 3, 5, 5, 5, 3, 2, 0],
							[0, 0, 3, 2, 2, 2, 3, 0, 0]])
	# Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
	filtered_LoG1 = scipy.ndimage.convolve(image, kernel_LoG1, mode = 'reflect')
	filtered_LoG2 = scipy.ndimage.convolve(image, kernel_LoG2, mode = 'reflect')
	if not os.path.exists("./log_filter"):
		os.makedirs("./log_filter")
	save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
	save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

	# (b)
	# Follow instructions in pdf to approximate LoG with a DoG
	data = np.load('log1d.npz')
	plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
	plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
	x = np.linspace(-1,1,data['log50'].shape[0])
	y1 = data['log50']
	y2 = data['gauss53'] - data['gauss50']
	plt.plot(x,y1,color='red',label='LoG')
	plt.plot(x,y2,color='blue',label='DoG')

	plt.legend()
	plt.savefig('./LoG_vs_DoG.png')
	# plt.show()
	print("LoG Filter is done. ")

 
	# -- TODO Task 4: Single-scale Blob Detection --
	# (a), (b): Detecting Polka Dots
	# First, complete gaussian_filter()
	print("Detecting small polka dots")
	# -- Detect Small Circles
	sigma_1, sigma_2 = 3.6, 4
	gauss_1 = gaussian_filter(image,sigma_1)
	gauss_2 = gaussian_filter(image,sigma_2)

	# calculate difference of gaussians
	DoG_small = gauss_2 - gauss_1

	# visualize maxima
	maxima = find_maxima(DoG_small, k_xy=10)
	visualize_scale_space(DoG_small, sigma_1, sigma_2 / sigma_1,
						  './polka_detections/polka_small_DoG.png')
	visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
					 './polka_detections/polka_small.png')

	# -- Detect Large Circles
	print("Detecting large polka dots")
	sigma_1, sigma_2 = 8.5, 9
	gauss_1 = gaussian_filter(image,sigma_1)  # to implement
	gauss_2 = gaussian_filter(image,sigma_2)  # to implement

	# calculate difference of gaussians
	DoG_large = gauss_2 - gauss_1  # to implement

	# visualize maxima
	# Value of k_xy is a sugguestion; feel free to change it as you wish.
	maxima = find_maxima(DoG_large, k_xy=10)
	visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
						  './polka_detections/polka_large_DoG.png')
	visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
					 './polka_detections/polka_large.png')


	# # -- TODO Task 5: Cell Counting --
	print("Detecting cells")

	# Detect the cells in any four (or more) images from vgg_cells
	# Create directory for cell_detections
	if not os.path.exists("./cell_detections"):
		os.makedirs("./cell_detections")
	
	cell_names = ['001cell','002cell','004cell','005cell']
	for cell_name in cell_names:
		cell_image = read_img('./cells/' + cell_name + '.png')
		sigma_1, sigma_2 = 3.6, 4
		gauss_1 = gaussian_filter(cell_image,sigma_1)
		gauss_2 = gaussian_filter(cell_image,sigma_2)
		DoG_Cell = gauss_2 - gauss_1
		maxima = find_maxima_for_cell(DoG_Cell, k_xy=10)
		print(cell_name + ':'+ str(len(maxima)))
		visualize_scale_space(DoG_Cell, sigma_1, sigma_2 / sigma_1,
								'./cell_detections/' + cell_name + '_DoG.png')
		visualize_maxima(cell_image, maxima, sigma_1, sigma_2 / sigma_1,
							'./cell_detections/' + cell_name + '.png')
	



if __name__ == '__main__':
	main()

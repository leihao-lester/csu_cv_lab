"""
Task 5 Code
"""
import numpy as np
from matplotlib import pyplot as plt
from common import save_img, read_img
from homography import fit_homography, homography_transform
import os
import cv2


def make_synthetic_view(img, corners, size):
	'''
	Creates an image with a synthetic view of selected region in the image
	from the front. The region is bounded by a quadrilateral denoted by the
	corners array. The size array defines the size of the final image.

	Input - img: image file of shape (H,W,3)
		corner: array containing corners of the book cover in 
		the order [top-left, top-right, bottom-right, bottom-left]  (4,2)
		size: array containing size of book cover in inches [height, width] (1,2)

	Output - A fronto-parallel view of selected pixels (the book as if the cover is
		parallel to the image plane), using 100 pixels per inch.
	'''
	xy_comma = np.float32([[0,0],[100 * size[0][1] - 1, 0],[100 * size[0][1] - 1, 100 * size[0][0] - 1],[0, 100 * size[0][0] - 1]])# 注意对应关系
	corners = corners.astype(np.float32)
	XY = np.c_[corners,xy_comma]
	# print("XY:",XY)
	H = fit_homography(XY)
	print("H:",H)
	new_img = cv2.warpPerspective(img, H, (int(100 * size[0][1]), int(100 * size[0][0])), flags=cv2.INTER_LINEAR)
	return new_img

def send_synthetic_view_back(img, img_corner, corners):
	dst_corners = np.float32([[0,0],[img_corner.shape[1], 0],[img_corner.shape[1], img_corner.shape[0]],[0, img_corner.shape[0]]])# 注意对应关系
	XY = np.c_[dst_corners, corners]
	# print("XY:",XY)
	H = fit_homography(XY)
	new_img = cv2.warpPerspective(img_corner, H, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if new_img[i,j][0] == 0 and new_img[i,j][1] == 0 and new_img[i,j][2] == 0:
				new_img[i,j] = img[i,j]
	return new_img
	
if __name__ == "__main__":
	# Task 5

	case_name = "threebody"
	# case_name = "palmer"

	I = read_img(os.path.join("task5",case_name,"book.jpg"))
	corners = np.load(os.path.join("task5",case_name,"corners.npy"))
	size = np.load(os.path.join("task5",case_name,"size.npy"))

	result = make_synthetic_view(I, corners, size)
	save_img(result, "./result_img/task5/" + case_name + "_frontoparallel.jpg")

	case_name2 = "threebody"
	I2 = read_img(os.path.join("task5",case_name2,"book.jpg"))
	corners2 = np.load(os.path.join("task5",case_name2,"corners.npy"))
	I_corners2 = read_img("threebody_doodle_cover.jpg")
	result2 = send_synthetic_view_back(I2, I_corners2, corners2)
	save_img(result2, "./result_img/task5/" + "threebody_doodle_book.jpg")


import os
from turtle import shape

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from PIL import Image

from common import read_img, save_img


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    # np.set_printoptions(threshold=np.inf)
    output = np.zeros(shape=(image.shape[0],image.shape[1]))
    I = np.zeros(shape=(image.shape[0] + window_size[0],image.shape[1] + window_size[1]))
    I[(int)(window_size[0]/2): (int)(window_size[0]/2 + image.shape[0]), (int)(window_size[1]/2):(int)(window_size[1]/2 + image.shape[1])] = image
    I_u_v = np.zeros(shape=(I.shape[0],I.shape[1]))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if i + u >= I.shape[0] or j + v >= I.shape[1]:
                I_u_v[i,j] = 0
            else:
                I_u_v[i,j] = I[i + u, j + v]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i,j] = np.sum(np.square(I_u_v[i:(int)(i + window_size[0]),j:(int)(j + window_size[1])] - I[i:(int)(i + window_size[0]),j:(int)(j + window_size[1])]))
    return output


def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    # compute the derivatives
    alpha = 0.06
    kx = np.array([[-1,0,1]]) * 0.5
    ky = np.transpose(kx)
    Ix = scipy.ndimage.convolve(image, kx, mode = 'constant')
    Iy = scipy.ndimage.convolve(image, ky, mode = 'constant')

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # For each image location, construct the structure tensor and calculate
    # the Harris response
    k_gauss = np.ones(window_size)
    M = np.zeros((image.shape[0], image.shape[1], 3))
    M[:,:,0] = scipy.ndimage.convolve(Ixx,k_gauss, mode = 'constant')
    M[:,:,1] = scipy.ndimage.convolve(Ixy,k_gauss, mode = 'constant')
    M[:,:,2] = scipy.ndimage.convolve(Iyy,k_gauss, mode = 'constant')
    
    R =  M[:,:,0]*M[:,:,2] - M[:,:,1]**2 - alpha*((M[:,:,0]+M[:,:,2]))**2 
    response = R

    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- TODO Task 1: Corner Score --
    # (a): Complete corner_score()
    
    # (b)
    # Define offsets and window size and calulcate corner score
    u, v, W = 0, 5, (5,5)
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score1.png")

    u, v, W = 0, -5, (5,5)
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score2.png")

    u, v, W = 5, 0, (5,5)
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score3.png")

    u, v, W = -5, 0, (5,5)
    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score4.png")

    # -- TODO Task 2: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()

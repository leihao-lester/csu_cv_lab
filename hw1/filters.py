import os

import numpy as np
import cv2
from PIL import Image


def read_img(path, greyscale=True):
    img = Image.open(path)
    if greyscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    return np.array(img).astype(np.float)


def save_img(img, path):
    img = img - img.min()
    img = img / img.max()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
    print(path, "is saved!")


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """

    # TODO: Use slicing to complete the function
    cut_width = patch_size[0]
    cut_length = patch_size[1]
    (width, length) = image.shape

    pic = np.zeros((cut_width, cut_length))

    num_width = int(width / cut_width)
    num_length = int(length / cut_length)
    output = []

    for i in range(0, num_width):
        for j in range(0, num_length):
            pic = image[i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length]
            output.append(pic)

    return output


def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    (H, W) = image.shape
    kernel = np.array(kernel)
    (h, w) = kernel.shape
    kernel = kernel[::-1, ...][..., ::-1]
    h_pad = (h - 1) // 2
    w_pad = (w - 1) // 2

    image = np.pad(image, pad_width=[(h_pad, h_pad), (w_pad, w_pad)], mode="constant", constant_values=0)
    output = np.zeros(shape=(H, W))
    for i in range(H):
        for j in range(W):
            output[i, j] = np.sum(np.multiply(image[i: i + h, j: j + w], kernel))

    return output


def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = [[1], [0], [-1]]  # 1 x 3
    ky = [[1],
          [0],
          [-1]]  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(np.square(Ix)+np.square(Iy))

    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    sx=[[1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]]
    sy=[[1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]]

    # TODO: Use convolve() to complete the function
    Gx, Gy = convolve(image, sx), convolve(image, sy)
    grad_magnitude = np.sqrt(np.square(Gx)+np.square(Gy))

    return Gx, Gy, grad_magnitude




def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- TODO Task 1: Image Patches --

    # First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    for i in range(0, 3):
        chosen_patches = patches[i]
        save_img(chosen_patches, "./image_patches/q0_{}_patch.png".format(i))


    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- TODO Task 2: Convolution and Gaussian Filter --
    #  Complete convolve()
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.
    kernel_size = 3
    sigma = 0.572
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_gaussian = np.multiply(kx, np.transpose(ky))

    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # Complete edge_detection()

    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")


    # -- TODO Task 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # Complete sobel_operator()
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    

if __name__ == "__main__":
    main()

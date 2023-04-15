"""
Task 7 Code
"""
import numpy as np
import common 
from common import save_img, read_img
from homography import homography_transform, RANSAC_fit_homography
import cv2
import os

from task6 import find_matches, warp_and_combine

def improve_image(scene, template, transfer):
    '''
    Detect template image in the scene image and replace it with transfer image.

    Input - scene: image (H,W,3)
            template: image (K,K,3)
            transfer: image (L,L,3)
    Output - augment: the image with 
    
    Hints:
    a) You may assume that the template and transfer are both squares.
    b) This will work better if you find a nearest neighbor for every template
       keypoint as opposed to the opposite, but be careful about directions of the
       estimated homography and warping!
    '''
    
    kp1, desc1 = common.get_AKAZE(scene)
    kp2, desc2 = common.get_AKAZE(template)
    matches = find_matches(desc1, desc2, 0.75)
    XY = np.hstack((kp1[matches[:, [0]], [0]], kp1[matches[:, [0]], [1]]))
    XY = np.hstack((XY, kp2[matches[:,[1]], [0]]))
    XY = np.hstack((XY, kp2[matches[:, [1]], [1]]))
    print(XY.shape)
    H = RANSAC_fit_homography(XY)

    pts1 = np.float32([[0,0], [transfer.shape[0],0], [transfer.shape[0],transfer.shape[0]], [0,transfer.shape[0]]])
    pts2 = np.float32([[0,0], [template.shape[0],0], [template.shape[0],template.shape[0]], [0,template.shape[0]]])
    H_tran_to_temp = cv2.getPerspectiveTransform(pts1, pts2)
    transfer = cv2.warpPerspective(transfer, H_tran_to_temp, (template.shape[0], template.shape[1]))
    augment = warp_and_combine(scene, transfer, H)
    return augment


if __name__ == "__main__":
    # Task 7

    scenes_list = ['bbb', 'florence', 'lacroix']
    seals_list = ['michigan.png','monk.png','um.png']
    scenes_name = 'lacroix'
    seals_name = 'michigan.png'

    img = read_img(os.path.join('task7', 'scenes', scenes_name, 'scene.jpg'))
    template_img = read_img(os.path.join('task7', 'scenes', scenes_name, 'template.png'))
    tranfer_img = read_img(os.path.join('task7', 'seals', seals_name))
    res = improve_image(img, template_img, tranfer_img)
    save_img(res, os.path.join('result_img', 'task7', scenes_name + '_' + seals_name))


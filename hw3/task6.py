"""
Task6 Code
"""
import numpy as np
import common 
from common import save_img, read_img
from homography import fit_homography, homography_transform, RANSAC_fit_homography
import os
import cv2

def compute_distance(desc1, desc2):
    '''
    Calculates L2 distance between 2 binary descriptor vectors.
        
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
    
    Output - dist: a (N,M) L2 distance matrix where dist(i,j)
             is the squared Euclidean distance between row i of 
             desc1 and desc2. You may want to use the distance
             calculation trick
             ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
    '''

    a = (np.linalg.norm(desc1, axis = 1, keepdims = True))**2
    b = ((np.linalg.norm(desc2, axis = 1, keepdims = True))**2).T
    zeros = np.zeros((desc1.shape[0], desc2.shape[0]))
    dist = np.sqrt(np.maximum(zeros, a + b - 2 * np.dot(desc1,desc2.T)))  # 注意是减去两者乘积
    return dist

def find_matches(desc1, desc2, ratioThreshold):
    '''
    Calculates the matches between the two sets of keypoint
    descriptors based on distance and ratio test.
    
    Input - desc1: Descriptor vector of shape (N,F)
            desc2: Descriptor vector of shape (M,F)
            ratioThreshhold : maximum acceptable distance ratio between 2
                              nearest matches
    
    Output - matches: a list of indices (i,j) 1 <= i <= N, 1 <= j <= M giving
             the matches between desc1 and desc2.
             
             This should be of size (K,2) where K is the number of 
             matches and the row [ii,jj] should appear if desc1[ii,:] and 
             desc2[jj,:] match.
    '''
    matches = []
    dist = compute_distance(desc1, desc2)
    dist_sort_index = np.argsort(dist, axis=1)
    for i in range(dist.shape[0]):
        nearest = dist_sort_index[i,0]
        secend_near = dist_sort_index[i,1]
        ratio = (dist[i, nearest]**2) / (dist[i, secend_near]**2)
        if ratio < ratioThreshold:
            matches.append([i, nearest])
    matches = np.array(matches)
    return matches

def draw_matches(img1, img2, kp1, kp2, matches):
    '''
    Creates an output image where the two source images stacked vertically
    connecting matching keypoints with a line. 
        
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            kp1: Keypoint matrix for image 1 of shape (N,4)
            kp2: Keypoint matrix for image 2 of shape (M,4)
            matches: List of matching pairs indices between the 2 sets of 
                     keypoints (K,2)
    
    Output - Image where 2 input images stacked vertically with lines joining 
             the matched keypoints

    Hint: see cv2.line
    '''
    #Hint:
    #Use common.get_match_points() to extract keypoint locations

    match_points = common.get_match_points(kp1, kp2, matches).astype(np.int32)
    img = np.vstack([img1,img2])
    color = (0, 0, 255) # bgr
    thickness = 1
    H1 = img1.shape[0]
    for i in range(match_points.shape[0]):
        start_point = (match_points[i, 0], match_points[i, 1])
        end_point = (match_points[i, 2], match_points[i, 3] + H1)
        cv2.line(img, start_point, end_point, color, thickness)
    output = img
    file_name = './result_img/task6/draw_matches_test.png'
    save_img(output, file_name)
    return output


def warp_and_combine(img1, img2, H):
    '''
    You may want to write a function that merges the two images together given
    the two images and a homography: once you have the homography you do not
    need the correspondences; you just need the homography.

    Writing a function like this is entirely optional, but may reduce the chance
    of having a bug where your homography estimation and warping code have odd
    interactions.
    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 2 of shape (H2,W2,3)
            H: homography mapping betwen them

    Output - V: stitched image of size (?,?,3); unknown since it depends on H
    '''

    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]
    corners1 = np.array([[0, 0], [W1, 0], [W1, H1], [0, H1]])
    corners2 = np.array([[0, 0], [W2, 0], [W2, H2], [0, H2]])
    corners2_in_1 = homography_transform(corners2, np.linalg.inv(H))
    corners1 = np.vstack([corners1, corners2_in_1])
    print("corners1:",corners1)
    final_w = int(max(corners1[:, 0])) - int(min(corners1[:, 0]))
    final_h = int(max(corners1[:, 1])) - int(min(corners1[:, 1]))
    warp1 = cv2.warpPerspective(img1, np.eye(3), (final_w, final_h), flags=cv2.INTER_LINEAR)
    warp2 = cv2.warpPerspective(img2, np.linalg.inv(H), (final_w, final_h), flags=cv2.INTER_LINEAR)
    mask1 = cv2.warpPerspective(np.ones(img1.shape), np.eye(3), (final_w, final_h), flags=cv2.INTER_LINEAR)
    mask2 = cv2.warpPerspective(np.ones(img2.shape), np.linalg.inv(H), (final_w, final_h), flags=cv2.INTER_LINEAR)

    V = (warp1 * mask1 + warp2 * mask2) / (mask1 + mask2)
    return V


def make_warped(img1, img2):
    '''
    Take two images and return an image, putting together the full pipeline.
    You should return an image of the panorama put together.

    
    Input - img1: Input image 1 of shape (H1,W1,3)
            img2: Input image 1 of shape (H2,W2,3)
    
    Output - Final stitched image

    Be careful about:
    a) The final image size 
    b) Writing code so that you first estimate H and then merge images with H.
    The system can fail to work due to either failing to find the homography or
    failing to merge things correctly.
    '''
    
    kp1, desc1 = common.get_AKAZE(img1)
    kp2, desc2 = common.get_AKAZE(img2)
    matches = find_matches(desc1, desc2, 0.75)
    draw_matches(img1, img2, kp1, kp2, matches)
    XY = np.hstack((kp1[matches[:, [0]], [0]], kp1[matches[:, [0]], [1]]))
    XY = np.hstack((XY, kp2[matches[:,[1]], [0]]))
    XY = np.hstack((XY, kp2[matches[:, [1]], [1]]))
    print(XY.shape)
    H = RANSAC_fit_homography(XY)

    stitched = warp_and_combine(img1, img2, H)
    return stitched 


if __name__ == "__main__":

    #Possible starter code; you might want to loop over the task 6 images
    to_stitch_list = ['eynsham', 'florence2', 'florence3', 'florence3_alt', 'lowetag', 'mertonchapel', 'mertoncourtyard', 'vgg']
    for to_stitch in to_stitch_list:
        I1 = read_img(os.path.join('task6',to_stitch,'p1.jpg'))
        I2 = read_img(os.path.join('task6',to_stitch,'p2.jpg'))
        res = make_warped(I1,I2)
        save_img(res,"./result_img/task6/" + "result_"+to_stitch+".jpg")
        print("finish ", to_stitch)
        break

    

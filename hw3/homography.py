"""
Homography fitting functions
You should write these
"""
import numpy as np
from common import homography_transform
import matplotlib.pyplot as plt
import scipy.linalg as linalg


def fit_homography(XY):
	'''
	Given a set of N correspondences XY of the form [x,y,x',y'],
	fit a homography from [x,y,1] to [x',y',1].
	
	Input - XY: an array with size(N,4), each row contains two
			points in the form [x_i, y_i, x'_i, y'_i] (1,4)
	Output -H: a (3,3) homography matrix that (if the correspondences can be
			described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T

	'''

	A = np.zeros(shape=(XY.shape[0] * 2,9),dtype=np.float32)
	for i in range(XY.shape[0]):
		A[2 * i,0] = -1.0 * XY[i,0]
		A[2 * i,1] = -1.0 * XY[i,1]
		A[2 * i,2] = -1
		A[2 * i,6] = XY[i,0] * XY[i,2]
		A[2 * i,7] = XY[i,1] * XY[i,2]
		A[2 * i,8] = XY[i,2]
		A[2 * i + 1,3] = -1.0 * XY[i,0]
		A[2 * i + 1,4] = -1.0 * XY[i,1]
		A[2 * i + 1,5] = -1
		A[2 * i + 1,6] = XY[i,0] * XY[i,3]
		A[2 * i + 1,7] = XY[i,1] * XY[i,3]
		A[2 * i + 1,8] = XY[i,3]
	_, _, vt = linalg.svd(A)
	H = vt[-1].reshape(3,3) # 注意是-1行，不是-1列
	H = H / H[2,2]
	return H 

def random_points_pick(XY):
	'''
	randomly pick 4 data sets in XY
	'''
	k = 4
	output = np.array([XY[np.random.randint(0, XY.shape[0])] for i in range(k)])
	return output

def points_distance(XY, H):
	'''
	compute distance
	'''
	p = XY[:, 0:2].copy()
	p_comma = XY[:, 2:].copy()
	T = homography_transform(p, H)
	dist = np.linalg.norm(p_comma - T, axis=1)
	return dist

def RANSAC_fit_homography(XY, eps=1, nIters=1000):
	'''
	Perform RANSAC to find the homography transformation 
	matrix which has the most inliers
		
	Input - XY: an array with size(N,4), each row contains two
			points in the form [x_i, y_i, x'_i, y'_i] (1,4)
			eps: threshold distance for inlier calculation
			nIters: number of iteration for running RANSAC
	Output - bestH: a (3,3) homography matrix fit to the 
					inliers from the best model.

	Hints:
	a) Sample without replacement. Otherwise you risk picking a set of points
	   that have a duplicate.
	b) *Re-fit* the homography after you have found the best inliers
	'''
	bestH, bestCount, bestInliers = np.eye(3), -1, np.zeros((XY.shape[0],))
	bestRefit = np.eye(3)
	for i in range(nIters):
		subset = random_points_pick(XY)
		H = fit_homography(subset)
		E = points_distance(XY, H)
		inliers = (E < eps)
		inliers_num = np.count_nonzero(inliers)
		if inliers_num > bestCount:
			bestH = H
			bestCount = inliers_num
			bestInliers = inliers
	refit_set = XY[bestInliers]
	bestRefit = fit_homography(refit_set)
	return bestRefit

if __name__ == "__main__":
	#If you want to test your homography, you may want write any code here, safely
	#enclosed by a if __name__ == "__main__": . This will ensure that if you import
	#the code, you don't run your test code too
	file = '/task4/points_case_5'
	data = np.load('.' + file + '.npy')

	plt.clf()
	plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
	plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
	xy = data[:,0:2]
	xy_pie = data[:,2:]
	A = np.c_[xy,np.ones(len(xy))]
	H = fit_homography(data)
	print(H)
	xy_pred = np.dot(H, A.T).T
	xy_pred[:,0] = xy_pred[:,0] / xy_pred[:,2]
	xy_pred[:,1] = xy_pred[:,1] / xy_pred[:,2]
	plt.scatter(xy[:,0], xy[:,1], 1, color='orange', label='x,y')
	plt.scatter(xy_pie[:,0], xy_pie[:,1], 1, color='blue', label='x\',y\'')
	plt.scatter(xy_pred[:,0], xy_pred[:,1], 1, color='red', label='x_pred,y_pred')
	plt.legend()
	plt.savefig('./result_img' + file + '.png')
	# plt.show()


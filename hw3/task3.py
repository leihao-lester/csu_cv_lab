import numpy as np
import matplotlib.pyplot as plt


def plot(data,file):
    plt.clf()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    xy = data[:,0:2]
    xy_pie = data[:,2:]
    A = np.c_[xy,np.ones(len(xy))]
    S = np.linalg.lstsq(A, xy_pie,rcond=None)[0]
    print(S)
    xy_pred = np.dot(A, S)
    plt.scatter(xy[:,0], xy[:,1], 1, color='orange', label='x,y')
    plt.scatter(xy_pie[:,0], xy_pie[:,1], 1, color='blue', label='x\',y\'')
    plt.scatter(xy_pred[:,0], xy_pred[:,1], 1, color='red', label='x_pred,y_pred')

    plt.legend()
    plt.savefig('./result_img' + file + '.png')
    # plt.show()

file1 = '/task3/points_case_1'
file2 = '/task3/points_case_2'
data1 = np.load('.' + file1 + '.npy')
data2 = np.load('.' + file2 + '.npy')

# print(data1.shape)
# print(data2.shape)
plot(data1,file1)
plot(data2,file2)




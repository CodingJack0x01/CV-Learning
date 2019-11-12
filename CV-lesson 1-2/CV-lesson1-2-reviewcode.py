#%%

import cv2
import numpy as np
from matplotlib import pyplot as plt

#%%
#使用plt 替代cv显示图片
def myImgShow4cv(img):
    B, G, R = cv2.split(img)
    img_rgb = cv2.merge((R, G, B))
    plt.imshow(img_rgb)
    plt.show()

#%%
img = cv2.imread('lenna.jpg')
cv2.imshow('lenna',img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
myImgShow4cv(img)
#%% md

### Gaussian kernel 高斯核 高斯滤波

#%%
#（图片，高斯核尺寸，在X方向的标准差）
g_img = cv2.GaussianBlur(img, (7,7), 2)
# cv2.imshow('lenna',img)
# cv2.imshow('g_lenna',g_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
myImgShow4cv(img)
myImgShow4cv(g_img)

#%%
#（卷积核的尺寸，标准差）
kernel_1d = cv2.getGaussianKernel(7, 2)

#%%

g1_img = cv2.sepFilter2D(img,-1,kernel_1d,kernel_1d)
# cv2.imshow('g1_lenna',g_img)
# cv2.imshow('g2_lenna',g1_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
myImgShow4cv(img)
myImgShow4cv(g_img)

#%%

g_img == g1_img

#%% md

### laplacian   Laplacian算子对噪声比较敏感

#%%

kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
lap_img = cv2.filter2D(img, -1, kernel)
# cv2.imshow('lapl_lenna',lap_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
myImgShow4cv(lap_img)
#%% md

### sobel   X方向上的sobel滤波

#%%

x_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sx_img = cv2.filter2D(img, -1, x_kernel)
# cv2.imshow('sobelx_lenna',sx_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
myImgShow4cv(sx_img)

#%%
#X方向上的sobel滤波
y_kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
sy_img = cv2.filter2D(img, -1, y_kernel)
# cv2.imshow('sobely_lenna',sy_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
myImgShow4cv(sy_img)

#%%

# cv2.imshow('sobelx_lenna',sx_img)
# cv2.imshow('sobely_lenna',sy_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()

#%% md

### medianblur 中值滤波

#%%

md_img = cv2.medianBlur(img, 7)
# cv2.imshow('lenna',img)
# cv2.imshow('md_lenna',md_img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
myImgShow4cv(img)
myImgShow4cv(md_img)
#%%

img.shape

#%%

md_img.shape

#%%

g1_img.shape


#%% md

### harris corner  harris角检测

#%%
img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#(图片，邻域大小,sobel核孔径参数。harris检测自由参数？)
img_harris = cv2.cornerHarris(img_gray, 2, 3, 0.03)
print(img_harris.shape)
#print(img_harris[0])
# cv2.imshow('harris',img_harris)
# key = cv2.waitKey()
# if key == 27:
#    cv2.destroyAllWindows()
plt.imshow(img_gray,cmap = plt.cm.gray)
plt.show()

plt.imshow(img_harris,cmap = plt.cm.gray)
plt.show()
#%%
print(np.max(img_harris))
threshold = np.max(img_harris)*0.03
print(threshold)

#%%

#dilate 对像素点进行膨胀，小的点放大成大的点
img_harris = cv2.dilate(img_harris, None)
plt.imshow(img_harris,cmap = plt.cm.gray)
plt.show()
#%%
#对harris处理的图中对超过最大值0.03的点，对应到原图中变成纯蓝色显示
img[img_harris>threshold] = [0,0,255]

#%%

# cv2.imshow('harris',img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
myImgShow4cv(img)
#%% md

### SIFT

#%%

img = cv2.imread('lenna.jpg')
#实例化
sift = cv2.xfeatures2d.SIFT_create()
#检测出图像中的关键点
kp = sift.detect(img,None)
print(kp)

#%%

len(kp)

#%%
#计算关键点对应的sift特征向量
kp,des = sift.compute(img,kp)
print(des.shape)

#%%

img_sift = cv2.drawKeypoints(img,kp,outImage=np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('harris',img_sift)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()
myImgShow4cv(img_sift)
#%%



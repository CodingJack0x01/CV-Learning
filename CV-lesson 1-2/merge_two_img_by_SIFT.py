# %%
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

#%%   读取图片
imgA = cv2.imread('testimg/imgA.png')
imgB = cv2.imread('testimg/imgB.png')
myImgShow4cv(imgA)
myImgShow4cv(imgB)

#%%
#实例化
sift = cv2.xfeatures2d.SIFT_create()
#检测出图像中的关键点
kp_A = sift.detect(imgA)
kp_B = sift.detect(imgB)
print(kp_A)
print(kp_B)

#%%
len(kp_A)
#%%
len(kp_B)

#%%
#计算关键点对应的sift特征向量
kp_A,features_A = sift.compute(imgA,kp_A)
print(features_A.shape)
#计算关键点对应的sift特征向量
kp_B,features_B = sift.compute(imgB,kp_B)
print(features_B.shape)

#%%
#打印特征点
img_sift_A = cv2.drawKeypoints(imgA,kp_A,outImage=np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_sift_B = cv2.drawKeypoints(imgB,kp_B,outImage=np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
myImgShow4cv(img_sift_A)
myImgShow4cv(img_sift_B)
#%%
#匹配特征点
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # 建立暴力匹配器
    matcher = cv2.DescriptorMatcher_create("BruteForce")

    # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    matches = []
    for m in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # 存储两个点在featuresA, featuresB中的索引值
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # 当筛选后的匹配对大于4时，计算视角变换矩阵
    if len(matches) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
        ptsB = np.float32([kpsB[i].pt for (i, _) in matches])

        # 计算视角变换矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        # 返回结果
        return (matches, H, status)

    # 如果匹配对小于4时，返回None
    return None
#%%
def drawMatches( imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int((kpsA[queryIdx].pt)[0]), int((kpsA[queryIdx].pt)[1]))
                ptB = (int((kpsB[trainIdx].pt)[0]) + wA, int((kpsB[trainIdx].pt)[1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis
#%%
# 匹配两张图片的所有特征点，返回匹配结果
ratio=0.75
reprojThresh=4.0
showMatches=False
M = matchKeypoints(kp_A, kp_B, features_A, features_B, ratio, reprojThresh)
#%%
# H是3x3视角变换矩阵
(matches, H, status) = M
print(status)
# 将图片A进行视角变换，result是变换后图片
result = cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
# 将图片B传入result图片最左端
result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
vis = drawMatches(imgA, imgB, kp_A, kp_B, matches, status)
myImgShow4cv(vis )
myImgShow4cv(result)
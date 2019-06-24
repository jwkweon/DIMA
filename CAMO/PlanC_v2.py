#랜덤한 카모 이미지 선택 => 그레이로 바꾸기 => 4가지 색 추출 => 입력 이미지로 부터 4가지 색 추 =>
#추출된 4가지 색으로 부터 색 칠하dd

#import kmean
import cv2
import numpy as np
import os
from scipy.misc import imread, imsave
import argparse
from sklearn.cluster import KMeans


def random_crop(img, min, max):
    randint = np.random.randint(min, max, size=2)
    x, y = randint[0], randint[1]#:::random 으로 바꾸기 -> 완료
    crop_img = img[x : x + 256 , y : y + 256]
    return crop_img

def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar

def image_color_cluster(image, k = 4):
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    color_list = []

    clt = KMeans(n_clusters = k)
    clt.fit(image)

    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    for i in range (300):
        if i == 0:
            temp_R=bar[0][i][0]
            temp_G=bar[0][i][1]
            temp_B=bar[0][i][2]
            #print(bar[0][i][:])
            color_list.append(bar[0][i][:])
        if temp_R != bar[0][i][0] or temp_G != bar[0][i][1] or temp_B != bar[0][i][2] and i != 0:
            temp_R=bar[0][i][0]
            temp_G=bar[0][i][1]
            temp_B=bar[0][i][2]
            #print(bar[0][i][:])
            color_list.append(bar[0][i][:])

    return color_list

def color2gray(img):
    img_Gray = img.copy()
    gray_img = img.sum(axis = 2) / 3


    for i in range(256):
        for j in range(256):
            for k in range(3):
                img_Gray[i, j, k] = gray_img[i, j]

    return img_Gray

def rgb_mapping(img, c_list, g_list):
    recons_img = img.copy()
    k = 0
    for i in range(256):
        for j in range(256):
            temp = []
            for n in range(4):
                if recons_img[i, j, k] >= g_list[n][k] :
                    dif = abs(recons_img[i, j, k] - g_list[n][k])
                else:
                    dif = abs(g_list[n][k] - recons_img[i, j, k])
                temp.append(dif)

            for l, m in enumerate(temp):
                if m == min(temp):
                    recons_img[i, j, :] = c_list[l]
                    break

    return recons_img

def main():
    image_path = "./CAMO/DESSERT/2.16.jpg"
    #image_path = "./kk.jpg"
    img = imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = random_crop(img, 0, 400)

    color_img_path = "./forest.jpg"
    color_img = imread(color_img_path)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_img = random_crop(color_img, 0, 500)

    color_list = image_color_cluster(color_img)
    img_Gray = color2gray(img)
    gray_color_list = image_color_cluster(img_Gray)

    result_img = rgb_mapping(img_Gray, color_list, gray_color_list)

    cv2.imshow('iddd',result_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

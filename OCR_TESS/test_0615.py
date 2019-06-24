# Contour 적용해 이미지(문서) 내의 단어가 들어있는 영역 획득을 위한 전처리 기법
# Color 정보를 없애는 과정과 AdaptThresh 적용 이후 Contour 획득 예정

import cv2
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_histogram(img):
    h = np.zeros((img.shape[0], 256), dtype=np.uint8)

    hist_item = cv2.calcHist([img],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0+10),(x,y+10),(255,255,255))

    cv2.line(h, (0, 0 + 10), (0, 5), (255, 255, 255) )
    cv2.line(h, (255, 0 + 10), (255, 5), (255, 255, 255))
    y = np.flipud(h)

    return y

def Morphology(img):
    kernel = np.ones((5, 5), np.uint8)        #모폴로지 커널

    reverse_img = cv2.bitwise_not(img)
    result = cv2.morphologyEx(reverse_img, cv2.MORPH_CLOSE, kernel)

    return result

def histogram(img, g_img):
    hist1 = cv2.calcHist([g_img], [0], None, [256], [0, 256])
    hist2, bins = np.histogram(g_img.ravel(), 256, [0, 256])
    hist3 = np.bincount(g_img.ravel(), minlength=256)

    plt.hist(g_img.ravel(), 256, [0, 256])
    plt.plot(hist1)
    plt.xlim([0, 256])
    #color = ('b', 'g', 'r')
    #for i, col in enumerate(color):
    #    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    #    plt.plot(hist, color=col)
    #    plt.xlim([0, 256])

    plt.show()

def main():
    bins = np.arange(256).reshape(256,1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required = True)
    args = parser.parse_args()
    filename = args.filename

    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img1 = img_gray.copy()
    img2 = img.copy()

    #히스토그램
    #histogram(img[250], gray[250])

    height, width, _ = img.shape
    #thr_w = width // 20
    thr_size = max(height, width) // 10
    print(height, width, thr_size)

    #threshold를 이용하여 binary image로 변환
    #ret, thresh = cv2.threshold(img_gray,127,255,0)
    img1 = cv2.fastNlMeansDenoising(img_gray, None, 15, 7, 21)
    thr1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 20)

    morph_img = Morphology(thr1)
    thr1 = cv2.bitwise_not(morph_img)


    #contours는 point의 list형태. 예제에서는 사각형이 하나의 contours line을 구성하기 때문에 len(contours) = 1. 값은 사각형의 꼭지점 좌표.
    #hierachy는 contours line의 계층 구조
    contours, hierachy = cv2.findContours(thr1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #image = cv2.drawContours(img, contours, -1, (0,255,0), 3)

    thr1 = cv2.cvtColor(thr1, cv2.COLOR_GRAY2BGR)

    for i in range(len(contours)):
        cnt = contours[i]
        #hull = cv2.convexHull(cnt)
        #img1 = cv2.drawContours(img1, [hull], 0,(0,255,0), 3)
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h <= max(thr_size ** 2, 10000) and w * h >= 25:
            img2 = cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2) #마지막 파라미터 2(default), -1이면 채우기
            #img2 = cv2.line(img2, (x + (w // 2), y + (h // 2)), (x + (w // 2), y + (h // 2)), (0, 0, 255), 5) #중앙값찍기(빨강)
            #cv2.imwrite('color_{}.jpg'.format(i), img2)    #모든과정 저장
            #img2 = cv2.drawContours(thr1, [hull], 0,(0,255,0), 3)


    #print(M.items())
    #print(contours[0].shape, len(contours))
    #print(cx, cy, height, width)
    #print(cv2.contourArea(cnt))     #면적 m00 값이거나 contourArea 함수로 구할 수 있음
    #print(cv2.arcLength(cnt, True), cv2.arcLength(cnt, False))

    #epsilon1 = 0.01*cv2.arcLength(cnt, True)
    #epsilon2 = 0.1*cv2.arcLength(cnt, True)

    #approx1 = cv2.approxPolyDP(cnt, epsilon1, True)
    #approx2 = cv2.approxPolyDP(cnt, epsilon2, True)

    cv2.drawContours(img1, [cnt],0,(0,255,0),3) # 215개의 Point
    #cv2.drawContours(img1, [approx1], 0,(0,255,0), 3) # 21개의 Point
    #cv2.drawContours(img2, [approx2], 0,(0,255,0), 3) # 4개의 Point

    #이미지 resize : 윈도우에서 보기 편함을 위해
    height_ratio = 1200

    if height >= height_ratio and height_ratio != height:
        resize_width = (width * height_ratio) // height
        height, width = height_ratio, resize_width

        img_resize = cv2.resize(img2 , (width , height))
    else:
        img_resize = img2


    cv2.imshow('image', img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

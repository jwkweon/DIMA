#영역 분할
#빨간색 영역으로 분할
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pytesseract

def Morphology(img):
    kernel = np.ones((5, 5), np.uint8)        #모폴로지 커널

    reverse_img = cv2.bitwise_not(img)
    result = cv2.morphologyEx(reverse_img, cv2.MORPH_CLOSE, kernel)

    return result

def find_roi(img, img2):
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    height, width = result.shape
    sum = []
    for i in range(height):
        #print(result[i].shape)
        s = 0
        for j in result[i]:
            s += j
        sum.append(s)

    #result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    for i in range(height):
        if sum[i] >= (width * 12):
            result = cv2.line(img2, (0, i), (width, i), (0, 0, 0), 1)
    return result

def cut_roi(img, img_ori):
    img_cp = img.copy()

    img_cp_gray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    img_cp2 = img_cp_gray.copy()
    morph_img = Morphology(img_cp2)
    bw_img = cv2.bitwise_not(morph_img)
    contours, hierachy = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        #result = cv2.rectangle(img_ori, (x, y), (x + w, y + h), (0, 255, 0), -1)
        result2 = cv2.line(img_ori, (0, y + (h // 2)), (w, y + (h // 2)), (0, 0, 255), 3)

    return result2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required = True)
    args = parser.parse_args()
    filename = args.filename

    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img1 = img_gray.copy()
    img2 = img.copy()
    img_final = img.copy()

    height, width, _ = img.shape
    img3 = np.ones_like(img)
    print('img3.shape = ', img3.shape)
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
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h <= max(thr_size ** 2, 10000) and w * h >= 25:
            img3 = cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), -1) #마지막 파라미터 2(default), -1이면 채우기
            #img2 = cv2.line(img2, (x + (w // 2), y + (h // 2)), (x + (w // 2), y + (h // 2)), (0, 0, 255), 5) #중앙값찍기(빨강)
            #cv2.imwrite('color_{}.jpg'.format(i), img2)    #모든과정 저장
            #img2 = cv2.drawContours(thr1, [hull], 0,(0,255,0), 3)

    #cv2.drawContours(img1, [cnt],0,(0,255,0),3) # 215개의 Point

    #글자가있는 곳 빨간 박스로 채움 (색변경가능)
    img3 = find_roi(img3, img2)
    #컨투어 계산을 다시해 빨간 영역과 흰색 영역을 분리해 흰색 영역에서 중앙값을 기준으로 분할지점 나누기

    img_final = cut_roi(img3, img_final)

    #이미지 resize : 윈도우에서 보기 편함을 위해
    height_ratio = 1200

    if height >= height_ratio and height_ratio != height:
        resize_width = (width * height_ratio) // height
        height, width = height_ratio, resize_width

        img_resize = cv2.resize(img_final , (width , height))
    else:
        img_resize = img_final

    cv2.imshow('image', img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

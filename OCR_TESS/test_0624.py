#영역 분할 및 저장
#영역 분할 및 tesseract 이후 저장
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

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
        s = 0
        for j in result[i]:
            s += j
        sum.append(s)

    for i in range(height):
        if sum[i] >= (width * 12):
            result = cv2.line(img2, (0, i), (width, i), (0, 0, 0), 1)
    return result

def cut_roi(img, img_ori, filename):
    img_cp = img.copy()
    height, width, _ = img.shape

    img_cp_gray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    img_cp2 = img_cp_gray.copy()
    morph_img = Morphology(img_cp2)
    bw_img = cv2.bitwise_not(morph_img)
    contours, hierachy = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    li_YH = [0]
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        li_YH.append(y + (h // 2))
        #result = cv2.rectangle(img_ori, (x, y), (x + width, y + h), (0, 255, 0), -1)
        result2 = cv2.line(img_ori, (0, y + (h // 2)), (w, y + (h // 2)), (0, 0, 255), 3)

    li_YH.append(height)
    li_YH.sort()

    for i in range(len(li_YH) - 1):
        if li_YH[i+1] - li_YH[i] > 0:
            cut_img = img_ori[li_YH[i]:li_YH[i+1], 0:width]
            cv2.imwrite('cut_{}_{}'.format(i, filename), cut_img)

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

    thr_size = max(height, width) // 10
    print(height, width, thr_size)

    img1 = cv2.fastNlMeansDenoising(img_gray, None, 15, 7, 21)
    thr1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 20)

    morph_img = Morphology(thr1)    #검정배경에 흰글씨
    thr1 = cv2.bitwise_not(morph_img)   #반전

    contours, hierachy = cv2.findContours(thr1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    thr1 = cv2.cvtColor(thr1, cv2.COLOR_GRAY2BGR)

    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h <= max(thr_size ** 2, 10000) and w * h >= 25:
            img3 = cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), -1) #마지막 파라미터 2(default), -1이면 채우기

    #글자가있는 곳 빨간 박스로 채움 (색변경가능)
    img3 = find_roi(img3, img2)
    #컨투어 계산을 다시해 빨간 영역과 흰색 영역을 분리해 흰색 영역에서 중앙값을 기준으로 분할지점 나누기

    img_final = cut_roi(img3, img_final, filename=filename)

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

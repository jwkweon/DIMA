import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pytesseract
import pandas as pd


def nothing(x):
    pass

def func_thr(img, window_name, file_name):
    ############ rotated
    height, width = img.shape

    img = cv2.bitwise_not(img)
    ret,thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
    	angle = -(90 + angle)
    else:
    	angle = -angle

    # rotate the image to deskew it
    #(h, w) = save_img.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (width, height),
    	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    print("[INFO] angle: {:.3f}".format(angle))

    rotated = cv2.bitwise_not(rotated)
    #cv2.imshow('rotated', rotated)
    #cv2.waitKey(0)

    cv2.imwrite('r_{}_{}'.format(int(angle), file_name), rotated)

    threshold_name = 'Ths'
    cv2.createTrackbar(threshold_name, window_name, 0, 255, nothing)
    cv2.setTrackbarPos(threshold_name, window_name, 140)


    while (1):
        height, width = img.shape
        ths = cv2.getTrackbarPos(threshold_name, window_name)
        ret, gray = cv2.threshold(rotated, ths, 255, cv2.THRESH_BINARY)
        save_img = gray.copy()



        height_ratio = 1200

        if height >= height_ratio and height_ratio != height:
            resize_width = (width * height_ratio) // height
            height, width = height_ratio, resize_width

            img_resize = cv2.resize(gray , (width , height))
        else:
            img_resize = gray

        img_resize = cv2.fastNlMeansDenoising(img_resize, None, 15, 7, 21)
        cv2.imshow(window_name, img_resize)

        if cv2.waitKey(30) & 0xFF == 27:
            cv2.imwrite('t_{}_{}'.format(ths, file_name), save_img)
            break
    #save_img = threshold 된 이미지

    ### rotated 가 최종
    dataframe = pytesseract.image_to_data(save_img, lang = 'kor3+eng', output_type = Output.DATAFRAME, config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")

    list_dataframe = dataframe_to_list(data_frame = dataframe)
    removed = df_list_removeNan(list_dataframe)
    topNheight_list = dflist_roi(removed)
    print(removed)

    cut_roi(img = save_img, axis_list = topNheight_list, file_name=file_name)


class Output:
    BYTES = 'bytes'
    DATAFRAME = 'data.frame'
    DICT = 'dict'
    STRING = 'string'

def dataframe_to_list(data_frame):
    Row_list =[]
    df = data_frame

    for i in range((df.shape[0])):
        Row_list.append(list(df.iloc[i, :]))

    return Row_list

#-1값 (Nan) 과 conf 95에 공백인 문자열 제거
def df_list_removeNan(df_list):
    result_list = []

    for i, j in enumerate(df_list):
        if j[10] != -1:
            if j[10] == 95 and j[11] == ' ': #(j[5] * j[4] * j[3]) == 1:
                continue
            result_list.append(j)

    return result_list

#각 라인별 문자의 수를 통해 top, height의 평균과 x 초기값 return 형식 (x_init, top_avg, height_avg)
def dflist_roi(df_list):
    result_list = [[0, 0, 0]]
    top, height, cnt, x_init = 0, 0, 1, 0

    for i, j in enumerate(df_list):
        if df_list[i][5] == 1 and df_list[i][11] != ' ':
            if (height // cnt) >= 10:
                result_list.append([df_list[i - x_init][6], top // cnt, height // cnt])
            top = df_list[i][7]
            height = df_list[i][9]
            cnt = 1
        else:
            x_init = df_list[i][5]
            if df_list[i][6] != 0 and df_list[i][7] != 0:
                cnt += 1
                top += df_list[i][7]
                height += df_list[i][9]

    return result_list

#cv2.line 부분을 지우고 h_std의 요소값(cut 할 height)별로 자르는 코드 추가필요
def cut_roi(img, axis_list,file_name):
    height, width = img.shape

    x_set = [i[0] for i in axis_list]
    x_set = sorted(x_set)
    x_std = sum(x_set[0:6]) // 6

    axis = axis_list[:]
    for i, j in enumerate(axis):
        j[2] = j[1] + j[2]

    h_std = []

    for i in range(1, len(axis)):
        cut_h = (axis[i][1] + axis[i - 1][2]) // 2
        h_std.append(cut_h)
        result = cv2.line(img, (0, cut_h), (width, cut_h), (0, 0, 0), 1)
    cv2.imshow('result', result)
    cv2.waitKey(0)

    h_std.sort()
    #result = img.copy()
    for i in range(len(h_std)-1):
        if h_std[i+1] - h_std[i] > 0:
            result = img[h_std[i]:h_std[i+1], 0:width]
            cv2.imwrite('c_{}_{}'.format(i,file_name), result)

#cut해서 들어온 항목들 결과 리스트에 추가
def add_result():
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required = True)
    parser.add_argument("--hsize", required = False)
    args = parser.parse_args()
    filename = args.filename

    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img1 = img_gray.copy()
    height, width, _ = img.shape

    if args.hsize == None:
        height_ratio = height
    else:
        height_ratio = int(args.hsize) #height = 원본 1200, 1500 등 리사이즈

    #print(height_ratio)

    if height >= height_ratio and height_ratio != height:
        resize_width = (width * height_ratio) // height
        height, width = height_ratio, resize_width

    #img1 은 잘린 이미지들이 들어가야함
    img1 = cv2.resize(img1 , (width , height))

    window_name = 'Threshold'
    cv2.namedWindow(window_name)
    func_thr(img=img1, window_name='Threshold',file_name= filename)

    cv2.destroyAllWindows()
    #boxes = pytesseract.image_to_boxes(img1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    chars = pytesseract.image_to_string(img1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")

    print(chars)


if __name__ == '__main__':
    main()

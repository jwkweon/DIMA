import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pytesseract
import pandas as pd
import csv

def Jaccard(str1,str2):
    arr1=[]
    arr2=[]
    gyo = 0

    for i in range(len(str1)-1):
        arr1.append(str1[i:i+1].upper())
        arr1.append(str1[i:i+2].upper())
    arr1.append(str1[-1:].upper())
    for i in range(len(str2)-1):
        arr2.append(str2[i:i+1].upper())
        arr2.append(str2[i:i+2].upper())
    arr2.append(str2[-1:].upper())

    for i in arr1:
        for j in arr2:
            if i == j and i != '@':
                gyo += arr1.count(i) if arr1.count(i) <= arr2.count(j) else arr2.count(j)
                for l in range(len(arr1)):
                    if i == arr1[l]:
                        arr1[l] = '@'
                for l in range(len(arr2)):
                    if i == arr2[l]:
                        arr2[l] = '@'
    hap = len(arr1)+len(arr2)-gyo
    if gyo == 0 and hap ==0:
        gyo = 1
        hap = 1

    return int(gyo / hap * 10000)

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
def cut_roi(img, axis_list):
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

    return result

def hangmok_correct(hangmok_list, check_word):
    max_index = 0
    max_val = 0
    for i, k in enumerate(hangmok_list):
        max_val = max(max_val, Jaccard(k, check_word))
        if Jaccard(k, check_word) == max_val:
            max_index = i

    return hangmok_list[max_index]

#cut해서 들어온 항목들 결과 리스트에 추가
def add_result():
    return result

#최종 결과 리스트를 csv 파일로 저장
def save_csv(df_list):
    result_dataframe = pd.DataFrame(df_list)
    result_dataframe.to_csv('result.csv', header=False, index=False, encoding='cp949')

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

    print(height_ratio)

    if height >= height_ratio and height_ratio != height:
        resize_width = (width * height_ratio) // height
        height, width = height_ratio, resize_width

    img1 = cv2.resize(img1 , (width , height))

    hangmok = ['사업자등록증', '법인사업자', '등록번호', '법인명(단체명)', '대표자', '개업년월일',
               '법인등록번호', '사업장소재지', '본점소재지', '사업의종류', '교부사유', '발급사유',
               '공동사업자', '사업자단위과세적용사업자여부', '전자세금계산서전용메일주소', '상호',
               '성명', '주민등록번호', '생년월일', '일반과세자', '개업연월일']

    #boxes = pytesseract.image_to_boxes(img1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    chars = pytesseract.image_to_string(img1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    dataframe = pytesseract.image_to_data(img1, lang = 'kor3+eng', output_type = Output.DATAFRAME, config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")

    list_dataframe = dataframe_to_list(data_frame = dataframe)

    print(chars)


if __name__ == '__main__':
    main()

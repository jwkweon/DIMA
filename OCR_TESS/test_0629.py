#tesseract 결과 field mapping
#dataframe으로부터 roi 추출
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

def df_list_removeNan(df_list):
    result_list = []

    for i, j in enumerate(df_list):
        if j[10] != -1:
            result_list.append(j)

    return result_list

#special char to blank
def dflist_spchar_to_blank(df_list):
    result_list = df_list[:]
    for i, j in enumerate(df_list):
        temp = ''
        for k in range(len(j[11])):
            if j[11][k] in ['_', '!', '#', '$', '%', '^', '&', '*', '{', '}', '=', ';', '/', '?', '<', '>' '~', '`', '”', '\'', '\"', '|']:
                temp += ' '
            else:
                temp += j[11][k]
        result_list[i][11] = temp

    return result_list

def dflist_define_line(df_list):
    result_list = []
    for i, j in enumerate(df_list):
        if i == 0:
            pass
        else:
            if df_list[i][6] == 0:
                df_list[i][6] = df_list[i-1][6]
            if df_list[i][7] == 0:
                df_list[i][7] = df_list[i-1][7]

    for i, j in enumerate(df_list):
        if i == 0:
            result_list.append(j)
            pass
        else:
            if df_list[i][6] < df_list[i-1][6]:
                result_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, '\n'])
                result_list.append(j)
            else:
                result_list.append(j)

    return result_list

def dflist_del_low_conf(df_list):
    result_list = []

    for i, j in enumerate(df_list):
        if j[10] > 45 or j[10] == -1:
            result_list.append(j)
        else:
            pass

    return result_list

def dflist_roi(df_list):
    result_list = [[0, 0, 0]]
    top, height, cnt, x_init = 0, 0, 1, 0

    for i, j in enumerate(df_list):
        if df_list[i][5] == 1:
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

def cut_roi(img, img_ori, axis_list):
    x_set = [i[0] for i in axis_list]
    x_set = sorted(x_set)
    x_std = sum(x_set[0:6]) // 6

    h_std = []
    for i, j in enumerate(axis_list):
        if i == 0:
            pass
        elif i <= 3:
            h_std[-1] = (h_std[-1] + j[1]) // 2
            h_std.append(j[1] + j[2])
        else:
            if j[0] <= x_std * 2 or j[0] >= x_std * 0.5:
                h_std[-1] = (h_std[-1] + j[1]) // 2
                h_std.append(j[1] + j[2])
            elif j[0] >= x_std * 2:
                h_std[-1] = j[1] + j[2]

    for i, j in enumerate(axis_list):
        h_std.append(j[1])
        h_std.append(j[1] + j[2])

    #print(x_std)

    height, width = img.shape

def field_extract(df_list, axis_list):
    x_set = [i[0] for i in axis_list]
    x_set = sorted(x_set)
    x_std = sum(x_set[0:6]) // 6

    wordnum_last = 0
    field = ['\n']
    temp = []

    for i, j in enumerate(df_list):
        if df_list[i][5] == 1:
            if len(temp) == 0:
                temp.append(df_list[i][11])
            else:
                field.append(' '.join(map(str, temp)))

                temp = []
                temp.append(df_list[i][11])
        else:
            temp.append(df_list[i][11])

    return field

#특수문자 제거 및 : 앞뒤로 공백
def array_dflist(df_list):
    result = []
    for i, j in enumerate(df_list):
        temp = []
        for m, k in enumerate(j):
            if k in ['、', '_', '!', '#', '.', '$', '%', '^', '&', '*', '{', '}', '=', ';', '/', '?', '<', '>' '~', '`', '”', '\'', '\"', '|']:
                pass

            elif k == ' ' and m >= 1:
                if j[m - 1] == ' ':
                    temp.append(k)
                else:
                    pass

            elif k == ':':
                temp.append(' ' + k + ' ')

            else:
                temp.append(k)
        temp = ''.join(map(str, temp))
        if temp != ' ' and temp != '\n':
            result.append(temp)

    return result

#:가 나오면 항목 자르기
def parse_data(df_list):
    f_data = open('ps_test.txt', mode='wt', encoding='utf-8')


    result = []

    for val in df_list:
        temp = []
        for i, j in enumerate(val):
            if ':' in j:
                temp.append(val[:i-1])
                #temp.append(val[i])
                temp.append(val[i+2:])
                break
            else:
                pass
        if len(temp) == 0:
            temp.append(val)
        #temp.append('\n')
        result.append(temp)
        f_data.write(''.join(temp))

    df = pd.DataFrame(result)
    df.to_csv('result2.csv', header=False, index=False, encoding='cp949')


    f_data.close()

    return result

def dict_hangmok(hangmok_list, check_word):
    max_index = 0
    max_val = 0
    for i, k in enumerate(hangmok_list):
        max_val = max(max_val, Jaccard(k, check_word))
        if Jaccard(k, check_word) == max_val:
            max_index = i

    return hangmok_list[max_index]


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

    #boxes = pytesseract.image_to_boxes(img1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    chars = pytesseract.image_to_string(img1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    ddd = pytesseract.image_to_data(img1, lang = 'kor3+eng', output_type = Output.DATAFRAME, config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    dddd = pytesseract.image_to_data(img1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")

    f = open('test.txt', mode='wt', encoding='utf-8')
    f.write(''.join(chars))
    f.close()

    f_data = open('test_data.txt', mode='wt', encoding='utf-8')
    f_data.write(dddd)
    f_data.close()

    list_ddd = dataframe_to_list(data_frame = ddd)

    removed = df_list_removeNan(list_ddd)
    topNheight_list = dflist_roi(removed)

    removed_sp = dflist_spchar_to_blank(removed)
    add_enter = dflist_define_line(removed_sp)
    del_low_prob = dflist_del_low_conf(add_enter)

    fe = field_extract(removed, topNheight_list)
    fe = array_dflist(fe)
    pa_data = parse_data(fe)
    #dd = array_dflist(del_low_prob)

    new = []
    for i, j in enumerate(removed_sp):
        print(j)
        #new.append(''.join(map(str, j)))

    f_data = open('test_data.txt', mode='wt', encoding='utf-8')
    f_data.write(''.join(dddd))
    f_data.close()

    hangmok = ['사업자등록증', '법인사업자', '등록번호', '법인명(단체명)', '대표자', '개업년월일',
               '법인등록번호', '사업장소재지', '본점소재지', '사업의종류', '교부사유', '발급사유',
               '공동사업자', '사업자단위과세적용사업자여부', '전자세금계산서전용메일주소', '상호',
               '성명', '주민등록번호', '생년월일']

    word = '사업의좀류'
    dict_hangmok(hangmok_list = hangmok, check_word = word)


    #df = pd.DataFrame(data)
    #df.to_csv('result2.csv', header=False, index=False, encoding='cp949')

    #cv2.imshow(filename, img_resize)
    #cv2.waitKey(0)

if __name__ == '__main__':
    main()

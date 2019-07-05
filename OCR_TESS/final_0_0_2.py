import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pytesseract
import pandas as pd
import csv

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

    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (width, height),
    	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    print("[INFO] angle: {:.3f}".format(angle))

    rotated = cv2.bitwise_not(rotated)

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
            #cv2.imwrite('t_{}_{}'.format(ths, file_name), save_img)
            break

    return save_img

def Jaccard(str1, str2):
    arr1=[]
    arr2=[]
    intersec = 0

    for i in range(len(str1)-1):
        arr1.append(str1[i:i+1])
        arr1.append(str1[i:i+2])
    arr1.append(str1[-1:])
    for i in range(len(str2)-1):
        arr2.append(str2[i:i+1])
        arr2.append(str2[i:i+2])
    arr2.append(str2[-1:])

    for i in arr1:
        for j in arr2:
            if i == j and i != '@':
                intersec += arr1.count(i) if arr1.count(i) <= arr2.count(j) else arr2.count(j)
                for l in range(len(arr1)):
                    if i == arr1[l]:
                        arr1[l] = '@'
                for l in range(len(arr2)):
                    if i == arr2[l]:
                        arr2[l] = '@'

    union = len(arr1)+len(arr2)-intersec

    if intersec == 0 and union ==0:
        intersec = 1
        union = 1

    return int(intersec / union * 10000)

#tesseract image_to_data 사용시 dataframe 형태로 return 위해
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
            if j[10] == 95 and j[11] == ' ':
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

#전체 이미지로부터 item의 x 시작점과 콜론 시작점 return
def division_std(df_list, axis_list):
    x_set = [i[0] for i in axis_list]
    x_set = sorted(x_set)
    x_std = sum(x_set[0:6]) // 6

    colon_set = [j[6] for i, j in enumerate(df_list) if j[11] in [':', '：']]
    colon_set = sorted(colon_set)
    colon_std = sum(colon_set[0:6]) // 6

    return x_std, colon_std

#cv2.line 부분을 지우고 h_std의 요소값(cut 할 height)별로 자르는 코드 추가필요
def cut_roi(img, axis_list, file_name):
    height, width = img.shape
    show = img.copy()

    mg = 2
    axis = axis_list[:]
    for i, j in enumerate(axis):
        j[2] = j[1] + j[2]

    cut_img = []
    for i, j in enumerate(axis):
        if i == 0:
            continue

        if j[1]-mg>0 or j[2]+mg < height:
            letter = img[j[1]-mg:j[2]+mg, 0:width]
            result = cv2.line(show, (0, j[1]-mg), (width, j[1]-mg), (0, 0, 0), 3)
            result = cv2.line(show, (0, j[2]+mg), (width, j[2]+mg), (0, 0, 0), 3)
            letter = np.array(letter)
            cut_img.append(letter)

    #cut된 이미지 보기 위함
    #cut_show(result)

    return cut_img

def cut_show(img):
    height, width = img.shape
    height_ratio = 1400

    if height >= height_ratio and height_ratio != height:
        resize_width = (width * height_ratio) // height
        height, width = height_ratio, resize_width

    img = cv2.resize(img , (width , height))
    cv2.imshow('result', img)
    cv2.waitKey(0)

def img_padding(img):
    letter = img
    background = letter.copy()
    HEIGHT, WIDTH = background.shape[:2]
    background = cv2.threshold(background, 255, 255, cv2.THRESH_BINARY_INV)

    if HEIGHT > 50 :
        ratio_h = float(25/HEIGHT)
        width_r= int(ratio_h*WIDTH)

        background = cv2.resize(background[-1], (WIDTH,50), interpolation=cv2.INTER_AREA)
        resized = cv2.resize(letter, (width_r,25), interpolation=cv2.INTER_AREA)
        resized_h, resized_w = resized.shape

        x_offset, y_offset = 10, 10

        for k in range(resized_h):
            for l in range(resized_w):
                background[y_offset + k][x_offset+l] = resized[k][l]
    else :
        x_offset, y_offset = 10, 10
        ratio_h = float(25/HEIGHT)
        width_r= int(ratio_h*WIDTH)
        background = cv2.resize(background[-1], (width_r+x_offset,50), interpolation=cv2.INTER_CUBIC)
        resized = cv2.resize(letter, (width_r,25), interpolation=cv2.INTER_AREA)
        resized_h, resized_w = resized.shape

        for k in range(resized_h):
            for l in range(resized_w):
                background[y_offset + k][x_offset+l] = resized[k][l]

    return background

#dictionary 기능을 위한 함수
#Jaccard 유사도로 유사한 항목 검사
def hangmok_correct(check_word):
    hangmok_list = ['사업자등록증', '(법인사업자)', '(법인사업자:본점)', '(면세법인사업자:본점)', '(부가가치세 면세사업자)',
               '등록번호', '법인명(단체명)', '대표자', '개업년월일', '법인등록번호', '사업장소재지', '본점소재지', '사업의종류',
               '교부사유', '발급사유', '공동사업자', '사업자단위과세적용사업자여부', '전자세금계산서전용메일주소',
               '상호', '성명', '주민등록번호', '생년월일', '(일반과세자)', '개업연월일', '_']
    max_index = 0
    max_val = 0
    for i, k in enumerate(hangmok_list):
        max_val = max(max_val, Jaccard(k, check_word))
        if Jaccard(k, check_word) == max_val:
            max_index = i

    return hangmok_list[max_index]

#콜론 2개가 들어오면 빈 공간의 크기가 가장 큰 것을 기준으로 나누기
def two_colon(img):
    df_two_colon = pytesseract.image_to_data(img, lang = 'kor3+eng', output_type = Output.DATAFRAME, config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    list_dataframe = dataframe_to_list(data_frame = df_two_colon)
    removed = df_list_removeNan(list_dataframe)

    x_list = []
    chars = []
    for i, j in enumerate(removed):
        chars.append(removed[i][11])
        if i == 0:
            x_list.append(removed[i][6])
            pass
        else:
            if removed[i][6] == 0:
                removed[i][6] = removed[i-1][6] + removed[i-1][8]
                x_list.append(removed[i][6])
            else:
                x_list.append(removed[i][6])

    x_diff = []
    for i in range(1, len(x_list)):
        x_diff.append(x_list[i] - x_list[i-1])

    i = x_diff.index(max(x_diff))

    return ''.join(chars[:i+1]), ''.join(chars[i+1:])

def check_item(img, colon_flag, item_flag):
    df_list = pytesseract.image_to_data(img, lang = 'kor3+eng', output_type = Output.DATAFRAME, config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    list_dataframe = dataframe_to_list(data_frame = df_list)
    removed = df_list_removeNan(list_dataframe)

    is_item = item_flag
    for i, j in enumerate(removed):
        if j[5] == 1:
            if j[6] >= colon_flag:
                is_item = 0
            else:
                is_item = 1

    return is_item

#cut해서 들어온 이미지 tesseract 결과 추출
def tess_roi(img, stop):
    chars = pytesseract.image_to_string(img, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")

    temp = []
    temp_ex = []
    flag = 1
    cnt_col1 = chars.count(':')
    cnt_col2 = chars.count('：') #이상한 콜론
    cnt_col = cnt_col1 + cnt_col2

    non_item_list = ['사업자등록증', '(법인사업자)', '(법인사업자:본점)', '(일반과세자)', '(면세법인사업자:본점)', '(부가가치세 면세사업자)']
    if hangmok_correct(chars) in non_item_list and stop == 0:
        flag = 3
        temp = hangmok_correct(chars)
        print([temp, temp_ex, flag])
        return temp, temp_ex, flag, stop

    if cnt_col == 0:
        temp.append(chars)
        flag = 0
    elif cnt_col == 1:
        for i, j in enumerate(chars):
            if j == ':' or j == '：':
                temp.append(hangmok_correct(chars[:i]))
                if hangmok_correct(chars[:i]) == '등록번호':
                    stop = 1
                    temp.append(chars[i+1:])
                #개업연월일 뒤 등록번호의 콜론이 인식이 안되는 오류에 대한 예외처리(인식률 개선 필요)
                elif hangmok_correct(chars[:i]) == '개업연월일' or hangmok_correct(chars[:i-1]) == '개업년월일':
                    temp_sep = ''
                    for m, n in enumerate(chars[i+1:]):
                        if n == '일':
                            temp_sep = chars[i+1:i+m+2]
                            temp.append(temp_sep)
                            break
                    temp_ex.append(hangmok_correct(chars[i+m+2:]))
                else:
                    temp.append(chars[i+1:])
    else:
        char1, char2 = two_colon(img)
        print(char1, char2)
        for i, j in enumerate(char1):
            if j == ':' or j == '：':
                temp.append(hangmok_correct(char1[:i]))
                temp.append(char1[i+1:])
        for i, j in enumerate(char2):
            if j == ':' or j == '：':
                temp_ex.append(hangmok_correct(char2[:i]))
                temp_ex.append(char2[i+1:])
        flag = 2

    print([temp, temp_ex, flag])
    return temp, temp_ex, flag, stop

#cut해서 들어온 항목들 결과 리스트에 추가
def add_result(input1, input2, flag, stop, output):
    if flag == 0 and stop == 1: #flag 0 : 콜론이 없는 문자열
        if len(input1) != 0:
            output[-1][-1] = output[-1][-1] + input1[0]
    elif flag == 1:             #flag 1 : 콜론이 1개가 있는 문자열 (항목 : 내용)의 구조거나 (항목 : 내용 항목)의 구조(예외적 상황)
        output.append(input1)
    elif flag == 2:             #flag 2 : 콜론이 2개가 있는 문자열 (항목 : 내용 항목 : 내용)의 구조로 중간 내용과 항목의 구별
        output.append(input1)
        output.append(input2)
    else:                       #flag 3 : 등록번호가 들어오기 전까지의 문자열들의 구분을 위함
        output.append([input1])

    return output

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
    gray_copy = img_gray.copy()
    height, width, _ = img.shape

    if args.hsize == None:
        height_ratio = height
    else:
        height_ratio = int(args.hsize) #height = 원본 1200, 1500 등 리사이즈

    print(height_ratio)

    if height >= height_ratio and height_ratio != height:
        resize_width = (width * height_ratio) // height
        height, width = height_ratio, resize_width

    gray_copy = cv2.resize(gray_copy , (width , height))

    #threshold 설정 (UI)
    window_name = 'Threshold'
    cv2.namedWindow(window_name)
    save_img = func_thr(img=gray_copy, window_name='Threshold',file_name= filename)
    cv2.destroyAllWindows()

    #tess 결과 확인용(임시)
    #chars = pytesseract.image_to_string(gray_copy, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")

    #원본 이미지로 컷
    dataframe = pytesseract.image_to_data(save_img, lang = 'kor3+eng', output_type = Output.DATAFRAME, config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")

    list_dataframe = dataframe_to_list(data_frame = dataframe)
    removed = df_list_removeNan(list_dataframe)
    topNheight_list = dflist_roi(removed)

    #x_std는 항목들의 시작 좌표의 평균 // colon_std : 항목과 내용을 구분짓는 콜론의 좌표
    x_std, colon_std = division_std(df_list = removed, axis_list = topNheight_list)

    print(x_std, colon_std)

    cut_img = cut_roi(img = save_img, axis_list = topNheight_list, file_name=filename)
    result_form = []

    stop_f = 0
    is_item = 0
    for i in cut_img:
        if is_item == 0:
            is_item = check_item(img = i, colon_flag = colon_std, item_flag = is_item)
            i = img_padding(img = i)
            hang1, hang2, flag, stop_f = tess_roi(i, stop = stop_f)
        else:
            i = img_padding(img = i)
            hang1, hang2, flag, stop_f = tess_roi(i, stop = stop_f)

        res2 = add_result(input1 = hang1, input2 = hang2, flag = flag, stop = stop_f, output = result_form)

    #사업의 종류 내용 지우기
    for i, j in enumerate(res2):
        if j[0] in ['사업의종류', '전자세금계산서전용메일주소', '사업자단위과세적용사업자여부']:
            j[1] = ''

    save_csv(df_list = res2)


if __name__ == '__main__':
    main()

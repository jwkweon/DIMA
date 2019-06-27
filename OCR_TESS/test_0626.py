#tesseract 결과 field mapping
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pytesseract
import pandas as pd


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
    ddd = pytesseract.image_to_data(img1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@").DATAFRAME

    #image_to_osd
    cnt = 0
    cnt2 = 0

    for i in range(len(chars)):
        if chars[i] == ' ':
            pass
        else:
            cnt += 1

    #print(type(chars), '\n',  ddd)
    res_chars = []
    res2_chars = []

    print(type(chars), type(res_chars))
    for i, j in enumerate(chars):
        if j == '\n':
            if chars[i+1] == '\n':
                pass
            else:
                res_chars.append(j)
        elif j in ['_', '!', '#', '$', '%', '^', '&', '*', '{', '}', '=', ';', '/', '?', '<', '>' '~', '`', '”', '\'', '\"', '|']:
            pass
        elif j == ' ':
            if chars[i:i+5] == '     ':
                res_chars.append('  ')
            else:
                pass
        elif j == ':':
            res_chars.append(' ' + j + ' ')
        else:
            res_chars.append(j)

    k = 0
    for i, j in enumerate(res_chars):
        if j == '\n':
            res2_chars.append(''.join(res_chars[k:i+1]))
            k = i +1
        else:
            pass

    #for

    #for i, j in enumerate(res2_chars):




    #print(''.join(res_chars))
    #print('=' * 20)
    #print(''.join(res2_chars))
    f = open('test.txt', mode='wt', encoding='utf-8')
    f.write(''.join(chars))
    f_data = open('test_data.txt', mode='wt', encoding='utf-8')
    f_data.write(ddd)

    f.close()
    f_data.close()


    #df = get_pandas_output()
    #df = ddd.DATAFRAME()
    print(df)




    #cv2.imshow(filename, img_resize)
    #cv2.waitKey(0)

if __name__ == '__main__':
    main()

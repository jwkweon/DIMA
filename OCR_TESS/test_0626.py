#tesseract 결과 field mapping
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import pytesseract

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

    if height >= height_ratio and height_ratio != height:
        resize_width = (width * height_ratio) // height
        height, width = height_ratio, resize_width

    img1 = cv2.resize(img1 , (width , height))

    #boxes = pytesseract.image_to_boxes(img1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    chars = pytesseract.image_to_string(img1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")

    cnt = 0
    cnt2 = 0

    for i in range(len(chars)):
        if chars[i] == ' ':
            pass
        else:
            cnt += 1


    print(chars)








    #cv2.imshow(filename, img_resize)
    #cv2.waitKey(0)

if __name__ == '__main__':
    main()

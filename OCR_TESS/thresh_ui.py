#threshold 유저 설정
import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

def func_adapthr(img, window_name, file_name):
    filt_name = 'f_size'
    c_name = 'C_val'
    cv2.createTrackbar(filt_name, window_name, 0, 50, nothing)
    cv2.createTrackbar(c_name, window_name, 0, 100, nothing)
    cv2.setTrackbarPos(filt_name, window_name, 5)
    cv2.setTrackbarPos(c_name, window_name, 15)
    height, width = img.shape

    while (1):
        height, width = img.shape
        filt_val = cv2.getTrackbarPos(filt_name, window_name)
        c_val = cv2.getTrackbarPos(c_name, window_name)

        gray = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, int(filt_val/2)*2+1, c_val)
        save_img = gray.copy()

        height_ratio = 1200

        if height >= height_ratio and height_ratio != height:
            resize_width = (width * height_ratio) // height
            height, width = height_ratio, resize_width

            img_resize = cv2.resize(gray , (width , height))
        else:
            img_resize = gray

        #img_resize = cv2.fastNlMeansDenoising(img_resize, None, 15, 7, 21)
        cv2.imshow(window_name, img_resize)

        if cv2.waitKey(30) & 0xFF == 27:
            cv2.imwrite('thr_{}_{}'.format(filt_val, file_name), save_img)
            break

def func_thr(img, window_name, file_name):
    threshold_name = 'Ths'
    cv2.createTrackbar(threshold_name, window_name, 0, 255, nothing)
    cv2.setTrackbarPos(threshold_name, window_name, 140)
    height, width = img.shape

    while (1):
        height, width = img.shape
        ths = cv2.getTrackbarPos(threshold_name, window_name)
        ret, gray = cv2.threshold(img, ths, 255, cv2.THRESH_BINARY)
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
            cv2.imwrite('thr_{}_{}'.format(ths, file_name), save_img)
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename",
                    required = True)
    args = parser.parse_args()
    filename = args.filename

    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    height, width, _ = img.shape

    #height_ratio = 1200

    #if height >= height_ratio and height_ratio != height:
    #    resize_width = (width * height_ratio) // height
    #    height, width = height_ratio, resize_width

    #    img_resize = cv2.resize(img_gray , (width , height))
    #else:
    #    img_resize = img_gray

    window_name = 'Threshold'
    cv2.namedWindow(window_name)

    #threshold 종류 선택
    #func_adapthr(img_gray, window_name = 'Threshold', file_name=filename)
    func_thr(img_gray, window_name = 'Threshold', file_name=filename)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

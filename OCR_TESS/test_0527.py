#2019/05/27
#pytesseract에서 traineddata 사용가능한지 확인하기위한 코드
# + 박스치기

import cv2
import pytesseract
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename",
                    required = True)
    args = parser.parse_args()
    filename = args.filename

    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img1 = img_gray.copy()
    height, width, _ = img.shape

    n_size = width // 140
    if n_size % 2 == 0:
        n_size += 1
    else:
        pass

    print(height, width, n_size)

    img1 = cv2.fastNlMeansDenoising(img_gray, None, 15, 7, 21)
    thr1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, n_size, 15) #default = 11


    height_ratio = height

    if height >= height_ratio and height_ratio != height:
        resize_width = (width * height_ratio) // height
        height, width = height_ratio, resize_width

        img = cv2.resize(img , (width , height))

    boxes = pytesseract.image_to_boxes(thr1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    chars = pytesseract.image_to_string(thr1, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")

    cnt = 0
    cnt2 = 0

    for i in range(len(chars)):
        if chars[i] == ' ':
            pass
        else:
            cnt += 1
            #print(chars[i])

    #print(len(chars), cnt)


    for i in boxes.splitlines():
        cnt2 += 1
        #print(i)
        box = i.split(' ')
        h_d, w_d = abs(int(box[2]) - int(box[4])), abs(int(box[1]) - int(box[3]))

        img = cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0), 1)
        '''
        #밑은 박스의 픽셀 크기가 400이 넘으면 박스를 그리지 않음#
        if h_d != 0 and w_d != 0 and h_d * w_d < 8000:
            if h_d / w_d <= 6 and w_d / h_d <= 5:
                img = cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0), 1)
                cv2.putText(img, box[0]+str(w_d * h_d), (int(box[1]), height - int(box[4])), font, fontScale, (0, 0, 255), 1, cv2.LINE_AA)
        '''
                #char_cut = img[height - int(box[4]) - h_d * 1.5:height - int(box[2]) + h_d * 1.5, int(box[1]) - 1.5 * w_d:int(box[3]) + 1.5 * w_d].copy()
                #cut = pytesseract.image_to_boxes(char_cut)
                #cut_box = cut.split(' ')
                #cv2.putText(img, cut_box[0], (int(box[3]), height - int(box[4])), font, fontScale, (255, 0, 255), 1, cv2.LINE_AA)

    #print(len(chars), cnt, cnt2)
    print(chars)

    height_ratio = 1200

    if height >= height_ratio and height_ratio != height:
        resize_width = (width * height_ratio) // height
        height, width = height_ratio, resize_width

        img_resize = cv2.resize(img , (width , height))
    else:
        img_resize = img
    #cv2.imwrite('res-' + filename, img)
    cv2.imshow(filename, img_resize)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
    #print(pytesseract.get_tesseract_version())

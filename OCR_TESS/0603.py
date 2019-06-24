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
    height, width, _ = img.shape

    if height >= 1200:
        resize_width = (width * 1200) // height
        height, width = 1200, resize_width

        img = cv2.resize(img , (width , height))

    boxes = pytesseract.image_to_boxes(img)

    for i in boxes.splitlines():
        box = i.split(' ')
        h_d, w_d = abs(int(box[2]) - int(box[4])), abs(int(box[1]) - int(box[3]))
        #밑은 박스의 픽셀 크기가 400이 넘으면 박스를 그리지 않음#
        #if h_d != 0 and w_d != 0 and h_d * w_d <= 400:
        #    if h_d / w_d <= 6 and w_d / h_d <= 5:
        img = cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 255, 0), 1)

    cv2.imwrite('res-' + filename, img)
    #cv2.imshow(filename, img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

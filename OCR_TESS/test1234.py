from PIL import Image     #pip install pillow
from pytesseract import * #pip install pytesseract
import configparser
import sys
import os
import csv
import re
from pymongo import MongoClient          #pip install pymongo, mongodb 설치
import pytesseract

## 읽어오는 부분
#Config Parser 초기화
config = configparser.ConfigParser()
#Config File 읽기
config.read(os.path.dirname(os.path.realpath(__file__)) + os.sep + 'envs' + os.sep + 'property.ini')

def strToTxt(txtName, outText):
    with open(txtName + '.txt', 'w', encoding='utf-8') as f:
        f.write(outText)

#텍스트 정제(전처리)
def cleanText(readData):
    #스팸 메세지에 포함되어 있는 특수 문자 제거
    text = re.sub('[-=+,#/\?^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    #양쪽(위,아래)줄바꿈 제거
    text = text.strip('\n')
    return text


#이미지 -> 문자열 추출
def ocrToStr(fullPath, outTxtPath, fileName, lang='kor3'): #디폴트는 영어로 추출
    #이미지 경로

    img = Image.open(fullPath)
    txtName = os.path.join(outTxtPath,fileName.split('.')[0])

    #추출(이미지파일, 추출언어, 옵션)
    #preserve_interword_spaces : 단어 간격 옵션을 조절하면서 추출 정확도를 확인한다.
    #psm(페이지 세그먼트 모드 : 이미지 영역안에서 텍스트 추출 범위 모드)
    #psm 모드 : https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
    outText = image_to_string(img, lang=lang, config='--psm 1 -c preserve_interword_spaces=1')
    #outText = pytesseract.image_to_string(img, lang = 'kor3+eng', config="--psm 4 --oem 1 -c tessedit_char_whitelist=-01234567890XYZ:@")
    print('+++ OCT Extract Result +++')
    print('Extract FileName ->>> : ', fileName, ' : <<<-')
    print('\n\n')
    #출력
    print(outText)
    #추출 문자 텍스트 파일 쓰기
    strToTxt(txtName, outText)

#문자열 -> 텍스트파일 개별 저장


#텍스트파일 -> csv 파일 생성
def txtToCsv(txtName, cateName, outTxtPath, outCsvPath):
    #파일 사이즈가 0이면 패스(미추출 파일)
    if os.path.getsize(os.path.join(outTxtPath,txtName)) != 0:
        with open(os.path.join(outTxtPath, txtName), 'r', encoding='utf-8') as r:
            with open(os.path.join(outCsvPath, config['FilneName']['CsvFileName']),'a',  encoding='utf-8', newline='') as w:

                writer = csv.writer(w, delimiter=',')
                clText = cleanText(r.read())
                #list =[]
                #content=[]
                #a=':'
                #i=0
                #if a not in clText:
                #    for row in clText[0:-1]:
                #        list.append(row)
                #else :
                #    while clText[i:i+1] != a:
                #        i += 1

                #    for row in clText[0:i]:
                #        list.append(row)

                #    for row2 in clText[i+2:-1]:
                #        content.append(row2)
                #print(list)
                #print(content)



                ##print('//////////')
                writer.writerow([cateName, clText])
                #writer.writerow([cateName, list, content])
                #spamList.append({"category" : cateName, "contents": clText})



#메인 시작
if __name__ == "__main__":
    #텍스트 파일 저장 경로
    outTxtPath = os.path.dirname(os.path.realpath(__file__))+ config['Path']['OcrTxtPath']
    #CSV 파일 저장 경로
    outCsvPath = os.path.dirname(os.path.realpath(__file__))+ config['Path']['TxtCsvPath']

    #OCR 추출 작업 메인
    #for root, dirs, files in os.walk(os.path.dirname(os.path.realpath(__file__)) + config['Path']['OriImgPath']):
    #    for fname in files:
    #        fullName = os.path.join(root, fname)
    #        #한글+영어 추출(kor, eng , kor+eng)
    #        ocrToStr(fullName, outTxtPath, fname,'kor3+eng')

    #CSV 변환 작업 메인
    for fname in os.listdir(os.path.dirname(os.path.realpath(__file__))+ config['Path']['OcrTxtPath']):
        cateName = ''.join([i for i in fname if not i.isdigit()]).split('.')[0].strip()
        txtToCsv(fname, cateName, outTxtPath, outCsvPath)


    #작업 완료 메시지
    print('+++ OCR Image >> Text >> CSV Convert Complete! +++')

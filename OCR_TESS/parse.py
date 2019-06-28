#parse 와 유사도
import os
import pandas as pd
import csv
from numpy import dot
from numpy.linalg import norm
import numpy as np

def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

with open(os.path.join('.\\', 'result3.csv'), 'w',  encoding='utf-8', newline='') as w:
    writer = csv.writer(w, delimiter=',')
    cateName, clText = ['123123'], ['222']
    writer.writerow([cateName, clText])



data = [['사업자등록증'], ['법인사없자'], ['등록번호', ':', '112-81-30811']]

df = pd.DataFrame(data)
df.to_csv('result2.csv', header=False, index=False, encoding='cp949')

class JamoSeparator():
    def __init__(self, string):
        self.string = string
        self.result = []
        self.cho_list = [char for char in "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"]
        self.jung_list = [char for char in "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"]
        self.jong_list = [char for char in " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]

    def run(self):
        for char in self.string:
            character_code = ord(char)
            # Do not process unless it is in Hangul syllables range.
            if 0xD7A3 < character_code or character_code < 0xAC00:
                continue

            cho_index = (character_code - 0xAC00) // 21 // 28
            jung_index = (character_code - 0xAC00 - (cho_index * 21 * 28)) // 28
            jong_index = character_code - 0xAC00 - (cho_index * 21 * 28) - (jung_index * 28)

            self.result.append(self.cho_list[cho_index])
            self.result.append(self.jung_list[jung_index])
            self.result.append(self.jong_list[jong_index])
            self.result.append("_")

    def get(self):
        return self.result

jamos = JamoSeparator(a)
jamos.run()
print("jamos: \n> {0}\n".format(jamos.get()))
a = '마'
ord(a)
ord(a) - 0xB9C8
47560 // (16 ** 3)
(47560 % (16 ** 3)) // (16 ** 2)
47560 % (16 ** 2) // (16 ** 1)
47560 % (16 ** 1)


cho_list = [i for i, char in enumerate("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ") if char == 'ㅁ']
cho_list

def jamo_to_bit(jamo_list):
    cho = jamo_list[0]
    jung = jamo_list[1]
    jong = jamo_list[2]

    cho_index = [i for i, char in enumerate("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ") if char == cho]
    jung_index = [i for i, char in enumerate("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ") if char == jung]
    jong_index = [i for i, char in enumerate(" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ") if char == jong]

    cho_bit = ['00010', '00011', '00100', '00101', '00110', '00111', '01000', '01001', '01010', '01011',
               '01100', '01101', '01110', '01111', '10000', '10001', '10010', '10011', '10100']
    jung_bit = ['00011', '00100', '00101', '00110', '00111', '01010', '01011', '01100', '01101', '01110',
                '01111', '10010', '10011', '10100', '10101', '10110', '10111', '11010', '11011', '11100', '11101']
    jong_bit = ['00000', '00010', '00011', '00100', '00101', '00110', '00111', '01000', '01001', '01010',
                '01011', '01100', '01101', '01110', '01111', '10000', '10001', '10011', '10100', '10101',
                '10110', '10111', '11000', '11001', '11010', '11011', '11100', '11101']

    result = []
    result.append(''.join([cho_bit[cho_index[0]], jung_bit[jung_index[0]], jong_bit[jong_index[0]]]))

    return result

######### 유사도 실험 ######################
a = '법인사업자'
sa = []
for i in a:
    print(i)
    jamos = JamoSeparator(i)
    jamos.run()
    sa.append(jamos.get())
print(sa)

sa_bit = []
for i in sa:
    sa_bit.append(int(jamo_to_bit(i)[0]))
print(sa_bit)

d = '범인사업자'
sd = []
for i in d:
    print(i)
    jamos = JamoSeparator(i)
    jamos.run()
    sd.append(jamos.get())
print(sd)

sd_bit = []
for i in sd:
    sd_bit.append(int(jamo_to_bit(i)[0]))
print(sd_bit)

e = '개업년월일'
se = []
for i in e:
    print(i)
    jamos = JamoSeparator(i)
    jamos.run()
    se.append(jamos.get())
print(se)

se_bit = []
for i in se:
    se_bit.append(int(jamo_to_bit(i)[0]))
print(se_bit)

doc1=np.array(sa_bit)
doc2=np.array(sd_bit)
doc3=np.array(se_bit)

print(cos_sim(doc1, doc2)) #문서1과 문서2의 코사인 유사도
print(cos_sim(doc1, doc3)) #문서1과 문서3의 코사인 유사도
print(cos_sim(doc2, doc3)) #문서2과 문서3의 코사인 유사도


#같은 글자수 일경우 어느정도의 유사도 보증 but 다른 글자수 처리 하는 방법 찾아얗마












b = jamo_to_bit(a)
b

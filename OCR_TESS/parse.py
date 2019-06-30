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



data = [['사업자등록증'], ['법인사없자'], ['등록번호', '112-81-30811']]

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

a = '권'
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

    cho_bit = [0b00010, 0b00011, 0b00100, 0b00101, 0b00110, 0b00111, 0b01000, 0b01001, 0b01010, 0b01011,
               0b01100, 0b01101, 0b01110, 0b01111, 0b10000, 0b10001, 0b10010, 0b10011, 0b10100]
    jung_bit = [0b00011, 0b00100, 0b00101, 0b00110, 0b00111, 0b01010, 0b01011, 0b01100, 0b01101, 0b01110,
                0b01111, 0b10010, 0b10011, 0b10100, 0b10101, 0b10110, 0b10111, 0b11010, 0b11011, 0b11100, 0b11101]
    jong_bit = [0b00000, 0b00010, 0b00011, 0b00100, 0b00101, 0b00110, 0b00111, 0b01000, 0b01001, 0b01010,
                0b01011, 0b01100, 0b01101, 0b01110, 0b01111, 0b10000, 0b10001, 0b10011, 0b10100, 0b10101,
                0b10110, 0b10111, 0b11000, 0b11001, 0b11010, 0b11011, 0b11100, 0b11101]

    result = []
    result.append([cho_bit[cho_index[0]], jung_bit[jung_index[0]], jong_bit[jong_index[0]]])

    return result

######### 유사도 실험 ######################

a = '대표자'
sa = []

for i in a:
    jamos = JamoSeparator(i)
    jamos.run()
    sa.append(jamos.get())

#print(sa)

sa_bit = []
for i in sa:
    sa_bit.append([k / sum(jamo_to_bit(i)[0]) for k in jamo_to_bit(i)[0]])
print(sa_bit)

d = '사업자'
sd = []
for i in d:
    print(i)
    jamos = JamoSeparator(i)
    jamos.run()
    sd.append(jamos.get())
print(sd)

sd_bit = []
for i in sd:
    sd_bit.append([k / sum(jamo_to_bit(i)[0]) for k in jamo_to_bit(i)[0]])
print(sd_bit)

e = '대자'
se = []
for i in e:
    print(i)
    jamos = JamoSeparator(i)
    jamos.run()
    se.append(jamos.get())
print(se)

se_bit = []
for i in se:
    se_bit.append([k / sum(jamo_to_bit(i)[0]) for k in jamo_to_bit(i)[0]])
print(se_bit)

doc1=np.array(sa_bit)
doc2=np.array(sd_bit)
doc3=np.array(se_bit)

print(cos_sim(doc1, np.transpose(doc2))) #문서1과 문서2의 코사인 유사도
print(cos_sim(doc1, np.transpose(doc3))) #문서1과 문서3의 코사인 유사도
print(cos_sim(doc2, np.transpose(doc3))) #문서2과 문서3의 코사인 유사도












#

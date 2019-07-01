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

def hangmok_correct(hangmok_list, check_word):
    max_index = 0
    max_val = 0
    for i, k in enumerate(hangmok_list):
        max_val = max(max_val, Jaccard(k, check_word))
        if Jaccard(k, check_word) == max_val:
            max_index = i

    return hangmok_list[max_index]



hangmok = ['사업자등록증', '법인사업자', '등록번호', '법인명(단체명)', '대표자', '개업년월일',
           '법인등록번호', '사업장소재지', '본점소재지', '사업의종류', '교부사유', '발급사유',
           '공동사업자', '사업자단위과세적용사업자여부', '전자세금계산서전용메일주소', '상호',
           '성명', '주민등록번호', '생년월일', '일반과세자', '개업연월일']

word = '명'
hangmok_correct(hangmok_list = hangmok, check_word = word)

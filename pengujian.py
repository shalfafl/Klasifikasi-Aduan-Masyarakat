# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:41:48 2019

@author: Shalfa-PC
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np
from collections import Counter
import operator 
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt 
import mysql.connector as db

mydb = db.connect(
  host="localhost",
  user="root",
  passwd="",
  database="ta"
)
mycursor = mydb.cursor()

data_bersih = pd.read_sql('SELECT * FROM data_preprocess', con=mydb)

#PEMBAGIAN DATA LATIH DAN UJI
msk = np.random.rand(len(data_bersih)) <= 0.70 #70% latih, 30% uji
train = data_bersih[msk]
test = data_bersih[~msk]

vectorizer = TfidfVectorizer(smooth_idf=False, norm=None) 
train_matrix = vectorizer.fit_transform(train["Pesan"].values.astype('U')).toarray()
fn = vectorizer.get_feature_names()
test_matrix = vectorizer.transform(test["Pesan"].values.astype('U')).toarray()

y_train_bidang = []
for index, row in train.iterrows():
    y_train_bidang.append(row["Bidang"])
y_test_bidang = []
for index, row in test.iterrows():
    y_test_bidang.append(row["Bidang"])
    
y_train_unit = []
for index, row in train.iterrows():
    y_train_unit.append(row["Unit_Kerja"])
y_test_unit = []
for index, row in test.iterrows():
    y_test_unit.append(row["Unit_Kerja"])
    
cosine_similarity = cosine_similarity(train_matrix, test_matrix)
jarak = pd.DataFrame(cosine_similarity)
data_jarak_peruji = jarak.T.values.tolist()

"""UNIT"""
jarak_label_unit=[]
jarak_nilai_unit=[]
jarak_sorted_unit=[]
for (idxI,i) in enumerate(data_jarak_peruji):
    dict_jarak_label_unit = {}
    dict_jarak_nilai_unit = {}
    for (idx, j) in enumerate(i):
        dict_jarak_label_unit[idx]=y_train_unit[idx]
        dict_jarak_nilai_unit[idx]=j
    jarak_label_unit.append(dict_jarak_label_unit)
    jarak_nilai_unit.append(dict_jarak_nilai_unit)

    jarak_sorted_unit.append(sorted(dict_jarak_nilai_unit.items(), key = 
             lambda kv:(kv[1], kv[0]), reverse=True))    

nilaiK=11
jarak_tetangga_unit=[]
for i in jarak_sorted_unit:
    nilaiI=0
    temp={}
    for (idxx,j) in enumerate (i):
        if nilaiI==nilaiK:
            break
        temp[idxx]=j
        nilaiI+=1
    jarak_tetangga_unit.append(temp)

#VOTING
#cocokin dl nama label
vot=[]
for (idx,i) in enumerate (jarak_tetangga_unit):
    xu=[]
    for (idy, j) in enumerate (i.items()):
        xu.append(dict_jarak_label_unit[j[1][0]])
    vot.append(xu) #nama label tetangga pd masing2 data uji
voting_unit=[]
for (idx,i) in enumerate (vot): 
    voti=Counter(i)
    voting_unit.append(voti) #voting data uji dr label tetangga

'''kalo hasil knn gini'''
hasil_knn_unit=[]
for (idx,i) in enumerate (voting_unit):
    predict_knn_unit=(max(i.items(), key=operator.itemgetter(1))[0])
    hasil_knn_unit.append(predict_knn_unit)
    
#LABEL SEMUA 
lu = list(Counter(y_train_unit)) 

#MEMBERSHIP
member_unit=[]
for (idx,i) in enumerate (jarak_tetangga_unit):
    memberships_unit=[]
    for (idy, j) in enumerate (i.items()):
        y = dict_jarak_label_unit[j[1][0]]
        membership_unt = dict()
        for c in lu:
            try:
                uci = 0.49 * (voting_unit[idx][c] / nilaiK)
                if c == y:
                    uci += 0.51
                membership_unt[c] = uci
            except:
                membership_unt[c] = 0
        memberships_unit.append(membership_unt)
    member_unit.append(memberships_unit) #perdata uji ke msg2 tetangga, pertetangga ke smua label
    
#bobot pangkat
m_fuzzy = 2
#ini bobot yang dijadikan perdata uji    
bobot_unit=[]
pangkat = 2 / (m_fuzzy - 1)
for (idx,i) in enumerate (jarak_tetangga_unit):
    xuu = 0
    bbt=[]
    for (idy, j) in enumerate (i.items()):
        try:
            xuu = 1 / pow(j[1][1], pangkat) #bisa 1/0 jd error maka pake try..
        except:
            xuu = 0
        bbt.append(xuu)
    bobot_unit.append(bbt) #per data uji ke msg2 tetangga

bobot_jml_unit=[]
for (idx,i) in enumerate (bobot_unit):
    bobot=0
    for (idy,j) in enumerate (i):
        bobot+=j
    bobot_jml_unit.append(bobot) #bobot per data uji
  
#Himpunan Fuzzy
hasil_himp_fuzzy_unit=[]
for (idx,i) in enumerate (member_unit):
    fuzzy_bobot_unit=(bobot_unit[idx])
    dict_himp_fuzzy_unit={}
    for (idy,j) in enumerate (i):
        temp={}
        for(idz,k) in enumerate (j.items()):
            try:
                hasil_kali_fuzzy_unit = (fuzzy_bobot_unit[idy] * k[1]) / bobot_jml_unit[idx]
                temp[k[0]]=(hasil_kali_fuzzy_unit)
            except:
                temp[k[0]]=0
        dict_himp_fuzzy_unit[idy]=(temp)
    hasil_himp_fuzzy_unit.append(dict_himp_fuzzy_unit)

#Predict
#jumlahkan per kelas
total_fuzzy_unit=[]
for (idx,i) in enumerate (hasil_himp_fuzzy_unit):
    temp={}
    for key,value in i.items():
        for k in value.keys():
            #print(value[k])
            temp[k] = temp.get(k,0) + value[k] #k awalnya 0 jd data semua label di tetangga di jmlkan
    total_fuzzy_unit.append(temp) #data uji ke msg2 label
    
hasil_prediksi_unit=[]
jarak_sorted_fuzzy=[]
for idx,i in enumerate (total_fuzzy_unit):
    predict_unit=(max(i.items(), key=operator.itemgetter(1))[0])
    hasil_prediksi_unit.append(predict_unit)  
    jarak_sorted_fuzzy.append(sorted(i.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))  

  
"""PENGUJIAN UNIT"""
labelsU = list(set(y_test_unit))
labelsU.sort() #biar urut abjad aja

cfU = pd.DataFrame(
    data=confusion_matrix(y_test_unit, hasil_prediksi_unit, labels=labelsU),
    columns=labelsU,
    index=labelsU
)
cfU

tpsU = {}
fpsU = {}
fnsU = {}
tnsU = {}

for label in labelsU:
    tpsU[label] = cfU.loc[label, label]
    fpsU[label] = cfU[label].sum() - tpsU[label]
    fnsU[label] = cfU.loc[label].sum() - tpsU[label]

#akurasi dengan TN
for label in set(y_test_unit):
    tnsU[label] = len(y_test_unit) - (tpsU[label] + fpsU[label] + fnsU[label])
#print("True Negatives:", tnsU)  
total_true_unit = sum(list(tpsU.values()) + list(tnsU.values()))
total_predictions_unit = sum(list(tpsU.values()) + list(tnsU.values()) + list(fpsU.values()) + list(fnsU.values()))
accuracy_global_new_unit = total_true_unit / total_predictions_unit if total_predictions_unit > 0. else 0.
print("Accuracy Unit:", accuracy_global_new_unit*100)
#total_predictionsU = cfU.values.sum()

#presisi dan recall Micro average
correct_predictionsU = sum(tpsU.values())
tpfpU = sum(list(tpsU.values())+list(fpsU.values()))
tpfnU = sum(list(tpsU.values())+list(fnsU.values()))
precisionUnit =  correct_predictionsU / tpfpU if tpfpU > 0. else 0
print("Precision Unit: ",precisionUnit*100)
recallUnit =  correct_predictionsU / tpfnU if tpfnU > 0. else 0
print("Recall Unit: ",recallUnit*100)

#diagram hasil pengujian
objects = ['akurasi', 'presisi', 'recall'] 
y_pos = np.arange(len(objects))
performance = [accuracy_global_new_unit, precisionUnit, recallUnit]

plt.bar(y_pos, performance, align='center', alpha=0.5, color = ['red', 'green', 'blue'])
plt.xticks(y_pos, objects)
plt.ylabel('Nilai (%)')
plt.title('Kategori Unit\n 90:10\n k=11')
plt.show()

"""BIDANG"""
jarak_label_bidang=[]
jarak_nilai_bidang=[]
jarak_sorted_bidang=[]

for (idxI,i) in enumerate(data_jarak_peruji):
    dict_jarak_label_bidang = {}
    dict_jarak_nilai_bidang = {}
    for (idx, j) in enumerate(i):
        dict_jarak_label_bidang[idx]=y_train_bidang[idx]
        dict_jarak_nilai_bidang[idx]=j
    jarak_label_bidang.append(dict_jarak_label_bidang)
    jarak_nilai_bidang.append(dict_jarak_nilai_bidang)

    jarak_sorted_bidang.append(sorted(dict_jarak_nilai_bidang.items(), key = 
             lambda kv:(kv[1], kv[0]), reverse=True))    

nilaiK=11
jarak_tetangga_bidang=[]
for i in jarak_sorted_bidang:
    nilaiI=0
    temp_bidang={}
    for (idxx,j) in enumerate (i):
        if nilaiI==nilaiK:
            break
        temp_bidang[idxx]=j
        nilaiI+=1
    jarak_tetangga_bidang.append(temp_bidang)

#VOTING
vot_bidang=[]
for (idx,i) in enumerate (jarak_tetangga_bidang):
    xb=[]
    for (idy, j) in enumerate (i.items()):
        xb.append(dict_jarak_label_bidang[j[1][0]])
    vot_bidang.append(xb)
voting_bidang=[]
for (idx,i) in enumerate (vot_bidang): 
    #print(Counter(i))
    voti_bidang=Counter(i)
    #print(voti)
    voting_bidang.append(voti_bidang)
    
'''kalo hasil knn gini'''
hasil_knn_bidang=[]
for (idx,i) in enumerate (voting_bidang):
    #print(i)
    #for (key, value) in i.items():
    #    print(key)
    predict_knn_bidang=(max(i.items(), key=operator.itemgetter(1))[0])
    hasil_knn_bidang.append(predict_knn_bidang)
    
#LABEL SEMUA    
lb = list(Counter(y_train_bidang))

#MEMBERSHIP
member_bidang=[]
for (idx,i) in enumerate (jarak_tetangga_bidang):
    memberships_bidang=[]
    for (idy, j) in enumerate (i.items()):
        yb = dict_jarak_label_bidang[j[1][0]]
        membership_bdg = dict()
        for c in lb:
            try:
                uci = 0.49 * (voting_bidang[idx][c] / nilaiK)
                if c == yb:
                    uci += 0.51
                membership_bdg[c] = uci
            except:
                membership_bdg[c] = 0
        memberships_bidang.append(membership_bdg)
    member_bidang.append(memberships_bidang)
    
#bobot pangkat
m_fuzzy = 2
#ini bobot yang dijadikan perdata uji    
bobot_bidang=[]
pangkat = 2 / (m_fuzzy - 1)
for (idx,i) in enumerate (jarak_tetangga_bidang):
    xbb = 0
    bbb=[]
    for (idy, j) in enumerate (i.items()):
        try:
            xbb = 1 / pow(j[1][1], pangkat)
        except:
            xbb = 0
        bbb.append(xbb)
    bobot_bidang.append(bbb)

bobot_jml_bidang=[]
for (idx,i) in enumerate (bobot_bidang):
    bobot=0
    for (idy,j) in enumerate (i):
        bobot+=j
    bobot_jml_bidang.append(bobot)
  
#Himpunan Fuzzy
hasil_himp_fuzzy_bidang=[]
for (idx,i) in enumerate (member_bidang):
    fuzzy_bobot_bidang=(bobot_bidang[idx])
    dict_himp_fuzzy_bidang={}
    for (idy,j) in enumerate (i):
        #print(j.items()) 
        temp_bidang={}
        #print(fuzzy_bobot[idy])
        for(idz,k) in enumerate (j.items()):
            #print(k)
            try:
                hasil_kali_fuzzy_bidang = (fuzzy_bobot_bidang[idy] * k[1]) / bobot_jml_bidang[idx]
                #print(hasil_kali_fuzzy)
                temp_bidang[k[0]]=(hasil_kali_fuzzy_bidang)
            except:
                temp_bidang[k[0]]=0
        dict_himp_fuzzy_bidang[idy]=(temp_bidang)
    hasil_himp_fuzzy_bidang.append(dict_himp_fuzzy_bidang)

#Predict
#jumlahkan per kelas
total_fuzzy_bidang=[]
for (idx,i) in enumerate (hasil_himp_fuzzy_bidang):
    temp_bidang={}
    for key,value in i.items():
        for k in value.keys():
            #print(value[k])
            temp_bidang[k] = temp_bidang.get(k,0) + value[k]
    total_fuzzy_bidang.append(temp_bidang)

hasil_prediksi_bidang=[]
for idx,i in enumerate (total_fuzzy_bidang):
    predict_bidang=(max(i.items(), key=operator.itemgetter(1))[0])
    hasil_prediksi_bidang.append(predict_bidang) 
 
'''PENGUJIAN BIDANG'''
labelsB = list(set(y_test_bidang))
labelsB.sort()

cfB = pd.DataFrame(
    data=confusion_matrix(y_test_bidang, hasil_prediksi_bidang, labels=labelsB),
    columns=labelsB,
    index=labelsB
)
cfB

tpsB = {}
fpsB = {}
fnsB = {}
tnsB = {}
for label in labelsB:
    tpsB[label] = cfB.loc[label, label]
    fpsB[label] = cfB[label].sum() - tpsB[label]
    fnsB[label] = cfB.loc[label].sum() - tpsB[label]
    
#akurasi dengan TN
for label in set(y_test_bidang):
    tnsB[label] = len(y_test_bidang) - (tpsB[label] + fpsB[label] + fnsB[label])
    
total_true_bidang = sum(list(tpsB.values()) + list(tnsB.values()))
total_predictions_bidang = sum(list(tpsB.values()) + list(tnsB.values()) + list(fpsB.values()) + list(fnsB.values()))
accuracy_global_new_bidang = total_true_bidang / total_predictions_bidang if total_predictions_bidang > 0. else 0.
print("Accuracy Bidang:", accuracy_global_new_bidang*100)

#presisi recall micro
correct_predictionsB = sum(tpsB.values())
tpfpB = sum(list(tpsB.values())+list(fpsB.values()))
tpfnB = sum(list(tpsB.values())+list(fnsB.values()))
precisionBidang =  correct_predictionsB / tpfpB if tpfpB > 0. else 0
print("Precision Bidang: ",precisionBidang*100)
recallBidang =  correct_predictionsB / tpfnB if tpfnB > 0. else 0
print("Recall Bidang: ",recallBidang*100)

#diagram hasil pengujian
objects = ['akurasi', 'presisi', 'recall'] 
y_pos = np.arange(len(objects))
performanceBidang = [accuracy_global_new_bidang, precisionBidang, recallBidang]

plt.bar(y_pos, performanceBidang, align='center', alpha=0.5, color = ['red', 'green', 'blue'])
plt.xticks(y_pos, objects)
plt.ylabel('Nilai (%)')
plt.title('Kategori Bidang\n 90:10\n k=11')
plt.show()

print(sum(tnsB.values()))
print(sum(tpsB.values()))
print(sum(fnsB.values()))
print(sum(fpsB.values()))
print(sum([1,2,3]+[1,1,1]))
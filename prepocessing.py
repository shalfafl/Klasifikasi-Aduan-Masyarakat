# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 06:58:47 2019

@author: Shalfa-PC
"""

import pandas as pd #untuk load data
import re #regular expression untuk preprocess
from nltk.tokenize import word_tokenize 
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import mysql.connector as db
import sqlalchemy #dr pandas masukin ke db butuh conector ini

mydb = db.connect(
  host="localhost",
  user="root",
  passwd="",
  database="ta"
)
mycursor = mydb.cursor()
#load data
data = pd.read_sql('SELECT * FROM data_upik', con=mydb)
#lowercase : menyingkat fungsi untuk split tokenisasi trs di lowercase trs digabung jd kalmiat lg
data['Pesan'] = data['Pesan'].apply(lambda x: " ".join(x.lower() for x in x.split())) 
                                        #di split dulu, lowercase, baru join lg
#cleansing
data['Pesan'] = data['Pesan'].str.replace("[^a-zA-Z]", " ") #selain a-z dihapus

#tokenisasi
tokenisasi=[]
for row in data['Pesan']: #perpesan =  2759 
    token = word_tokenize(row)
    tokenisasi.append(token)

#spelling_normalization
table={} #buat tabel dan sinonimnya dulu di coding jd berpasang-pasang 1 kata salah - 1 kata benar
with open('daftar_sinonim.txt','r') as syn:
    for row in syn:
        match=re.match(r'(\w+)\s+=\s+(.+)',row) #buat nyocokin
        if match:
            primary,synonyms=match.groups() #di grupin kata asli dan kata salah
            synonyms=[synonym.lower() for synonym in synonyms.split()] 
            #sinonim dilowercase dan disendiri2-in. jd berpasang-pasang.
           #print(synonyms) #daftar kata ejanya
            for synonym in synonyms:
                table[synonym]=primary.lower()
                #print(table[synonym]) #kata yg salah td dikata benarkan
              
#cek + ganti sinonim
spelling=[]
for idx,value in enumerate (tokenisasi):
    temp=[]
    for idy,value1 in enumerate (value):
        temp.append(''.join(table.get(word.lower(),word) for word in re.findall(r'(\W+|\w+)',value1)))
        #tempt tu list kosong trs mau dikasih nilai dg cara join di word, word itu perbaikan dr value1
    spelling.append(temp)
    
#stopword_removal   
stop_factory = StopWordRemoverFactory()
data_stopword = stop_factory.get_stop_words()
#print(data_stopword) #liat daftar stopword
stopword = stop_factory.create_stop_word_remover()
stopword_removal=[]
for idx,value in enumerate (spelling):
    temp=[]
    for idy,value1 in enumerate (value):
        temp.append(stopword.remove(value1))
    stopword_removal.append(temp)
    
#stemming:
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stemming=[]
for idx,value in enumerate (stopword_removal):
    temp=[]
    for idy,value1 in enumerate (value):
        temp.append(stemmer.stem(value1))
    stemming.append(temp)
    
#hasil stemming menghasilkan ada index yg gapunya nilai.. jd biar disatuin jd butuh hasil_preprocess
pesan_preprocess=[]
for idx,value in enumerate (stemming):
    punctuations = ''' '''
    no_punct = ""
    for idy,value1 in enumerate (value):
        if value1 not in punctuations:
            no_punct = no_punct + value1 + ' '
        k = no_punct
    pesan_preprocess.append(k)

#jadiin satu di data
data['Pesan'] = pd.Series(pesan_preprocess)

database_username = 'root'
database_password = ''
database_ip       = 'localhost'
database_name     = 'ta'
database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.format(database_username, database_password, database_ip, database_name))

data.to_sql(con=database_connection, name='data_preprocess', if_exists='replace',index=False)
 
print("Jumlah sms: ", len(pesan_preprocess))
label_bidang=data['Bidang'].drop_duplicates()
print("Jumlah bidang: ", (len(label_bidang)))
label_unit=data['Unit_Kerja'].drop_duplicates()
print("Jumlah unit: ", (len(label_unit)))
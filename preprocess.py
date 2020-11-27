import csv
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

file = {}
crimefile = open("/home/usuario/Documentos/UNIR/AplicacaoKeepDistance/database/bancodedados", 'r')
reader = csv.reader(crimefile)
allRows = [row for row in reader]
word = 'Classe'
maior = 0
for i in range(len(allRows)):
    if allRows[i][0] != "Round":
        if len(allRows[i][0]) == 17:
             MAC = allRows[i][0]
             file[MAC] = []
            
file[word] = []        
for i in range(len(allRows)):
    if allRows[i][0] != "Round":
        if len(allRows[i][0]) == 17:
            MAC = allRows[i][0]
            SIG = allRows[i+1][0]
            file[MAC].append(SIG)
            if len(file[MAC]) > maior:
                maior = len(file[MAC])
                
    elif allRows[i][0] == "Round":
        local = allRows[i+1][0]
        for j in file:
            while len(file[word]) != maior:
                file[word].append(local)
            while len(file[j])!= maior:
                file[j].append('0')
        
            
with open('saida.json', 'w') as outfile:
    json.dump(file, outfile)

df = pd.read_json (r"/home/usuario/Documentos/UNIR/AplicacaoKeepDistance/database/saida.json")
export_csv = df.to_csv (r"/home/usuario/Documentos/UNIR/AplicacaoKeepDistance/database/saida.csv", index = None, header=None)

dados = pd.read_csv("/home/usuario/Documentos/UNIR/AplicacaoKeepDistance/database/saida.csv", header=None)
previsores = dados.iloc[:, 0:37].values
classe = dados.iloc[:, 37].values

scaler = MinMaxScaler()
data = scaler.fit_transform(previsores)

le = LabelEncoder()
classe = le.fit_transform(classe)
classe = np_utils.to_categorical(classe)
aux = []
for i in range(len(classe)):
    for j in range(0,5):
        if classe[i][j] == 1:
            aux.append(j)
        

data = np.c_[data, aux]

np.random.shuffle(data)
np.savetxt(fname="/home/usuario/Documentos/UNIR/AplicacaoKeepDistance/database/saida.csv", X=data, fmt="%.15f", delimiter=",")

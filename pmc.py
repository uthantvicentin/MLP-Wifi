import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import scikitplot as skplt


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Erro Treinamento (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Erro Validação (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    #plt.title('Erro')
    plt.xlabel('Época')
    plt.ylabel('Erro')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Acurácia Treino (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Acurácia Validação( ' + str(format(history.history[l][-1],'.5f'))+')')

    #plt.title('Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()


dados = pd.read_csv("/home/usuario/Documentos/UNIR/AplicacaoKeepDistance/database/saida.csv", header=None)

previsores = dados.iloc[:, 0:37].values
classe = dados.iloc[:, 37].values

le = LabelEncoder()
classe = le.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)
def ANN():
    classificador = Sequential()
    classificador.add(Dense(units = 56, activation = 'sigmoid', 
                                input_dim = 37))
    classificador.add(Dropout(0.2))
           
    for i in range(6):
        classificador.add(Dense(units = 56, activation = 'sigmoid'))
        classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 5, activation = 'softmax'))
    classificador.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
    return classificador

#classificador.fit(previsores, classe_dummy, epochs = 500, batch_size = 2)


classifier = KerasClassifier(build_fn = ANN, epochs = 500, batch_size = 2)

resultados = cross_val_score(estimator = classifier, X = previsores,
                             y = classe, scoring = 'accuracy')

history = classifier.fit(previsores, classe, validation_split = 0.3, batch_size = 2)
predictions = cross_val_predict(classifier, previsores, classe)
skplt.metrics.plot_confusion_matrix(classe, predictions, normalize=True,
                                    title="Matriz de Confusão")



plt.show()

plot_history(history)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf 
from tensorflow.keras.models import Sequential


""" Ucitavanje dataseta """
data = pd.read_csv('texture.csv')


""" Sredjivanje izlaza - klasa """
data['class'].unique()


""" Provera da li ima duplikata unutar dataseta """
tot = len(set(data.index))
last = data.shape[0] - tot
print('Ima {} duplikata.\n'.format(last))


""" Provera da li ima null vrednosti unutar dataseta """
null_count = 0
for val in data.isnull().sum():
    null_count += val
print('Ima {} null vrednosti.\n'.format(null_count))


""" Prikaz broja vrednosti unutar odredjenog atributa """
def show_features(data):
    col_count, col_var = [], []
    for col in data:
        col_count.append(len(data[col].unique()))
        col_var.append(data[col].unique().sum())
    data_dict = {'Broj': col_count, 'Promenljive': col_var}
    data_table = pd.DataFrame(data_dict, index=data.columns)
    print(data_table)
     
show_features(data) 


""" Informacije u vezi podataka """
data.info()


""" Graficki prikaz atributa  """
plt.title('Odnos broja primeraka po pripadnosti')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['class'].plot(kind='hist', color= "red")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A1')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A1'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A2')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A2'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A3')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A3'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A4')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A4'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A5')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A5'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A6')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A6'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A7')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A7'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A8')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A8'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A9')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A9'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A10')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A10'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A11')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A11'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A12')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A12'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A13')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A13'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A14')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A14'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A15')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A15'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A16')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A16'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A17')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A17'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A18')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A18'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A19')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A19'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A20')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A20'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A21')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A21'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A22')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A22'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A23')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A23'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A24')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A24'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A25')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A25'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A26')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A26'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A27')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A27'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A28')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A28'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A29')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A29'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A30')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A30'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A31')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A31'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A32')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A32'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A33')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A33'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A34')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A34'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A35')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A35'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A36')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A36'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A37')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A37'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A38')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A38'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A39')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A39'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()

plt.title('Atribut-A40')
plt.ylabel('Broj pojavljivanja')
plt.xlabel('Pripadnost klasama')
data['A40'].plot(kind='hist', color= "green")
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.show()


""" Odredjivanje ulaznih i izlaznih podataka - Atributi:Klase """
y = data["class"].values                                                                    
x = data.drop(["class"], axis=1).values                                                     
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)      


""" Logistic Regression """
lr = LogisticRegression(solver="lbfgs")
lr.fit(x_train, y_train)
print("Logistic Regression Classification preciznost testa: {}%".format(round(lr.score(x_test, y_test)*100, 2)))
print('\n')


""" Random Forest Classifier """
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
print("Random Forest Classifier preciznost testa: {}%".format(round(rf.score(x_test, y_test)*100, 2)))


""" Crtanje konfuzionih matrica """
""" lr """
y_pred_lr = lr.predict(x_test)
y_true_lr = y_test
cm = confusion_matrix(y_true_lr, y_pred_lr)
f, ax = plt.subplots(figsize =(5, 5))
sns.heatmap(cm, annot = True, linewidths=0.5, linecolor="red", fmt = ".0f",ax=ax)
plt.xlabel("y_pred_lr")
plt.ylabel("y_true_lr")
plt.show()


""" rf """
y_pred_rf = rf.predict(x_test)
y_true_rf = y_test
cm = confusion_matrix(y_true_rf, y_pred_rf)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm, annot = True, linewidths=0.5,linecolor="red", fmt = ".0f", ax=ax)
plt.xlabel("y_pred_rf")
plt.ylabel("y_true_rf")
plt.show()


""" Treniranje neuronske mreze """
print("\n Neuronska mreza kroz epohe: \n")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledx_train = scaler.fit_transform(x_train)
rescaledx_test = scaler.fit_transform(x_test)
rescaledx_val = scaler.fit_transform(x_val)

x_train_normal = rescaledx_train
x_val_normal = rescaledx_val

tf.random.set_seed(0)
model = tf.keras.models.Sequential(layers=[tf.keras.layers.Dense(40, activation="relu"),
                                          tf.keras.layers.Dense(100, activation='relu'),
                                          tf.keras.layers.Dense(15, activation="softmax")])

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), optimizer = tf.keras.optimizers.SGD(0.1), metrics=["accuracy"])
proces = model.fit(x_train_normal, y_train, epochs=20, validation_data=(x_val_normal, y_val))
print(model.summary())

""" Evaluacija podataka """
print ("\n")
print("Evaluacija na testiranim podacima")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)
print("\n")
print("Generisanje predvidjanja na 4 uzorka")
predictions = model.predict(x_test[:4])
print("oblik predvidjanja:", predictions.shape)

import os
import random
import sys
import glob 
import librosa
import keras
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import model_from_json
from keras.utils import to_categorical

input_duration=3
# Create DataFrame for Data intel
data_df = pd.DataFrame(columns=['path', 'source', 'actor', 'gender',
                                'intensity', 'statement', 'repetition', 'emotion'])
dir_list = os.listdir('data/')
dir_list.sort()
count = 0
for i in dir_list:
    file_list = os.listdir('data/' + i)
    for f in file_list:
        nm = f.split('.')[0].split('-')
        path = 'data/' + i + '/' + f
        src = int(nm[1])
        actor = int(nm[-1])
        emotion = int(nm[2])
        
        if int(actor)%2 == 0:
            gender = "female"
        else:
            gender = "male"
        
        if nm[3] == '01':
            intensity = 0
        else:
            intensity = 1
        
        if nm[4] == '01':
            statement = 0
        else:
            statement = 1
        
        if nm[5] == '01':
            repeat = 0
        else:
            repeat = 1
            
        data_df.loc[count] = [path, src, actor, gender, intensity, statement, repeat, emotion]
        count += 1
# defining the emotion classes
label_list = []
for i in range(len(data_df)):
    if data_df.emotion[i] == 2:
        lb = "_calm"
    elif data_df.emotion[i] == 3:
        lb = "_happy"
    elif data_df.emotion[i] == 4:
        lb = "_sad"
    elif data_df.emotion[i] == 5:
        lb = "_angry"
    elif data_df.emotion[i] == 6:
        lb = "_fearful"    
    else:
        lb = "_none"
    label_list.append(data_df.gender[i] + lb)    
data_df['label'] = label_list


data2_df = data_df.copy()
data2_df = data2_df[data2_df.label != "male_none"]
data2_df = data2_df[data2_df.label != "female_none"].reset_index(drop=True)

#we separate the data into training and test data, speech of actor 1-20 were used for training and validation data
#speech of actor 21-24 were used for test data
tmp1 = data2_df[data2_df.actor == 21]
tmp2 = data2_df[data2_df.actor == 22]
tmp3 = data2_df[data2_df.actor == 23]
tmp4 = data2_df[data2_df.actor == 24]
data3_df = pd.concat([tmp1, tmp3],ignore_index=True).reset_index(drop=True)
data2_df = data2_df[data2_df.actor != 21]
data2_df = data2_df[data2_df.actor != 22]
data2_df = data2_df[data2_df.actor != 23].reset_index(drop=True)
data2_df = data2_df[data2_df.actor != 24].reset_index(drop=True)

#data2_df = training and validation data
#data3_df = test data

#for training and validation data
print("Extracting features for training and validation data: ")
data = pd.DataFrame(columns=['feature'])
data1 = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(data2_df))):
    X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate),axis=0)
    feature = mfccs
    feature1 = mel
    data1.loc[i] = [feature1]
    data.loc[i] = [feature]
df3 = pd.DataFrame(data['feature'].values.tolist())
df_3 = pd.DataFrame(data1['feature'].values.tolist())
labels = data2_df.label
newdf = pd.concat([df3, df_3, labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})
rnewdf.isnull().sum().sum()
rnewdf = rnewdf.fillna(0)

#for testing data
print("Extracting features for testing data: ")
datax = pd.DataFrame(columns=['feature'])
datax1 = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(data3_df))):
    X_t, sample_rate = librosa.load(data3_df.path[i], res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X_t, sr=sample_rate, n_mfcc=13), axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X_t, sr=sample_rate),axis=0)
    feature = mfccs
    feature1 = mel
    datax1.loc[i] = [feature1]
    datax.loc[i] = [feature]
df3 = pd.DataFrame(datax['feature'].values.tolist())
df_3 = pd.DataFrame(datax1['feature'].values.tolist())
labels = data3_df.label
testnewdf = pd.concat([df3, df_3, labels], axis=1)
rtestnewdf = testnewdf.rename(index=str, columns={"0": "label"})
rtestnewdf.isnull().sum().sum()
rtestnewdf = rtestnewdf.fillna(0)

# since the amount of data is quite less to train the model, 
# we apply different data augmentation techniques to increase the data
def noise(data):
    
    noise_amp = 0.005*np.random.uniform()*np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data
    
def shift(data):
   
    s_range = int(np.random.uniform(low=-5, high = 5)*500)
    return np.roll(data, s_range)
    
def stretch(data, rate=0.8):

    data = librosa.effects.time_stretch(data, rate)
    return data
    
def pitch(data, sample_rate):
    
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    data = librosa.effects.pitch_shift(data.astype('float64'), 
                                      sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
    return data

def dyn_change(data):
    
    dyn_change = np.random.uniform(low=1.5,high=3)
    return (data * dyn_change)
    
def speedNpitch(data):
    
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.0  / length_change
    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data

#1st augmentation
print("Extracting features for training and validation data when augmented with noise: ")
syn_data1 = pd.DataFrame(columns=['feature', 'label'])
syn_data12 = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(data2_df))):
    X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
    if data2_df.label[i]:
        X = noise(X)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate),axis=0)
        feature = mfccs
        feature11 = mel
        syn_data12.loc[i] = [feature11]
        syn_data1.loc[i] = [feature, data2_df.label[i]]

#2nd augmentation
print("Extracting features for training and validation data when augmented with pitch tuning: ")
syn_data2 = pd.DataFrame(columns=['feature', 'label'])
syn_data22 = pd.DataFrame(columns=['feature'])
for i in tqdm(range(len(data2_df))):
    X, sample_rate = librosa.load(data2_df.path[i], res_type='kaiser_fast',duration=input_duration,sr=22050*2,offset=0.5)
    if data2_df.label[i]:
        X = pitch(X, sample_rate)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate),axis=0)
        feature21 = mel
        feature = mfccs
        syn_data22.loc[i] = [feature21]
        syn_data2.loc[i] = [feature, data2_df.label[i]]

syn_data1 = syn_data1.reset_index(drop=True)
syn_data2 = syn_data2.reset_index(drop=True)
syn_data12 = syn_data12.reset_index(drop=True)
syn_data22 = syn_data22.reset_index(drop=True)

df4 = pd.DataFrame(syn_data1['feature'].values.tolist())
df41 = pd.DataFrame(syn_data12['feature'].values.tolist())
labels4 = syn_data1.label
syndf1 = pd.concat([df4, df41, labels4], axis=1)
syndf1 = syndf1.rename(index=str, columns={"0": "label"})
syndf1 = syndf1.fillna(0)

df4 = pd.DataFrame(syn_data2['feature'].values.tolist())
df41 = pd.DataFrame(syn_data22['feature'].values.tolist())
labels4 = syn_data2.label
syndf2 = pd.concat([df4, df41, labels4], axis=1)
syndf2 = syndf2.rename(index=str, columns={"0": "label"})
syndf2 = syndf2.fillna(0)

# Combining the Augmented data with original
combined_df = pd.concat([rnewdf, syndf1, syndf2], ignore_index=True)
combined_df = combined_df.fillna(0)
#  Stratified Shuffle Split
y = combined_df.label
X = combined_df.drop(['label'], axis=1)
split_var = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)
for train_index, val_index in split_var.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

y_test = rtestnewdf.label
X_test = rtestnewdf.drop(['label'], axis=1)
X_test = np.array(X_test)
y_test = np.array(y_test)


X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_val = np_utils.to_categorical(lb.fit_transform(y_val))

x_traincnn = np.expand_dims(X_train, axis=2)
x_valcnn = np.expand_dims(X_val, axis=2)

x_testcnn = np.expand_dims(X_test, axis=2)
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# defining CNN model

model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(5)) # Edit according to target class number
model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

# Compiling the model

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Model Training
#saving the best model
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)
mcp_save = ModelCheckpoint('aug_noiseNpitch_2class2_np.h5', save_best_only=True, monitor='val_loss', mode='min')
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=600, validation_data=(x_valcnn, y_val))

#saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#load model into json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("aug_noiseNpitch_2class2_np.h5")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
y_pred = loaded_model.predict(x_testcnn, batch_size=16, verbose=1)

#converting y_pred into class labels
pred=y_pred.argmax(axis=1)
a =pred.astype(int).flatten()
predictions = (lb.inverse_transform((a)))
preddf = pd.DataFrame({'predictedvalues': predictions})

#converting y_test into class labels
actual=y_test.argmax(axis=1)
a1 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((a1)))
actualdf = pd.DataFrame({'actualvalues': actualvalues})

finaldf = actualdf.join(preddf)

y_true = finaldf.actualvalues
y_pred = finaldf.predictedvalues
accuracy_score(y_true, y_pred)*100
print("Accuracy on test data: {0}, F1-score on test data: {1}".format(accuracy_score(y_true, y_pred)*100, f1_score(y_true, y_pred, average='macro') *100))

    
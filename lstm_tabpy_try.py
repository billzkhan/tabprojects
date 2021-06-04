import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
%matplotlib inline

from numpy.random import seed
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

# seed(10)
# tf.random.set_seed(10)

speed_1 = pd.read_csv('D:/BILAL/5. Flooding/daejon network master data/MRT_BASE_TF_INFO_5MN_20200730.csv')

#taking out one link speed w.r.t time
speed_1 = speed_1[(speed_1['LINK_ID']==1830001905)]
#datetime editing
speed_1['time'] = speed_1['HH_ID'].astype(str)+':'+speed_1['MN_ID'].astype(str)
speed_1['YMD_ID'] = speed_1['YMD_ID'].astype(str)
speed_1['date'] = speed_1['YMD_ID']+ " "+ speed_1['time']
speed_1[['date']] = speed_1[['date']].apply(pd.to_datetime, format='%Y%m%d %H:%M:%S.%f')
#converting long to wide dataframe
speed_1 = speed_1.pivot(index='date', columns='LINK_ID', values='TRVL_SPD').reset_index()
speed_1.columns = ['date','link_1']
#final data
df = speed_1
df.to_csv("D:/BILAL/link_2.csv")

train,test = df.loc[df['date'] <= '2020-07-30 18:40:00'], df.loc[df['date']> '2020-07-30 18:40:00']
train.set_index('date', inplace=True)
test.set_index('date', inplace=True)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.fit_transform(test)
scaler_filename = 'scaler_data'
joblib.dump(scaler,scaler_filename)

X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

#define autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1],X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True,
              kernel_regularizer= regularizers.l2(0.00))(inputs)
    L2 = LSTM(4,activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation ='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation ='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model

model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()

#fit model
model.fit(X_train,X_train, epochs=100,batch_size=16, validation_split=0.05,verbose=1) #why used trainX two times

import tabpy_client
from tabpy.tabpy_tools.client import Client
client = tabpy_client.Client('http://localhost:9004/')

def testing(_arg1,_arg2):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

      
    d = {'date':_arg1,'link_1':_arg2}
    df = pd.DataFrame(data=d, index=[0])
    # d[['date']] = d[['date']].apply(pd.to_datetime, format='%Y%m%d %H:%M:%S.%f')
    df.set_index('date', inplace=True)
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    df = df.reshape(df.shape[0],1,df.shape[1])
    x_pred = model.predict(df)
    x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[2])
    x_pred = np.concatenate(x_pred, axis=0 )
    return np.array(x_pred)

    # return x_pred.tolist()

testing('2020-07-30 18:40:00', 20)
client.deploy('testing', testing,'Predicts anomaly_pred', override = True)


































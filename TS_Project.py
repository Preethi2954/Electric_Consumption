#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import (Input, LSTM, Dense, Concatenate, AdditiveAttention)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[2]:


df = pd.read_csv("Electric_Consumption.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df["Datetime"] = pd.to_datetime( df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")

df.drop(columns=["Date", "Time"], inplace=True)
df.set_index("Datetime", inplace=True)


# In[5]:


df.replace("?", np.nan, inplace=True)
df = df.astype(float)
df = df.resample("h").mean()
df = df.dropna()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values.astype(np.float32))

print("NaNs in scaled_data:", np.isnan(scaled_data).sum())


# In[9]:


def create_sequences(data, input_len=24, output_len=1):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, input_len=24, output_len=1)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# In[10]:


baseline = tf.keras.Sequential([LSTM(64, input_shape=(X.shape[1], X.shape[2])), Dense(1)])

baseline.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

baseline.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)


# In[11]:


# Encoder
encoder_inputs = Input(shape=(X.shape[1], X.shape[2]))
encoder_outputs, state_h, state_c = LSTM(64, return_sequences=True, return_state=True)(encoder_inputs)

# Decoder
decoder_inputs = Input(shape=(1, X.shape[2]))
decoder_outputs = LSTM(64, return_sequences=True)(decoder_inputs, initial_state=[state_h, state_c])

# Additive Attention (STABLE)
attention = AdditiveAttention()
context = attention([decoder_outputs, encoder_outputs])

# Concatenate
decoder_concat = Concatenate(axis=-1)([decoder_outputs, context])

# Final output
output = Dense(1)(decoder_concat)

# Model
attention_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)

attention_model.compile(optimizer=Adam(learning_rate=0.0003,clipnorm=1.0), loss="mse")

attention_model.summary()


# In[12]:


decoder_train_input = X_train[:, -1:, :]
decoder_test_input  = X_test[:, -1:, :]

attention_model.fit([X_train, decoder_train_input], y_train.reshape(-1, 1, 1), epochs=20, batch_size=32, validation_split=0.1, verbose=1)


# In[13]:


preds = attention_model.predict([X_test, decoder_test_input])

y_pred = preds.flatten()
y_true = y_test.flatten()

print("NaNs in y_true:", np.isnan(y_true).sum())
print("NaNs in y_pred:", np.isnan(y_pred).sum())

print("Attention MAE:", mean_absolute_error(y_true, y_pred))
print("Attention RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))


# In[14]:


baseline_preds = baseline.predict(X_test).flatten()

print("Baseline MAE:", mean_absolute_error(y_true, baseline_preds))
print("Baseline RMSE:", np.sqrt(mean_squared_error(y_true, baseline_preds)))


# In[15]:


plt.figure(figsize=(12,4))
plt.plot(y_true[:200], label="Actual")
plt.plot(y_pred[:200], label="Attention Prediction")
plt.legend()
plt.title("Attention vs Actual (First 200 Points)")
plt.show()


# In[ ]:





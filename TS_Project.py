#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[2]:


INPUT_LEN = 24
OUTPUT_LEN = 1
UNITS = 64
LR = 0.0003
EPOCHS_BASELINE = 10
EPOCHS_ATTENTION = 15
BATCH_SIZE = 32
START_TOKEN = 0.0


# In[3]:


def load_and_preprocess(path):

    df = pd.read_csv(path)

    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")

    df.drop(columns=["Date", "Time"], inplace=True)
    df.set_index("Datetime", inplace=True)

    df.replace("?", np.nan, inplace=True)
    df = df.astype(float)

    df = df.resample("h").mean().dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values.astype(np.float32))

    return scaled, scaler


# In[4]:


def create_sequences(data, input_len=24, output_len=1):

    X, y = [], []

    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len, 0])

    return np.array(X), np.array(y)


# In[5]:


def build_baseline(input_shape):

    model = tf.keras.Sequential([LSTM(UNITS, input_shape=input_shape), Dense(1)])

    model.compile(optimizer="adam", loss="mse")
    return model


# In[6]:


class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V  = Dense(1)

    def call(self, query, values):

        query = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[7]:


def build_attention_model(input_shape):

    encoder_inputs = Input(shape=input_shape)

    encoder_outputs, state_h, state_c = LSTM(UNITS, return_sequences=True, return_state=True)(encoder_inputs)

    decoder_inputs = Input(shape=(OUTPUT_LEN, 1))

    decoder_outputs, _, _ = LSTM(UNITS, return_sequences=True, return_state=True)(decoder_inputs, initial_state=[state_h, state_c])

    attention = BahdanauAttention(UNITS)

    context_vector, attention_weights = attention(decoder_outputs[:, -1, :], encoder_outputs)

    concat = Concatenate(axis=-1)([decoder_outputs[:, -1, :], context_vector])

    output = Dense(1)(concat)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[output, attention_weights])

    model.compile(optimizer=Adam(LR, clipnorm=1.0), loss=["mse", None])

    return model


# In[8]:


def build_decoder_input(y):

    decoder_input = np.zeros((len(y), OUTPUT_LEN, 1))
    decoder_input[:, 0, 0] = START_TOKEN

    if OUTPUT_LEN > 1:
        decoder_input[:, 1:, 0] = y[:, :-1]

    return decoder_input


# In[9]:


def rolling_split(X, y, window=6000, step=2000):

    for start in range(0, len(X) - window - step, step):
        end = start + window
        yield X[:end], y[:end], X[end:end+step], y[end:end+step]


# In[10]:


scaled_data, scaler = load_and_preprocess("Electric_Consumption.csv")
X, y = create_sequences(scaled_data, INPUT_LEN, OUTPUT_LEN)

baseline_rmse_scores = []
attention_rmse_scores = []

print("\n===== ROLLING ORIGIN EVALUATION =====")

for i, (X_tr, y_tr, X_val, y_val) in enumerate(rolling_split(X, y)):

    print(f"\n--- Window {i+1} ---")

    baseline = build_baseline((X.shape[1], X.shape[2]))
    baseline.fit(X_tr, y_tr, epochs=EPOCHS_BASELINE, batch_size=BATCH_SIZE, verbose=0)

    baseline_preds = baseline.predict(X_val).flatten()

    baseline_rmse = np.sqrt(mean_squared_error(y_val.flatten(), baseline_preds))
    attention_model = build_attention_model((X.shape[1], X.shape[2]))

    decoder_train = build_decoder_input(y_tr)
    decoder_val = build_decoder_input(y_val)

    attention_model.fit([X_tr, decoder_train], y_tr.reshape(-1, 1), epochs=EPOCHS_ATTENTION, batch_size=BATCH_SIZE, verbose=0)

    attention_preds = attention_model.predict([X_val, decoder_val])[0].flatten()

    attention_rmse = np.sqrt(mean_squared_error(y_val.flatten(), attention_preds))

    baseline_rmse_scores.append(baseline_rmse)
    attention_rmse_scores.append(attention_rmse)

    print("Baseline RMSE :", baseline_rmse)
    print("Attention RMSE:", attention_rmse)


# In[12]:


results = pd.DataFrame({"Model": ["Baseline LSTM", "Seq2Seq Attention"],
                        "Average Rolling RMSE": [np.mean(baseline_rmse_scores),
                                                 np.mean(attention_rmse_scores)]})
print("\n===== FINAL COMPARISON =====")
print(results)


# In[13]:


attention_extractor = Model(inputs=attention_model.inputs, outputs=attention_model.output[1])

weights = attention_extractor.predict([X_val[:1], decoder_val[:1]])


# In[14]:


plt.figure(figsize=(12, 2))
plt.imshow(weights[0].T, aspect="auto")
plt.title("Temporal Attention Distribution")
plt.xlabel("Past Timesteps")
plt.yticks([])
plt.colorbar()
plt.show()


print("""
Interpretation:
The attention distribution assigns higher importance to recent timesteps,
indicating strong short-term temporal dependency in electricity consumption.
Moderate weights across mid-range timesteps suggest the model also captures
daily usage patterns, while distant past values show diminishing influence.
""")


# In[16]:


plt.figure(figsize=(12,4))
plt.plot(y_true[:200], label="Actual")
plt.plot(y_pred[:200], label="Attention Prediction")
plt.legend()
plt.title("Attention vs Actual (First 200 Points)")
plt.show()


# In[ ]:





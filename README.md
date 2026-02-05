# Electric Consumption Forecasting using Seq2Seq with Bahdanau Attention

# Project Overview

This project implements a **Seq2Seq (Encoder–Decoder) LSTM model with Bahdanau Attention** for time-series forecasting of electric power consumption.  
The goal is to predict future electricity usage using historical hourly data while providing **interpretable attention-based insights** into model decisions.

The system compares:

- Baseline LSTM Model
- Seq2Seq Encoder–Decoder with Custom Bahdanau Attention

The architecture emphasizes **interpretability, modular design, and robust evaluation**, following best practices for deep learning in time-series forecasting.

# Objectives

- Forecast electric consumption using deep learning.
- Implement a custom **Bahdanau Attention mechanism**.
- Compare baseline vs attention-based models.
- Visualize attention weights to interpret temporal importance.
- Apply modular software engineering practices.
- Perform research-style evaluation.

# Model Architecture

# Encoder
- LSTM (64 units)
- Processes historical sequences (24-hour window)
- Outputs hidden states and context representation.

# Decoder
- LSTM initialized with encoder states.
- Generates prediction using previous timestep input.

# Attention Mechanism
Custom **Bahdanau Attention** computes alignment scores between:

- Decoder hidden state (query)
- Encoder outputs (values)

This produces a context vector highlighting important past time steps.

# Dataset

Dataset used: Electric Consumption from UCI

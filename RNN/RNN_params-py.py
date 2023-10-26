import numpy as np
import tensorflow as tf
import pandas as pd
import os
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Input, SimpleRNN, Dense, LSTM
from tensorflow.keras.models import Model

# Read data
path_read = "C:\\Users\\Propietario\\Desktop\\ib\\5-Maestría\\GenData-PCA-UMAP\\Datos\\Datos_PCA"
df_datasignals = pd.read_csv(path_read + "\\df_PCA_Signals.csv")

# Nos quedamos con los datos problematicos
df_datasignals_test = df_datasignals[df_datasignals['σs'] < 0.6]

# make sequences of lenght 10 with the two features pc1 and pc2
#df_datasignals_test = df_datasignals_test.drop(['Unnamed: 0'], axis=1)
datasignals = df_datasignals_test.drop(['σs'], axis=1)
datasignals = datasignals.drop(['lcm'], axis=1)
datasignals = datasignals.to_numpy()

dataparams = df_datasignals_test.drop(['pc1'], axis=1)
dataparams = dataparams.drop(['pc2'], axis=1)
dataparams = dataparams.to_numpy()

# Define the sequence length (n)
sequence_length = 100  # You can change this to your desired sequence length
num_sequences = len(dataparams) // sequence_length

# Reshape the data into sequences while maintaining the shape
sequences_signals = datasignals[:num_sequences * sequence_length].reshape(num_sequences, sequence_length, 2)
sequences_params = dataparams[:num_sequences * sequence_length].reshape(num_sequences, sequence_length, 2)

# Convert the list of sequences into a numpy array
sequences_signals = np.array(sequences_signals)
sequences_params = np.array(sequences_params)

# Define the input layers for the RNN
input_signals = Input(shape=(sequence_length, 2), name="input_signals")

# Define a deep RNN with LSTM layers
lstm_layer1 = LSTM(128, return_sequences=True, activation='selu', name='lstm_layer1')(input_signals)
lstm_layer2 = LSTM(64, return_sequences=True, activation='selu', name='lstm_layer2')(lstm_layer1)
lstm_layer3 = LSTM(32, return_sequences=True, activation='selu', name='lstm_layer3')(lstm_layer2)

# Define the output layer
output_layer = Dense(2, activation='linear', name="output")(lstm_layer3)

# Create the model
model = Model(inputs=input_signals, outputs=output_layer)

# Create the model
model = Model(inputs=input_signals, outputs=output_layer)

custom_optimizer = keras.optimizers.Adam(learning_rate=0.00005)  # Set your desired learning rate

# Compile the model
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

# Train the model
model.fit(sequences_signals, sequences_params, epochs=1000, batch_size=100, verbose=2)

# Make predictions
predicted_params = model.predict(sequences_signals)

# Get the data from sequences_params 
# and predicted_params into a single array
sequences_params = sequences_params.reshape(num_sequences * sequence_length, 2)
predicted_params = predicted_params.reshape(num_sequences * sequence_length, 2)

selected_indices = np.arange(0, len(sequences_params), 100)

# print(sequences_params[selected_indices ,0])

# Plot the original data and the predicted data
plt.figure(figsize=(10, 6))
plt.title("Original Data vs. Predicted Data")
plt.scatter(sequences_params[selected_indices, 0], sequences_params[selected_indices, 1], label="Original Data")
plt.scatter(predicted_params[selected_indices, 0], predicted_params[selected_indices, 1], label="Predicted Data")
plt.show()

# # Plot the original data and the predicted data
# plt.figure(figsize=(10, 6))
# plt.title("Original Data vs. Predicted Data")
# plt.xlabel("Time Steps")
# plt.ylabel("Values")
# # for i in range(len(sequences_signals)):
# plt.plot(sequences_params[1, :, 0], label="Original Data", linestyle='--')
# plt.plot(predicted_params[1, :, 0], label="Predicted Data", linestyle='-')
# plt.legend()
# plt.show()
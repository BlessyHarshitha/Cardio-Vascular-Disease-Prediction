import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional

# Step 1: Data Loading and Preprocessing
# Load the dataset
data = pd.read_csv('/mnt/data/heart disease.csv')

# Inspect data
print(data.head())

# Preprocessing
X = data.drop(columns=['target'])  # Assuming 'target' is the label
y = data['target']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Model Development

# Deep Neural Network (DNN)
def create_dnn():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Convolutional Neural Network (CNN)
def create_cnn():
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Recurrent Neural Network (RNN)
def create_rnn():
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Bidirectional RNN (BiRNN)
def create_birnn():
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Ensemble Model Training and Prediction

# Reshape data for CNN/RNN
X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create models
dnn_model = create_dnn()
cnn_model = create_cnn()
rnn_model = create_rnn()
birnn_model = create_birnn()

# Train models
dnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
cnn_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
rnn_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
birnn_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Step 4: Ensemble Predictions
dnn_pred = dnn_model.predict(X_test)
cnn_pred = cnn_model.predict(X_test_reshaped)
rnn_pred = rnn_model.predict(X_test_reshaped)
birnn_pred = birnn_model.predict(X_test_reshaped)

# Averaging predictions for ensemble
ensemble_pred = (dnn_pred + cnn_pred + rnn_pred + birnn_pred) / 4
ensemble_pred = np.where(ensemble_pred > 0.5, 1, 0)

# Step 5: Evaluation
accuracy = accuracy_score(y_test, ensemble_pred)
precision = precision_score(y_test, ensemble_pred)
recall = recall_score(y_test, ensemble_pred)
f1 = f1_score(y_test, ensemble_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

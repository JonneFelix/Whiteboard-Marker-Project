import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from collections import Counter
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Masking, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

csv_dir = r"C:\Users\Hermes\PycharmProjects\MLunddasQSamPC\venv\csvForLabeling"
epochen = 100
batchsize = 256
kernal_size = 7

# Read all CSV files into a list of dataframes
all_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith(".csv")]
dataframes = [pd.read_csv(f) for f in all_files]

labels = []
for df in dataframes:
    labels.append(df["Target"][0])

label = np.array(labels)

# Labels zu Zahlen machen, z.B. A zu 0, B zu 1 usw.
buchstaben_zu_zahlen = {chr(ord('A') + i): i for i in range(26)}
labels = [buchstaben_zu_zahlen[buchstabe] for buchstabe in labels]

scaled_dataframes = []
for df in dataframes:
    df = df.drop(["timestamp", "x", "y", "z", "la", "r", "i", "j", "k", "qa", "cleaned_z_std", "Target"], axis=1)
    df = df.dropna()  # Drop rows with NaN values
    df = df.astype(np.float32)  # Convert all columns to float32
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    scaled_dataframes.append(scaled_df)

dataTensors = [tf.convert_to_tensor(df) for df in scaled_dataframes]
dataTensors = [tf.pad(t, paddings=[[0, 400 - tf.shape(t)[0]], [0, 0]]) for t in dataTensors]

X = np.array(dataTensors)
y = np.array(labels)
y = to_categorical(y, num_classes=26)  # Ensure we have 26 classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

label_counts = Counter(labels)
print("Label counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

num_classes = 26

model = Sequential()
model.add(Input(shape=(400, X.shape[2])))
model.add(Masking(mask_value=0))
model.add(Conv1D(filters=32, kernel_size=kernal_size, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv1D(filters=64, kernel_size=kernal_size, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv1D(filters=128, kernel_size=kernal_size, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv1D(filters=256, kernel_size=kernal_size, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv1D(filters=512, kernel_size=kernal_size, activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ModelCheckpoint Callback to save the best model
checkpoint_filepath = 'best_model_temp.keras'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                   save_best_only=True,
                                   monitor='val_accuracy',
                                   mode='max',
                                   verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=epochen, batch_size=batchsize, validation_split=0.2,
          callbacks=[early_stopping, model_checkpoint])

# Load the best model saved during training
model.load_weights(checkpoint_filepath)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Save the best model with the specified naming convention
index = 1
filename = f'mein_modell{index}_accuracy{accuracy}.h5'
while os.path.exists(filename):
    index += 1
    filename = f"mein_modell{index}_accuracy{accuracy}.h5"
model.save(filename)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(buchstaben_zu_zahlen.keys()), yticklabels=sorted(buchstaben_zu_zahlen.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

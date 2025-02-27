import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import cv2
import random
import os
import imageio.v2 as imageio
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.resnet import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping

# Suppress warnings
import warnings
warnings.simplefilter("ignore", UserWarning)

# TensorFlow memory management (optional)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Set dataset directory
directory = r'dataset'
categories = ['Bengin cases', 'Malignant cases', 'Normal cases']

# Preprocessing images
img_size = 256
data = []

for category in categories:
    category_path = os.path.join(directory, category)
    class_num = categories.index(category)

    for file in os.listdir(category_path):
        file_path = os.path.join(category_path, file)
        img = cv2.imread(file_path, 0)  # Read in grayscale
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            data.append([img, class_num])

# Shuffle and separate features (X) and labels (y)
random.shuffle(data)
X = np.array([item[0] for item in data]).reshape(-1, img_size, img_size, 1)
y = np.array([item[1] for item in data])

# Normalize X
X = X / 255.0

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=10)

# Apply SMOTE to balance the training data
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten for SMOTE
smote = SMOTE()
X_train_sampled, y_train_sampled = smote.fit_resample(X_train_flat, y_train)
X_train_sampled = X_train_sampled.reshape(X_train_sampled.shape[0], img_size, img_size, 1)  # Reshape back

# Data generators
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(X_train_sampled, y_train_sampled, batch_size=8, shuffle=True)
validation_generator = val_datagen.flow(X_valid, y_valid, batch_size=8, shuffle=True)

# Load pre-trained ResNet50 model
resnet_base = ResNet50(weights=None, include_top=False, input_shape=X_train.shape[1:])
for layer in resnet_base.layers:
    layer.trainable = False  # Freeze pre-trained layers

# Build model
model = Sequential([
    resnet_base,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Assuming 3 categories
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Training
callback = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(train_generator, epochs=20, validation_data=validation_generator, callbacks=[callback])

# Save the model
model_save_path = "lung_cancer_detection_model.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluation
y_pred = model.predict(X_valid, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print("Classification Report:")
print(classification_report(y_valid, y_pred_bool))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true=y_valid, y_pred=y_pred_bool))

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

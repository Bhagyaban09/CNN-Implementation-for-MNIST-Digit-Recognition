#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models, utils
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Fetch dataset from UCI repository
optical_recognition = fetch_ucirepo(id=80)

# Extract data as pandas dataframes
X = optical_recognition.data.features
y = optical_recognition.data.targets

# Convert dataframes to numpy arrays if not already in that format
X = np.array(X)
y = np.array(y)

# Normalize and reshape data for CNN input
X = X.reshape((X.shape[0], 8, 8, 1)).astype('float32') / 16  # Images are 8x8 and pixel values range from 0 to 16
y = utils.to_categorical(y, 10)  # Assuming there are 10 classes

# Build the CNN Model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(8, 8, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# Prepare for K-Fold Cross Validation
kf = KFold(n_splits=5)
fold_no = 1
losses = []
accuracies = []

for train, test in kf.split(X):
    print(f'Training fold {fold_no}...')
    history = model.fit(X[train], y[train], 
                        batch_size=128, epochs=10, 
                        validation_data=(X[test], y[test]))
    
    scores = model.evaluate(X[test], y[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    losses.append(scores[0])
    accuracies.append(scores[1])
    fold_no += 1

# Average scores after cross-validation
print(f'Average loss: {np.mean(losses)}, Average Accuracy: {np.mean(accuracies)*100}%')

# Visualization of training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Predictions for Confusion Matrix
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred_classes))


# In[ ]:





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import canny
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from imblearn.over_sampling import RandomOverSampler
import time


# Load the FER2013 dataset
data = pd.read_csv('fer2013.csv')

# Extract the pixel values and labels
pixels = data['pixels'].tolist()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
labels = data['emotion'].tolist()

# Convert pixels to numpy array and normalize
images = np.array([np.fromstring(pixel, dtype='int', sep=' ') for pixel in pixels])
images = images.reshape(-1, 48, 48, 1)
images = images.astype('float32') / 255.0

# Apply Canny edge detection
edge_images = np.zeros_like(images)
for i in range(len(images)):
    edge_images[i, :, :, 0] = canny(images[i, :, :, 0])

# Convert labels to categorical format
labels = to_categorical(labels)

# Perform random oversampling
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(edge_images.reshape(len(edge_images), -1), labels)
X_resampled = X_resampled.reshape(-1, 48, 48, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Label the images in the training set
train_labels = [emotion_labels[np.argmax(label)] for label in y_train]
train_labels = np.array(train_labels)
# Display a few images with labels using seaborn
n_samples = 5  # Number of images to display
indices = np.random.choice(range(len(X_train)), n_samples, replace=False)
sample_images = X_train[indices]
sample_labels = train_labels[indices]

fig, axes = plt.subplots(1, n_samples, figsize=(12, 4))

for i, ax in enumerate(axes):
    ax.imshow(sample_images[i].reshape(48, 48), cmap='gray')
    ax.set_title(sample_labels[i])
    ax.axis('off')

plt.tight_layout()
plt.show()

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#start time
start_time=time.time()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=40)

training_time=time.time() - start_time

print("training time:",training_time)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_labels = [emotion_labels[np.argmax(label)] for label in y_pred]
y_test_labels = [emotion_labels[np.argmax(label)] for label in y_test]

# Compute the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=emotion_labels)

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Plot the validation accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

accuracy = accuracy_score(y_test_labels, y_pred_labels)

# Print prediction results in percentage
print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

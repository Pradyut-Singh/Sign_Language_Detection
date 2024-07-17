import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Constants
DATA_DIR = 'SavedImages'
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 10
CLASSES = ['A', 'B']


# Load and preprocess data
def load_data(data_dir, classes):
    images = []
    labels = []
    print(f"Loading images from {data_dir}...")

    # Iterate through class directories
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Error: Directory {class_dir} does not exist.")
            continue

        for filename in os.listdir(class_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(class_dir, filename)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: {img_path} could not be read.")
                    continue
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                images.append(image)
                labels.append(label)

    print(f"Loaded {len(images)} images.")

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)
    return images, labels


# Load the data
images, labels = load_data(DATA_DIR, CLASSES)

# Check if images and labels were loaded correctly
if images.size == 0 or labels.size == 0:
    raise ValueError("No images or labels were loaded. Check the data directory and image files.")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to categorical (one-hot encoding)
num_classes = len(CLASSES)
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

# Save the model
model.save('hand_detection_model.h5')

print("Model training completed and saved as 'hand_detection_model.h5'.")

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Constants
IMG_SIZE = 300
CLASSES = ['A', 'B']
MODEL_PATH = 'best_model.keras'
OFFSET = 20

# Load the trained model
model = load_model(MODEL_PATH)


def preprocess_image(image):
    """Preprocess an image for prediction."""
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype='float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def predict_image(model, image):
    """Predict the class of an image using the trained model."""
    image = preprocess_image(image)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return CLASSES[predicted_class[0]]


# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        imgCrop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]

        # Ensure the cropped image does not go out of bounds
        if 0 <= y - OFFSET < y + h + OFFSET <= img.shape[0] and 0 <= x - OFFSET < x + w + OFFSET <= img.shape[1]:
            try:
                predicted_class = predict_image(model, imgCrop)
                cv2.putText(img, f'Class: {predicted_class}', (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(img, (x - OFFSET, y - OFFSET), (x + w + OFFSET, y + h + OFFSET), (255, 0, 0), 2)
            except Exception as e:
                print(f"Error in prediction: {e}")
                cv2.putText(img, 'Error', (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, 'Out of bounds', (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


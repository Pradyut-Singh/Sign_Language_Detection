import cv2
import os
from cvzone.HandTrackingModule import HandDetector

# Create a directory to save images if it doesn't exist
save_dir = 'SavedImages/A'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
img_size = 300
img_count = 0

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        try:
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            imgCrop = cv2.resize(imgCrop, (img_size, img_size))
            cv2.imshow("ImageCrop", imgCrop)
        except Exception as e:
            print(f"Error cropping image: {e}")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('p') and hands:
        # Save the cropped image when 'p' is pressed
        save_path = os.path.join(save_dir, f"hand_{img_count}.jpg")
        cv2.imwrite(save_path, imgCrop)
        print(f"Saved {save_path}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()

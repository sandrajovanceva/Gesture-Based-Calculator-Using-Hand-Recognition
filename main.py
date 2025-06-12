import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('gesture_model.h5')

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)

classes = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/", "=", ""
]
text = ""

while True:
    success, img = cam.read()
    imgg = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x, y = [], []

            for lm in handLms.landmark:
                h, w, c = imgg.shape
                x.append(int(lm.x * w))
                y.append(int(lm.y * h))

            xmin, xmax = min(x) - 20, max(x) + 20
            ymin, ymax = min(y) - 20, max(y) + 20
            gesture_image = imgg[ymin:ymax, xmin:xmax]

            processed_gesture = cv2.resize(gesture_image, (128, 128))
            processed_gesture = processed_gesture / 255.0
            processed_gesture = np.expand_dims(processed_gesture, axis=0)

            predictions = model.predict(processed_gesture)
            class_id = np.argmax(predictions)
            gesture = classes[class_id]

            if gesture == "=":
                try:
                    text = str(eval(text))
                except:
                    text = "Error"
            else:
                text += gesture

            mpDraw.draw_landmarks(imgg, handLms, mpHands.HAND_CONNECTIONS)

    cv2.putText(imgg, text, (60, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
    cv2.imshow("Cam", imgg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

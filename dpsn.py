import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Вчитај го тренираниот модел
model = load_model('gesture_model.h5')

# Mediapipe за детекција на раце
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Камера
cam = cv2.VideoCapture(0)

# Листи за мапирање на класи
classes = ["", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "+", "-", "*", "/", "="]
text = ""

while True:
    success, img = cam.read()
    imgg = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x, y = [], []

            # Екстракција на координати
            for lm in handLms.landmark:
                h, w, c = imgg.shape
                x.append(int(lm.x * w))
                y.append(int(lm.y * h))

            # Подготовка на ROI (Region of Interest)
            xmin, xmax = min(x) - 20, max(x) + 20
            ymin, ymax = min(y) - 20, max(y) + 20
            roi = imgg[ymin:ymax, xmin:xmax]

            # Преобразба за моделот
            roi_resized = cv2.resize(roi, (128, 128))
            roi_resized = roi_resized / 255.0
            roi_resized = np.expand_dims(roi_resized, axis=0)

            # Препознавање на гест
            predictions = model.predict(roi_resized)
            class_id = np.argmax(predictions)
            gesture = classes[class_id]

            # Ажурирање на текстот
            if gesture == "=":
                try:
                    text = str(eval(text))
                except:
                    text = "Error"
            else:
                text += gesture

            mpDraw.draw_landmarks(imgg, handLms, mpHands.HAND_CONNECTIONS)

    # Прикажување на текстот
    cv2.putText(imgg, text, (60, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
    cv2.imshow("Cam", imgg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

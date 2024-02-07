import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('mymodel.h5')



def preprocess_face_image(face_img):
    face_img = cv2.resize(face_img, (150, 150))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img



cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        processed_face = preprocess_face_image(face_img)


        prediction = model.predict(processed_face)[0][0]


        if prediction > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), ((0, 255, 0)), 3)
            cv2.putText(frame, 'NO MASK', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (((0, 0, 255))), 3)
            cv2.putText(frame, 'MASK', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ((0, 0, 255)), 2)


    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
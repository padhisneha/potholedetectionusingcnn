import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame

pygame.init()
pygame.mixer.music.load('C:\\Users\\sneha padhi\\OneDrive\\Desktop\\RTRP\\beep.mp3')
# Load the pre-trained model
model = load_model('C:\\Users\\sneha padhi\\OneDrive\\Desktop\\RTRP\\model2.h5')

# Path to the video file
video_path = 'C:\\Users\\sneha padhi\\OneDrive\\Desktop\\RTRP\\DASH CAM 2016 01 29 (42 Miles of Potholes).mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to 64x64
    img_resized = cv2.resize(frame, (64, 64))
    
    # Preprocess the frame (normalize, convert to numpy array, etc.)
    img_resized = img_resized.astype('float32') / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Perform prediction on the preprocessed frame
    prediction_raw = model.predict(img_resized, verbose=0)
    maxArg = np.argmax(prediction_raw[0])
    pred = maxArg
    conf = prediction_raw[0][maxArg]
    #print(pred,conf)
    if pred == 0 and conf >=0.7:
        cv2.putText(frame, str('pothole'), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2, cv2.LINE_AA)
        pygame.mixer.music.play()
    # Display frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

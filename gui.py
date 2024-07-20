import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk
import pygame

# Load the pre-trained model
model = load_model('C:\\Users\\sneha padhi\\OneDrive\\Desktop\\RTRP\\model2.h5')

# Path to the video file
video_path = 'C:\\Users\\sneha padhi\\OneDrive\\Desktop\\RTRP\\DASH CAM 2016 01 29 (42 Miles of Potholes).mp4'

# Initialize Tkinter
root = tk.Tk()
root.title("Pothole Detection System")

# Create a Tkinter canvas to display the video feed
canvas = tk.Canvas(root, width=1240, height=600)
canvas.pack()

# Create a Tkinter label to display the detection result
label = tk.Label(root, text="", font=("Helvetica", 24))
label.pack()

# Initialize Pygame for sound playback
pygame.init()
pygame.mixer.music.load('C:\\Users\\sneha padhi\\OneDrive\\Desktop\\RTRP\\beep.mp3')

# Open the video file
cap = cv2.VideoCapture(video_path)

def process_frame():
    ret, frame = cap.read()
    if not ret:
        return

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

    # Display detection result
    if pred == 0 and conf >= 0.7:
        label.config(text="Pothole Detected!")
        pygame.mixer.music.play()
    else:
        label.config(text="")

    # Convert the frame to a Tkinter-compatible image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    # Display the frame on the canvas
    canvas.create_image(0, 0, image=img, anchor='nw')
    canvas.image = img

    # Schedule the next frame processing
    root.after(1, process_frame)

# Start processing frames
process_frame()

# Run the Tkinter event loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
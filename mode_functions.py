#Imports
import os
import random
import cv2
import numpy as np
from keras.models import load_model as LoadModel
from keras.utils import load_img, img_to_array
from keras.optimizers import RMSprop
from model_functions import train_model, showImagePrediction, image_width, image_height
from ultralytics import YOLO                                         
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog

#Global variables
images = []
index = 0
parent_frame = ""
model = ""

#Functions for image mode
def image_mode(parent):      
    global images, index, parent_frame, model
    parent_frame = parent 
                                                                                          # Runs the original program's image mode. Navigate thru images in the test set
    if os.path.exists("hotdog.h5"):
        model = LoadModel("hotdog.h5")
        print("Loaded model hotdog.h5")

        model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    )
    else:
        print("No saved model found. Training new model.")
        model = train_model()
        model.save("hotdog.h5")
        print("Model trained and saved as hotdog.h5")
        return
    

    #Load Test Image (random from test directory which has images in test/hotdog and test/nothotdog) and Predict
    hotdog_dir = "dataset/test/hotdog"
    nothotdog_dir = "dataset/test/nothotdog"

    #Load all images from the directories
    hotdog_images = os.listdir(hotdog_dir)
    nothotdog_images = os.listdir(nothotdog_dir)

    #Create two separate lists for hotdog and not hotdog images
    hotdog_images_labeled = [(os.path.join(hotdog_dir, img), 1) for img in hotdog_images]
    nothotdog_images_labeled = [(os.path.join(nothotdog_dir, img), 0) for img in nothotdog_images]

    #Combine the labeled images into a single list
    images = hotdog_images_labeled + nothotdog_images_labeled

    #Seed the random number generator
    random.seed()

    #Shuffle the images
    random.shuffle(images) #Devskim: ignore DS148264

    #Initialize the index to 0
    index = 0

    show_image(index)

#Function to show the image
def show_image(index):
    global parent_frame, model

    #Load the image
    try:
        test_image = load_img(
            images[index][0], target_size=(image_width, image_height))
    except Exception as e:
        print(f'Error loading image {images[index][0]}: {e}')
        return

    #Covert the image to a numpy array
    test_image = img_to_array(test_image)

    #Normalize the image
    test_image /= 255.0

    #Add a fourth dimension to the image (since Keraas excepts a list of images)
    test_image = np.expand_dims(test_image, axis=0)

    #Make a prediction
    result = model.predict(test_image)

    #Print Prediction
    if result[0][0] > 0.5:
        prediction = 'hotdog'
    else:
        prediction = 'not hotdog'

    print(f'Raw prediction: {result}')
    print(
        f'The image {images[index][0]} is a {prediction} with {result[0][0]} confidence')

    #Check if the prediction is correct
    correct_prediction = (
        (prediction == 'hotdog' and images[index][1] == 1) or
        (prediction == 'not hotdog' and images[index][1] == 0)
    )

    #Show the image with the prediction label
    showImagePrediction(parent_frame, images[index][0], prediction,
                        result[0][0], correct_prediction)
#Function to show the next image
def next_image():
    global index
    if not images:
        return
    index = (index + 1) % len(images)
    show_image(index)
#Function to show the previous image
def previous_image():
    global index
    if not images:
        return
    index = (index - 1) % len(images)
    show_image(index)
    

def boundbox_crop(keras_model,crop):                                                                                        # Take YOLO bounding box and apply model to each box
    crop_resized = cv2.resize(crop, (image_width, image_height))
    crop_array = img_to_array(crop_resized) / 255.0
    crop_array = np.expand_dims(crop_array, axis=0)
    result = keras_model.predict(crop_array, verbose=0)
    prediction = "hotdog" if result[0][0] > 0.5 else "not hotdog"
    confidence = result[0][0] if prediction == "hotdog" else 1 - result[0][0]
    return prediction, confidence

#Function to upload an image
def upload_image_mode(parent):
    global parent_frame, model
    parent_frame = parent

    #Clear the widgets
    for w in parent_frame.winfo_children():
        w.destroy()

    if os.path.exists("hotdog.h5"):
        model = LoadModel("hotdog.h5")
        print("Loaded model hotdog.h5")

        model.compile(
            loss='binary_crossentropy',
            optimizer=RMSprop(learning_rate=0.001),
            metrics=['accuracy']
        )
    else:
        print("No saved model found. Training new model.")
        model = train_model()
        model.save("hotdog.h5")
        print("Model trained and saved as hotdog.h5")

    #Pick an image file 
    image_path = filedialog.askopenfilename(
        title="Select an image",
        initialdir=os.getcwd(),
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )

    if not image_path:
        print("No image selected.")
        return

    try:
        test_image = load_img(image_path, target_size=(image_width, image_height))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    #Covert the image to a numpy array
    test_image = img_to_array(test_image)

    #Normalize the image
    test_image /= 255.0

    #Add a fourth dimension to the image (since Keraas excepts a list of images)
    test_image = np.expand_dims(test_image, axis=0)

    #Make a prediction
    result = model.predict(test_image)

    #Print Prediction
    if result[0][0] > 0.5:
        prediction = 'hotdog'
    else:
        prediction = 'not hotdog'

    print(f'Raw prediction: {result}')
    print(
        f'The image {images[index][0]} is a {prediction} with {result[0][0]} confidence')

    #Check if the prediction is correct
    correct_prediction = (
        (prediction == 'hotdog' and images[index][1] == 1) or
        (prediction == 'not hotdog' and images[index][1] == 0)
    )

    #Show the image with the prediction label
    showImagePrediction(parent_frame, images[index][0], prediction,
                        result[0][0], correct_prediction)


def video_mode(parent):                                       # determines if video contains a hotdog.
    videos_dir = "videos"
    if not os.path.exists(videos_dir):                              # looks if folder exists
        print("No videos folder found")
        return
    #Upload a video
    video_path = filedialog.askopenfilename(
        title="Select a video",
        initialdir=os.getcwd(),
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )

    video_file = os.path.basename(video_path)

    video_files = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]               # looks for videos in folder
    if not video_files:
        print("No video files found in 'videos' folder.")
        return
    
    if os.path.exists("hotdog.h5"):                                                                                                 # loads keras model
        keras_model = LoadModel("hotdog.h5")
        print("Loaded hotdog.h5 model")
        keras_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    else:
        print("No saved hotdog model found. Train first!")
        return


    yolo_model = YOLO("yolov8n.pt")                                                              # loads pre trained model

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open ", video_file)
        return

    print("Processing video:", video_file)

    for widget in parent.winfo_children():
        widget.destroy()

    video_label = ctk.CTkLabel(parent, text="Loading video...")
    video_label.pack(pady=10)

    DISPLAY_WIDTH = 550
    DISPLAY_HEIGHT = 550

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            print("Finished video")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # YOLO expects RGB

        # run YOLO + hotdog model
        for result in yolo_model.predict(frame_rgb, verbose=False):  # draws bounding box
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                prediction, confidence = boundbox_crop(keras_model, crop)
                color = (0, 255, 0) if prediction == 'hotdog' else (0, 0, 255)
                label = f"{prediction} ({confidence:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                cv2.putText(frame, label, (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        frame_rgb_disp = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb_disp)
        tk_img = ImageTk.PhotoImage(pil_img)

        video_label.configure(image=tk_img, text="")
        video_label.image = tk_img 

        parent.after(30, update_frame)

    update_frame()
    #Instructions
    instructions = ctk.CTkLabel(
        parent,
        text="Press Q to Quit to Home Screen",
        font=ctk.CTkFont(size=14)
    )
    instructions.pack(pady=(0, 10))

#Function for webcame mode
def webcam_mode(parent):
    #Clear the widgets
    for widget in parent.winfo_children():
        widget.destroy()

    if os.path.exists("hotdog.h5"):
        keras_model = LoadModel("hotdog.h5")
        print("Loaded hotdog.h5 model")
        keras_model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )
    else:
        print("No saved hotdog model found. Train first!")
        return

    yolo_model = YOLO("yolov8n.pt")

    #Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    print("Webcam mode started. Close the window to stop.")

    video_label = ctk.CTkLabel(parent, text="Starting webcam...")
    video_label.pack(pady=10)

    DISPLAY_WIDTH = 600
    DISPLAY_HEIGHT = 600

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam")
            cap.release()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for result in yolo_model.predict(frame_rgb, verbose=False):
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                pred, conf = boundbox_crop(keras_model, crop)
                color = (0, 255, 0) if pred == 'hotdog' else (0, 0, 255)
                label = f"{pred} ({conf:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        frame_rgb_disp = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb_disp)
        tk_img = ImageTk.PhotoImage(pil_img)

        video_label.configure(image=tk_img, text="")
        video_label.image = tk_img  # keep ref

        parent.after(30, update_frame)

    update_frame()
    #Instructions
    instructions = ctk.CTkLabel(
        parent,
        text="Press Q to Quit to Home Screen",
        font=ctk.CTkFont(size=14)
    )
    instructions.pack(pady=(0, 10))
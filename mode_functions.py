import os
import random
import cv2
import numpy as np
from keras.models import load_model as LoadModel
from keras.utils import load_img, img_to_array
from keras.optimizers import RMSprop
from model_functions import train_model, showImagePrediction, image_width, image_height
from ultralytics import YOLO                                         



def image_mode():                                                                                                       # Runs the original program's image mode. Navigate thru images in the test set
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
    

    # Load Test Image (random from test directory which has images in test/hotdog and test/nothotdog) and Predict
    hotdog_dir = "dataset/test/hotdog"
    nothotdog_dir = "dataset/test/nothotdog"

    # Load all images from the directories
    hotdog_images = os.listdir(hotdog_dir)
    nothotdog_images = os.listdir(nothotdog_dir)

    # Create two separate lists for hotdog and not hotdog images
    hotdog_images_labeled = [(os.path.join(hotdog_dir, img), 1) for img in hotdog_images]
    nothotdog_images_labeled = [(os.path.join(nothotdog_dir, img), 0) for img in nothotdog_images]

    # Combine the labeled images into a single list
    images = hotdog_images_labeled + nothotdog_images_labeled

    # Seed the random number generator
    random.seed()

    # Shuffle the images
    random.shuffle(images) # Devskim: ignore DS148264

    # Initialize the index to 0
    index = 0

    # Print user controls
    print("===== Usage Instructions =====")
    print("Press 'a' to view the previous image.")
    print("Press 'd' to view the next image.")
    print("Press any other key to exit.")
    print("================================\n")

    while True:
        # Load the image
        try:
            test_image = load_img(
                images[index][0], target_size=(image_width, image_height))
        except:
            print(f'Error loading image {images[index][0]}')
            images.pop(index)
            if index >= len(images):
                index = 0
            continue

        # Convert the image to a numpy array
        test_image = img_to_array(test_image)

        # Normalize the image
        test_image /= 255.0

        # Add a fourth dimension to the image (since Keras expects a list of images)
        test_image = np.expand_dims(test_image, axis=0)

        # Make a prediction
        result = model.predict(test_image)

        # Print Prediction
        if result[0][0] > 0.5:
            prediction = 'hotdog'
        else:
            prediction = 'not hotdog'
        print(f'Raw prediction: {result}')
        print(
            f'The image {images[index][0]} is a {prediction} with {result[0][0]} confidence')

        # Check if the prediction is correct
        correct_prediction = (prediction == 'hotdog' and images[index][1] == 1) or (
            prediction == 'not hotdog' and images[index][1] == 0)

        # Show the image with the prediction label
        showImagePrediction(images[index][0], prediction,
                            result[0][0], correct_prediction)

        # Wait for user input
        key = cv2.waitKey(0)

        # Move to the next or previous image based on user input
        if key == ord('a'):
            index = (index - 1) % len(images)
        elif key == ord('d'):
            index = (index + 1) % len(images)
        else:
            break

    # Close all windows
    cv2.destroyAllWindows()

def boundbox_crop(keras_model,crop):                                                                                        # Take YOLO bounding box and apply model to each box
    crop_resized = cv2.resize(crop, (image_width, image_height))
    crop_array = img_to_array(crop_resized) / 255.0
    crop_array = np.expand_dims(crop_array, axis=0)
    result = keras_model.predict(crop_array, verbose=0)
    prediction = "hotdog" if result[0][0] > 0.5 else "not hotdog"
    confidence = result[0][0] if prediction == "hotdog" else 1 - result[0][0]
    return prediction, confidence

def video_mode():                                       # determines if video contains a hotdog.
    videos_dir = "videos"
    if not os.path.exists(videos_dir):                              # looks if folder exists
        print("No videos folder found")
        return

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

    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open ", video_file)
            continue

        print("Processing video:", video_file)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                  # YOLO expects RGB images

            for result in yolo_model.predict(frame_rgb):                                            # draws bounding box
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    prediction, confidence = boundbox_crop(keras_model, crop)
                    color = (0, 255, 0) if prediction=='hotdog' else (0, 0, 255)                    #if it sees hot dog, green, if not, red
                    label = f"{prediction} ({confidence:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow(f"Hotdog Detection: {video_file}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def webcam_mode():
    if os.path.exists("hotdog.h5"):                                                                                                 # loads keras model
        keras_model = LoadModel("hotdog.h5")
        print("Loaded hotdog.h5 model")
        keras_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    else:
        print("No saved hotdog model found. Train first!")
        return


    yolo_model = YOLO("yolov8n.pt")                                                              # loads pre trained model

    cap = cv2.VideoCapture(0)                                                                    # opens webcam
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    print("Press 'q' to quit webcam mode.")

    while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                  # YOLO expects RGB images

            for result in yolo_model.predict(frame_rgb):                                            # draws bounding box
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    prediction, confidence = boundbox_crop(keras_model, crop)
                    color = (0, 255, 0) if prediction=='hotdog' else (0, 0, 255)                    #if it sees hot dog, green, if not, red
                    label = f"{prediction} ({confidence:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow(f"Webcam Hotdog Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

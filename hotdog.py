from model_functions import *
from keras.optimizers import RMSprop
from mode_functions import image_mode, video_mode, webcam_mode

def main():

    #TEMP UNTIL OZ MAKES UI

    print("===== Hotdog Detector Menu =====")
    print("1. Image Mode")
    print("2. Video Mode")
    print("3. Webcam Mode")
    choice = input("Select mode (1 or 2 or 3): ").strip()


    # Load existing model or train if not found
    if os.path.exists("hotdog.h5"):
        model = LoadModel("hotdog.h5")
        print("Loaded existing model hotdog.h5")
    else:
        print("No saved model found. Training a new model...")
        model = train_model()  # function that contains your training code
        model.save("hotdog.h5")
        print("Model trained and saved as hotdog.h5")


    if choice == "1":
        image_mode()
    elif choice == "2":
        video_mode()
    elif choice =="3":
        webcam_mode()
    else:
        print("Invalid choice. Exiting.")
        sys.exit(0)


    '''
    Main function for Hotdog or Not Hotdog
    Program will train a model to classify images as hotdog or not hotdog
    If a model is provided as a command line argument, it will load the model and skip training
    It will then load a random image from the test directory and predict if it is a hotdog or not hotdog
    displaying the image and prediction label
    '''

if __name__ == "__main__":
    main()

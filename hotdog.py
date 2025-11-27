#Imports
import customtkinter as ctk
from model_functions import *
from mode_functions import image_mode, upload_image_mode, video_mode, webcam_mode, previous_image, next_image

#Global mode
mode = None

#Clears the app between selections
def clear_content():
    for widget in content_frame.winfo_children():
        widget.destroy()

#Home screen of the app
def home():
    #Title
    title = ctk.CTkLabel(
        content_frame,
        text = "Is it a Hot Dog?",
        font = ctk.CTkFont(size = 28, weight = "bold")
    )
    title.pack(padx = 50, pady = 10)
    #Subtitle
    subtitle = ctk.CTkLabel(
        content_frame,
        text = "Try out the model!",
        font = ctk.CTkFont(size = 25),
    )
    subtitle.pack(padx = 10, pady = 10)
    #Test Button
    test = ctk.CTkButton(
        content_frame, 
        text = "Test Images",
        width = 160,      # wider
        height = 80,      # taller
        command = test_model, 
        font=ctk.CTkFont(size=20),
        hover_color="#ffdb58",
        fg_color = "#931009")
    test.pack(padx =  10, pady = 10)
    #Subtitle
    subtitle2 = ctk.CTkLabel(
        content_frame,
        text = "Choose a mode:",
        font = ctk.CTkFont(size = 25)
    )
    subtitle2.pack(padx = 10, pady = 10)
    #Upload Image Button
    image = ctk.CTkButton (
        content_frame,
        text = "Upload Image",
        width = 160,
        height = 80,
        command = upload_image,
        font=ctk.CTkFont(size=20),
        hover_color = "#ffdb58",
        fg_color = "#931009"
    )
    image.pack(padx = 10, pady = 10)
    #Upload Video Button
    video = ctk.CTkButton (
        content_frame,
        text = "Upload Video",
        width = 160,
        height = 80,
        command = upload_video,
        font=ctk.CTkFont(size=20),
        hover_color = "#ffdb58",
        fg_color = "#931009"
    )
    video.pack(padx = 10, pady = 10)
    #Webcam Button
    webcam = ctk.CTkButton (
        content_frame,
        text = "Use Webcam",
        width = 160,
        height = 80,
        command = camera,
        font=ctk.CTkFont(size=20),
        hover_color = "#ffdb58",
        fg_color = "#931009"
    )
    webcam.pack(padx = 10, pady = 10)

#Test Model function
def test_model():
    global mode
    clear_content()
    mode = "test"
    image_mode(content_frame)
#Upload Image function
def upload_image():
    global mode
    clear_content()
    mode = "image"
    upload_image_mode(content_frame)
#Camera function
def camera():
    global mode
    clear_content()
    mode = "webcam"
    webcam_mode(content_frame)
#Upload Video function
def upload_video():
    global mode
    clear_content()
    mode = "video"
    video_mode(content_frame)
#Function to listen to keys
def on_key(event):
    global mode
    print("KEY EVENT:", repr(event.char), event.keysym, "mode:", mode)  # DEBUG
    ch = event.char.lower()
    #If q is pressed clears widgets
    if ch == 'q':
        mode = None
        clear_content()
        home()
        return
    #If a and d is pressed, cycles images
    if mode == "test":
        if ch == 'a':
            previous_image()
        elif ch == 'd':
            next_image()
#Main function
def main():
    #Global variables
    global app, content_frame

    # Load existing model or train if not found
    if os.path.exists("hotdog.h5"):
        model = LoadModel("hotdog.h5")
        print("Loaded existing model hotdog.h5")
    else:
        print("No saved model found. Training a new model...")
        model = train_model()  # function that contains your training code
        model.save("hotdog.h5")
        print("Model trained and saved as hotdog.h5")

    #Define app
    app = ctk.CTk()
    app.bind_all("<Key>", on_key)
    app.title("Hotdog Detector")
    app.geometry("800x650")

    #Define content frame
    content_frame = ctk.CTkFrame(app, fg_color="#C48D54")
    content_frame.pack(fill="both", expand=True, padx=20, pady=20)

    #Call home function
    home()

    #Opens app
    app.mainloop()

    '''
    Main function for Hotdog or Not Hotdog
    Program will train a model to classify images as hotdog or not hotdog
    If a model is provided as a command line argument, it will load the model and skip training
    It will then load a random image from the test directory and predict if it is a hotdog or not hotdog
    displaying the image and prediction label
    '''
#Calls main
if __name__ == "__main__":
    main()

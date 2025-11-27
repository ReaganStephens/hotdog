#Imports
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.models import load_model as LoadModel, Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.utils import plot_model
import platform
import customtkinter as ctk
from PIL import Image, ImageTk


#Global Variables
image_width = 299
image_height = 299
batch_size = 16
num_epochs = 20

DISPLAY_WIDTH = 400
DISPLAY_HEIGHT = 400

def train_model():
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary',
        classes=['nothotdog', 'hotdog']
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        'dataset/test',
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary',
        classes=['nothotdog', 'hotdog']
    )

    # Build model
    model = getModel()

    # Visualize model
    plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
    print("Model architecture visualization saved as 'model_architecture.png'")

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = ModelCheckpoint('hotdog_checkpoint.h5', save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Load weights if checkpoint exists
    if os.path.exists('hotdog_checkpoint.h5'):
        print("Loading weights from 'hotdog_checkpoint.h5'")
        model.load_weights('hotdog_checkpoint.h5')
        print("Weights loaded successfully")

    # Train
    print("Training model...")
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )

    # Evaluate
    print("Evaluating model...")
    evaluation_results = model.evaluate(validation_generator, steps=len(validation_generator))
    print(f"Loss: {evaluation_results[0]}")
    print(f"Accuracy: {evaluation_results[1]}")

    return model

def getModel():
    '''
    returns a model with a InceptionV3 base model and custom classification layers
    '''
    # Load a pre-trained InceptionV3 model without the top classification layer
    base_model = InceptionV3(
        weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))

    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers on top of the pre-trained model
    # apply global average pooling to reduce the spatial dimensions of the feature maps
    x = GlobalAveragePooling2D()(base_model.output)
    # apply a fully-connected layer with 1024 hidden units and leaky ReLU activation
    x = Dense(1024, activation='leaky_relu', kernel_regularizer=l2(0.01))(x)
    # apply batch normalization to standardize the activations of the previous layer
    x = BatchNormalization()(x)
    # apply dropout regularization to prevent overfitting to the training data
    x = Dropout(0.5)(x)
    # apply a final linear transformation and sigmoid activation function to produce the final output of the model
    prediction = Dense(1, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=prediction)

    # Use the RMSprop optimizer with an initial learning rate
    initial_learning_rate = 0.001  # default learning rate
    if platform.machine() in ['arm64', 'arm64e']:
        optimizer = tf.keras.optimizers.legacy.RMSprop(
            learning_rate=initial_learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=initial_learning_rate)

    # Compile the model with binary cross-entropy loss and accuracy metric
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def lr_schedule(epoch, initial_lr=0.001, min_lr=1e-6, max_lr=1e-3):
    '''
    Learning Rate Scheduler
    takes the epoch, initial learning rate, minimum learning rate, and maximum learning rate as arguments
    returns the learning rate for the epoch
    '''
    lr = initial_lr
    if epoch > 10:
        lr *= 0.1
    elif epoch > 5:
        lr *= 0.5
    lr = max(lr, min_lr)
    lr = min(lr, max_lr)
    return lr


def showImagePrediction(parent, image_path, prediction, confidence, correct_prediction):
    """
    Displays the image with prediction label inside a CustomTkinter widget.
    'parent' should be a CTkFrame or CTk window where the image will appear.
    """
    #Clear Widgets
    for widget in parent.winfo_children():
        widget.destroy()

    #Load original image
    img = cv2.imread(image_path)

    #Resize image 
    img = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    #Label and background color
    if prediction == 'hotdog':
        label = f"{prediction} ({confidence:.2f})"
        bg_color = (0, 255, 0)  # green
    else:
        label = f"{"not hotdog"} ({1 - confidence:.2f})"
        bg_color = (0, 0, 255)  # red

    font_color = (255, 255, 255)

    #Label height
    label_bar_height = 60

    # Create final canvas with room below the image
    total_height = DISPLAY_HEIGHT + label_bar_height
    canvas = np.zeros((total_height, DISPLAY_WIDTH, 3), dtype=np.uint8)

    # Put image on top
    canvas[0:DISPLAY_HEIGHT, :] = img

    #Draw background bar
    cv2.rectangle(canvas,
                  (0, DISPLAY_HEIGHT),
                  (DISPLAY_WIDTH, total_height),
                  bg_color, -1)

    #Draw label centered
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = (DISPLAY_WIDTH - text_size[0]) // 2
    text_y = DISPLAY_HEIGHT + (label_bar_height + text_size[1]) // 2

    cv2.putText(canvas, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)

    #Draw checkmark or X over a circle
    center_x = DISPLAY_WIDTH // 2
    center_y = DISPLAY_HEIGHT

    cv2.circle(canvas, (center_x, center_y), 25, bg_color, -1)

    if correct_prediction:
        cv2.line(canvas, (center_x - 10, center_y),
                 (center_x - 2, center_y + 12), font_color, 3)
        cv2.line(canvas, (center_x - 2, center_y + 12),
                 (center_x + 12, center_y - 12), font_color, 3)
    else:
        cv2.line(canvas, (center_x - 10, center_y - 10),
                 (center_x + 10, center_y + 10), font_color, 3)
        cv2.line(canvas, (center_x - 10, center_y + 10),
                 (center_x + 10, center_y - 10), font_color, 3)

    #Convert image
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(canvas)
    tk_img = ImageTk.PhotoImage(pil_img)

    #Display image label
    img_label = ctk.CTkLabel(parent, image=tk_img, text="")
    img_label.image = tk_img  # keep reference
    img_label.pack(pady=10)

    #Display instructions
    instructions = ctk.CTkLabel(
        parent,
        text="Press A for Previous Image  |   Press D for Next\nPress Q to Quit to Home Screen",
        font=ctk.CTkFont(size=14)
    )
    instructions.pack(pady=(0, 10))
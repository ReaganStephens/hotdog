
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, load_img, img_to_array
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.applications.inception_v3 import InceptionV3
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.models import load_model as LoadModel, Model
import platform
import cv2
import numpy as np
import tensorflow as tf

import os
import sys
import random
import cv2
import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.models import load_model as LoadModel, Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.utils import plot_model, load_img, img_to_array
import platform



# Global Variables
image_width = 299
image_height = 299
batch_size = 16
num_epochs = 20

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


def showImagePrediction(image_path, prediction, confidence, correct=True):
    '''
    Displays the image with the prediction label
    takes the image path, prediction label, confidence, and whether the prediction is correct as arguments
    '''
    # Load the image
    img = cv2.imread(image_path)

    # Add the prediction label to the image
    if prediction == 'hotdog':
        label = f"{prediction} ({confidence:.2f})"
        bg_color = (0, 255, 0)  # green background for hotdog prediction
    else:
        label = f"{prediction} ({1 - confidence:.2f})"
        bg_color = (0, 0, 255)  # red background for not hotdog prediction

    # Set the font color to white
    font_color = (255, 255, 255)

    # Get the size of the image and the prediction label
    img_height, img_width, _ = img.shape
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    # Calculate the position of the label
    label_x = (img_width - label_size[0]) // 2
    label_y = img_height + label_size[1] + 20

    # Create a new image with the same width and a taller height to accommodate the label
    new_img = np.zeros((label_y, img_width, 3), np.uint8)
    new_img[:img_height, :] = img

    # Draw the background rectangle for the label
    cv2.rectangle(new_img, (0, img_height),
                  (img_width, label_y), bg_color, -1)
    
    # Draw the background rectangle for the checkmark or X
    cv2.circle(new_img, (img_width // 2, img_height), 30, bg_color, -1)

    # Draw the prediction label
    cv2.putText(new_img, label, (label_x, img_height + label_size[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)

    # Calculate the position of the checkmark or X
    symbol_x = img_width // 2 - 15
    symbol_y = img_height - 15
    
    # Draw the checkmark or X
    if correct:
        # Draw a checkmark using lines
        cv2.line(new_img, (symbol_x, symbol_y + 10), (symbol_x + 10, symbol_y + 20), font_color, 3)
        cv2.line(new_img, (symbol_x + 10, symbol_y + 20), (symbol_x + 25, symbol_y - 5), font_color, 3)
    else:
        # Draw an "X" using lines
        cv2.line(new_img, (symbol_x + 5, symbol_y), (symbol_x + 25, symbol_y + 20), font_color, 3)
        cv2.line(new_img, (symbol_x + 5, symbol_y + 20), (symbol_x + 25, symbol_y), font_color, 3)


    # Show the image
    cv2.imshow("Hotdog or Not Hotdog", new_img)
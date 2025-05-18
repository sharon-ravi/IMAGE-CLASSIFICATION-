import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D, MaxPooling2D,
                                     Input, GlobalAveragePooling2D, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2 # A good pre-trained model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random

# --- Configuration ---
NUM_CLASSES = 10
IMG_ROWS, IMG_COLS, IMG_CHANNELS = 32, 32, 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
BATCH_SIZE = 64
EPOCHS_SCRATCH = 50 # Epochs for CNN from scratch
EPOCHS_TRANSFER = 20 # Epochs for initial transfer learning head training
EPOCHS_FINETUNE = 10 # Epochs for fine-tuning (if done)

# CIFAR-10 Class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- 1. Load and Preprocess CIFAR-10 Data ---
def load_and_preprocess_data():
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(f"x_train shape: {x_train.shape}") # (50000, 32, 32, 3)
    print(f"y_train shape: {y_train.shape}") # (50000, 1)
    print(f"x_test shape: {x_test.shape}")   # (10000, 32, 32, 3)
    print(f"y_test shape: {y_test.shape}")   # (10000, 1)

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train_categorical = to_categorical(y_train, NUM_CLASSES)
    y_test_categorical = to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train_categorical, y_train), (x_test, y_test_categorical, y_test)

# --- 2. Build a CNN from Scratch ---
def build_cnn_from_scratch(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 3. Use a Pre-trained Model (MobileNetV2) with Transfer Learning ---
def build_transfer_learning_model(input_shape, num_classes, fine_tune_at_layer=None):
    # MobileNetV2 expects input images of at least 32x32, but performs better with larger ones.
    # For CIFAR-10, we'll use its native 32x32.
    # If you wanted to upsample: TARGET_IMG_SIZE = 96; input_shape=(TARGET_IMG_SIZE, TARGET_IMG_SIZE, 3)
    # and resize images before passing to preprocess_input.

    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=input_shape) # Use original CIFAR-10 input shape

    # Freeze the base model (or part of it if fine_tune_at_layer is set)
    if fine_tune_at_layer is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        # Freeze all layers before the `fine_tune_at_layer`
        for layer in base_model.layers[:fine_tune_at_layer]:
            layer.trainable = False


    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x) # Increased dense layer size
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # For fine-tuning, use a lower learning rate
    lr = 1e-5 if fine_tune_at_layer is not None else 1e-3
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Utility to plot training history ---
def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Utility to display predictions ---
def display_predictions(model, x_test_raw, y_test_raw, class_names, num_display=10):
    # Preprocess x_test_raw for MobileNetV2 if it's the transfer learning model
    # For the CNN from scratch, it's already normalized (0-1)
    x_test_for_pred = x_test_raw.astype('float32') / 255.0 # Default for scratch CNN
    if "mobilenetv2" in model.name: # Check if it's the MobileNetV2 based model
        x_test_for_pred = tf.keras.applications.mobilenet_v2.preprocess_input(x_test_raw.copy())


    predictions = model.predict(x_test_for_pred)
    predicted_classes = np.argmax(predictions, axis=1)

    plt.figure(figsize=(15, 7))
    for i in range(num_display):
        idx = random.randint(0, len(x_test_raw) - 1)
        plt.subplot(2, num_display // 2, i + 1)
        plt.imshow(x_test_raw[idx].astype('uint8')) # Display original unnormalized image
        
        true_label = class_names[y_test_raw[idx][0]]
        pred_label = class_names[predicted_classes[idx]]
        confidence = np.max(predictions[idx]) * 100

        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)", color=color, fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    (x_train, y_train_cat, y_train_orig), (x_test, y_test_cat, y_test_orig) = load_and_preprocess_data()
    
    # Keep a copy of raw test images for display
    (_, _), (x_test_raw, _) = cifar10.load_data()


    # --- Option 1: Train CNN from Scratch ---
    print("\n--- Training CNN from Scratch ---")
    cnn_scratch_model = build_cnn_from_scratch(INPUT_SHAPE, NUM_CLASSES)
    cnn_scratch_model.summary()

    # Data Augmentation for CNN from scratch (can also be used for transfer learning)
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    history_scratch = cnn_scratch_model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE),
        epochs=EPOCHS_SCRATCH,
        validation_data=(x_test, y_test_cat),
        verbose=1
    )
    plot_history(history_scratch, "CNN from Scratch")
    loss_scratch, acc_scratch = cnn_scratch_model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"CNN from Scratch - Test Accuracy: {acc_scratch*100:.2f}%")
    print("Displaying predictions from CNN from scratch...")
    display_predictions(cnn_scratch_model, x_test_raw, y_test_orig, CLASS_NAMES)


    # --- Option 2: Transfer Learning with MobileNetV2 ---
    print("\n--- Training with Transfer Learning (MobileNetV2) ---")
    
    # Preprocess input specifically for MobileNetV2
    # (scales pixel values from -1 to 1, among other things)
    x_train_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(x_train * 255.0) # Needs 0-255 input
    x_test_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input(x_test * 255.0)   # Needs 0-255 input

    # Build and train the head
    transfer_model = build_transfer_learning_model(INPUT_SHAPE, NUM_CLASSES)
    transfer_model.name = "mobilenetv2_transfer" # For display_predictions check
    transfer_model.summary()

    print("Training the new classification head...")
    history_transfer = transfer_model.fit(
        x_train_mobilenet, y_train_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_TRANSFER,
        validation_data=(x_test_mobilenet, y_test_cat),
        verbose=1
    )
    plot_history(history_transfer, "Transfer Learning - Head Training")
    loss_transfer, acc_transfer = transfer_model.evaluate(x_test_mobilenet, y_test_cat, verbose=0)
    print(f"Transfer Learning (Head Trained) - Test Accuracy: {acc_transfer*100:.2f}%")

    # --- Option 2b: Fine-tuning (Optional) ---
    # Unfreeze some layers of the base model and train with a very low learning rate
    # Find a good layer to start fine-tuning from.
    # For MobileNetV2, len(base_model.layers) is around 154. Let's unfreeze from layer 100.
    # fine_tune_from_layer_index = 100
    # transfer_model_finetune = build_transfer_learning_model(INPUT_SHAPE, NUM_CLASSES,
    #                                                      fine_tune_at_layer=fine_tune_from_layer_index)
    # transfer_model_finetune.name = "mobilenetv2_finetune"
    # print(f"\nFine-tuning from layer {fine_tune_from_layer_index}...")
    # # We need to load weights from the previously head-trained model
    # # transfer_model_finetune.set_weights(transfer_model.get_weights()) # This might not work as expected due to recompile
    # # Better to train it further or rebuild and load weights selectively if needed.
    # # For simplicity, let's continue training the 'transfer_model' after unfreezing some layers.

    base_model = transfer_model.layers[1] # The MobileNetV2 base model is the second layer in our 'transfer_model'
    base_model.trainable = True
    fine_tune_at = 100 # Unfreeze from this layer onwards
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    transfer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Very low LR for fine-tuning
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    transfer_model.summary() # See trainable params change
    
    print("\nFine-tuning the model...")
    history_finetune = transfer_model.fit(
        x_train_mobilenet, y_train_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_TRANSFER + EPOCHS_FINETUNE, # Continue training
        initial_epoch=history_transfer.epoch[-1] + 1, # Start from where previous training left off
        validation_data=(x_test_mobilenet, y_test_cat),
        verbose=1
    )
    plot_history(history_finetune, "Transfer Learning - Fine-tuning")
    loss_finetune, acc_finetune = transfer_model.evaluate(x_test_mobilenet, y_test_cat, verbose=0)
    print(f"Transfer Learning (Fine-tuned) - Test Accuracy: {acc_finetune*100:.2f}%")
    print("Displaying predictions from Transfer Learning model...")
    display_predictions(transfer_model, x_test_raw, y_test_orig, CLASS_NAMES)
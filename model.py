import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Directories
train_dir = 'dataset/dataset/train'
val_dir = 'dataset/dataset/test'

# Clean out non-image files
def clean_directory(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                os.remove(os.path.join(root, file))

clean_directory(train_dir)
clean_directory(val_dir)

# Data Generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator_raw = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

val_generator_raw = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Fix generator mismatch using tf.data.Dataset
def cast_generator(gen):
    for x, y in gen:
        yield tf.cast(x, tf.float32), tf.cast(y, tf.float32)

train_generator = tf.data.Dataset.from_generator(
    lambda: cast_generator(train_generator_raw),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
    )
)

val_generator = tf.data.Dataset.from_generator(
    lambda: cast_generator(val_generator_raw),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
    )
)

# Model Definition
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile Model
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=loss,
    metrics=['accuracy']
)

# Class Weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator_raw.classes),
    y=train_generator_raw.classes
)
class_weights = dict(enumerate(class_weights))

history = model.fit(
    train_generator,
    epochs=60,
    validation_data=val_generator,
    class_weight=class_weights,
    steps_per_epoch=len(train_generator_raw),
    validation_steps=len(val_generator_raw)
)

# ---------------------------
# Evaluate Model
# ---------------------------
loss, accuracy = model.evaluate(val_generator, steps=len(val_generator_raw))
print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

# ---------------------------
# Predict & Report
# ---------------------------
val_generator_raw.reset()
pred_probs = model.predict(val_generator_raw, steps=len(val_generator_raw), verbose=1)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_generator_raw.classes
class_names = list(val_generator_raw.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save Model in Keras native format
model.save('bruise_detection_model.keras')

# Load the model back
model = tf.keras.models.load_model('bruise_detection_model.keras')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)















# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Directories
# train_dir = 'dataset/dataset/train'
# val_dir = 'dataset/dataset/test'

# # Data generators
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# val_datagen = ImageDataGenerator(rescale=1.0/255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=True
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=False  # Important for evaluation consistency
# )

# # Load base model
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Freeze all layers initially
# for layer in base_model.layers:
#     layer.trainable = False

# # Unfreeze last 20 layers
# for layer in base_model.layers[-20:]:
#     layer.trainable = True

# # Add custom layers
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# predictions = Dense(4, activation='softmax')(x)

# # Final model
# model = Model(inputs=base_model.input, outputs=predictions)

# # Compile model
# model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Train model
# history = model.fit(
#     train_generator,
#     epochs=55,
#     validation_data=val_generator
# )

# # Evaluate model
# loss, accuracy = model.evaluate(val_generator)
# print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

# # Predictions
# val_generator.reset()
# pred_probs = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1, verbose=1)
# y_pred = np.argmax(pred_probs, axis=1)
# y_true = val_generator.classes

# # Class names
# class_names = list(val_generator.class_indices.keys())

# # Classification report
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# # Confusion matrix
# cm = confusion_matrix(y_true, y_pred)

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()

# model.save('bruise_detection_model.h5')

# model = tf.keras.models.load_model('bruise_detection_model.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)





# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Directories
# train_dir = 'dataset/dataset/train'
# val_dir = 'dataset/dataset/test'

# # Data generators
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# val_datagen = ImageDataGenerator(rescale=1.0/255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=True
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=False
# )

# # Load EfficientNetB0 base model
# base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Freeze layers
# for layer in base_model.layers:
#     layer.trainable = False
# for layer in base_model.layers[-20:]:
#     layer.trainable = True

# # Add custom layers
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# predictions = Dense(4, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# # Compile
# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train
# history = model.fit(train_generator, epochs=60, validation_data=val_generator)

# # Evaluate
# loss, accuracy = model.evaluate(val_generator)
# print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

# # Predict and evaluate
# val_generator.reset()
# pred_probs = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1, verbose=1)
# y_pred = np.argmax(pred_probs, axis=1)
# y_true = val_generator.classes
# class_names = list(val_generator.class_indices.keys())

# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix - EfficientNetB0")
# plt.tight_layout()
# plt.show()









# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Directories
# train_dir = 'dataset/dataset/train'
# val_dir = 'dataset/dataset/test'

# # Data generators
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# val_datagen = ImageDataGenerator(rescale=1.0/255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=True
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=False
# )

# # Load ResNet50 base model
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Freeze layers
# for layer in base_model.layers:
#     layer.trainable = False
# for layer in base_model.layers[-20:]:
#     layer.trainable = True

# # Add custom layers
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# predictions = Dense(4, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# # Compile
# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train
# history = model.fit(train_generator, epochs=60, validation_data=val_generator)

# # Evaluate
# loss, accuracy = model.evaluate(val_generator)
# print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

# # Predict and evaluate
# val_generator.reset()
# pred_probs = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1, verbose=1)
# y_pred = np.argmax(pred_probs, axis=1)
# y_true = val_generator.classes
# class_names = list(val_generator.class_indices.keys())

# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix - ResNet50")
# plt.tight_layout()
# plt.show()



















# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# img_height = img_width = 32
# batch_size = 20

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     "dataset/dataset/train",
#     image_size = (img_height, img_width),
#     batch_size = batch_size
# )
  
# test_ds = tf.keras.utils.image_dataset_from_directory(
#     "dataset/dataset/test",
#     image_size = (img_height, img_width),
#     batch_size = batch_size
# )

# num_classes = 4

# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1./255),
#     tf.keras.layers.Conv2D(32, 3, activation="relu"),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation="relu"),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation="relu"),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(num_classes)
# ])

# model.compile(
#     optimizer="adam",
#     loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"]
# )

# model.fit(
#     train_ds,
#     epochs=100
# )

# model.evaluate(test_ds)

# y_true = []
# y_pred = []

# for images, labels in test_ds:
#     logits = model.predict(images)
#     predictions = tf.argmax(logits, axis=1)
    
#     y_true.extend(labels.numpy())
#     y_pred.extend(predictions.numpy())

# # Classification report
# print("Classification Report:\n")
# print(classification_report(y_true, y_pred))

# # Confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# class_names = test_ds.class_names  # Auto-extracted from directory names

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()








#####################################################
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# train_dir = 'dataset/dataset/train'
# val_dir = 'dataset/dataset/test'

# train_datagen = ImageDataGenerator(
#     rescale=1.0/255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# val_datagen = ImageDataGenerator(rescale=1.0/255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=16,
#     class_mode='categorical'
# )

# val_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(224, 224),
#     batch_size=16,
#     class_mode='categorical'
# )


# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# for layer in base_model.layers:
#     layer.trainable = False

# for layer in base_model.layers[-20:0]:
#     layer.trainable = True

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# predictions = Dense(4, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# history = model.fit(
#     train_generator,
#     epochs=1,
#     validation_data=val_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     validation_steps=val_generator.samples // val_generator.batch_size
# )

# loss, accuracy = model.evaluate(val_generator)

# y_true = []
# y_pred = []

# for images, labels in val_generator:
#     print(type(images))
#     print(type(labels))
#     logits = model.predict(images)
#     predictions = tf.argmax(logits, axis=1)
    
#     y_true.extend(labels.tolist())
#     y_pred.extend(predictions.numpy().tolist())

#     if len(y_true) >= val_generator.samples:
#         break

# # Classification report
# print("Classification Report:\n")
# print(classification_report(y_true, y_pred))

# # Confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# class_names = val_generator.class_names  # Auto-extracted from directory names

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()

# print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# # model.save('bruise_detection_model.h5')
#####################################################









'''
######################################3

import numpy as np
from sklearn.metrics import classification_report

# Get ground truth labels
val_generator.reset()  # Ensure generator is at the beginning
y_true = val_generator.classes

# Get predicted probabilities
y_pred_probs = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)

# Convert predicted probabilities to class indices
y_pred = np.argmax(y_pred_probs, axis=1)

# Get class labels
class_labels = list(val_generator.class_indices.keys())

# Generate classification report
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

###############################################

import matplotlib.pyplot as plt

# Plot accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Directories for training and validation data
train_dir = 'dataset/train'
val_dir = 'dataset/validation'

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Rescaling for the validation set
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Creating the training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Creating the validation data generator
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load MobileNetV2 as the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the initial layers and unfreeze the last 20 layers for fine-tuning
for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(9, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save('bruise_detection_model.h5')

# --------------------------------------------------------
# PERFORMANCE METRICS: Precision, Recall, F1-Score, and Confusion Matrix
# --------------------------------------------------------

# Reset the validation generator to ensure it's at the beginning
val_generator.reset()

# Get ground truth labels
y_true = val_generator.classes

# Get predicted probabilities
y_pred_probs = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)

# Convert predicted probabilities to class indices
y_pred = np.argmax(y_pred_probs, axis=1)

# Get class labels
class_labels = list(val_generator.class_indices.keys())

# Generate and print the classification report
report = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:")
print(report)

# Generate and plot the confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# --------------------------------------------------------
# PLOT ACCURACY AND LOSS VS. EPOCH
# --------------------------------------------------------

# Plot training and validation accuracy
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()
'''
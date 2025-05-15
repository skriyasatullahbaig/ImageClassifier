import tensorflow as tf
from tensorflow.keras import layers, models, Input
from preprocess import train_data, val_data

# Build CNN Model
model = models.Sequential([
    Input(shape=(150, 150, 3)),  # Correct way to specify input shape
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the model
model.save("ImageClassifier/cat_dog_breed_classifier.h5")
print("Model training complete and saved!")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Подготовка на податоци
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    'dataset/', target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')

val_data = datagen.flow_from_directory(
    'dataset/', target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')

# CNN архитектура
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(16, activation='softmax')  # Број на класи (пр. 16 различни гестови)
])

# Компилација
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Тренинг
model.fit(train_data, validation_data=val_data, epochs=10)

# Зачувување на моделот
model.save('gesture_model.h5')

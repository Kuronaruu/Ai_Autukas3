import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Путь к папкам
train_dir = '/Users/karen/Desktop/archive (2)/dataset/train'
val_dir = '/Users/karen/Desktop/archive (2)/dataset/valid'

# Подготовка генераторов данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Генераторы
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Создание модели CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# Компиляция модели
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение модели
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Сохранение модели
model.save('plant_classifier_model.keras')

# Построение графиков точности
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Построение графиков потерь
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Оценка модели на валидационной выборке
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# ---- Отображение предсказания на одной валидационной картинке ----

# Загружаем одну картинку из validation_generator
validation_generator.reset()
img_batch, label_batch = next(validation_generator)
img = img_batch[0]  # первая картинка
true_label = np.argmax(label_batch[0])  # правильная метка

# Предсказание модели
pred = model.predict(np.expand_dims(img, axis=0))
predicted_label = np.argmax(pred)
predicted_confidence = pred[0][predicted_label] * 100  # перевод в %

# Отображение картинки с предсказанием и процентной вероятностью
plt.imshow(img)
plt.title(f"Predicted: {list(validation_generator.class_indices.keys())[predicted_label]} "
          f"({predicted_confidence:.2f}%)\n"
          f"True: {list(validation_generator.class_indices.keys())[true_label]}")
plt.axis('off')
plt.show()

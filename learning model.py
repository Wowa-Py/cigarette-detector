import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics
import tensorflow.keras.preprocessing.image as image
from sklearn.model_selection import train_test_split
import numpy as np

# Подготовка датасета
train_dir = '/content/cigarette detector model/Data_image/smoking'
test_dir = '/content/cigarette detector model/Data_image/not_smoking'

train_data = image.ImageDataGenerator(rescale=1./255).flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
test_data = image.ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary')

# Определение метки класса для обучающих и тестовых данных
train_labels = train_data.classes
test_labels = test_data.classes

# Определение архитектуры модели
def build_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (10, 10), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (7, 7), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (4, 4), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (4, 4), activation='relu'),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Загрузка датасета
train_dataset = train_data.next()[0]
test_dataset = test_data.next()[0]

# Определение функции потерь и оптимизатора
model = build_model()
model.compile(loss=losses.BinaryCrossentropy(), optimizer=optimizers.SGD(lr=0.001, momentum=0.9), metrics=[metrics.BinaryAccuracy()])

# Обучение модели
model.fit(train_dataset, train_labels, epochs=10, batch_size=32)

# Оценка производительности модели
test_loss, test_acc = model.evaluate(test_dataset, test_labels)
print('Test accuracy:', test_acc)

# Сохранение весов модели
model.save_weights('/content/cigarette detector model/weights.h5')
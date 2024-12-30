import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.0001
DATASET_PATH = r"Oral Cancer Final Dataset\Oral Images Dataset\augmented_data"


def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE) / 255.0
    return image, label

def load_dataset(dataset_path, batch_size, subset):
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        subset=subset,
        seed=123
    )
    dataset = dataset.map(preprocess_image).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = load_dataset(DATASET_PATH, BATCH_SIZE, subset="training")
val_dataset = load_dataset(DATASET_PATH, BATCH_SIZE, subset="validation")

all_labels = []
for _, labels in train_dataset.unbatch():  
    all_labels.append(labels.numpy())
all_labels = np.concatenate(all_labels, axis=0) 

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),  
    y=all_labels  
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)



base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_mobilenet_model.keras', save_best_only=True, monitor='val_loss', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
]


model.summary()


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


model.save('final_mobilenet_model.keras')

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()

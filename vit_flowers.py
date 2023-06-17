import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from vit_keras import vit


warnings.filterwarnings('ignore')
print('TensorFlow Version ' + tf.__version__)


DATASET_PATH = './flowers'
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

generator = ImageDataGenerator(
    rescale=1./255,
    samplewise_center=True,
    samplewise_std_normalization=True,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

training_set = generator.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    seed=1,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical',
    subset='training')
validation_set = generator.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    seed=1,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical',
    subset='validation')


vit_model = vit.vit_b16(
    image_size=IMAGE_SIZE,
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False,
    classes=5)
model = Sequential([
    vit_model,
    BatchNormalization(),
    Dense(512, activation=tfa.activations.gelu),
    Dropout(0.2),
    BatchNormalization(),
    Dense(128, activation=tfa.activations.gelu),
    Dropout(0.2),
    Dense(5, 'softmax')
], name='vit_flowers_model')

early_stopping = EarlyStopping(patience=30, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('./cpt-epoch-{epoch:03d}-val_acc-{val_accuracy:03f}.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
model.compile(optimizer=tfa.optimizers.RectifiedAdam(learning_rate=LEARNING_RATE),
              loss=CategoricalCrossentropy(label_smoothing=0.2),
              metrics=['accuracy'])
model.fit(x=training_set,
          steps_per_epoch=training_set.n // training_set.batch_size,
          validation_data=validation_set,
          validation_steps=validation_set.n // validation_set.batch_size,
          epochs=EPOCHS,
          callbacks=[early_stopping, model_checkpoint])
model.save('vit_flowers.h5')
model.save_weights('vit_flowers_weights.h5')

with open('vit_flowers.json', 'w') as json_file:
    json_file.write(model.to_json())

history = model.history.history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history['loss'], label='Training Loss')
ax1.plot(history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Model Loss')
ax1.legend()
ax2.plot(history['accuracy'], label='Training Accuracy')
ax2.plot(history['val_accuracy'], label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Model Accuracy')
ax2.legend()
plt.tight_layout()
plt.show()

predicted_classes = np.argmax(model.predict(validation_set, steps=validation_set.n // validation_set.batch_size + 1), axis=1)
true_classes = validation_set.classes
class_labels = list(validation_set.class_indices.keys())
confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(16, 16))
sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)
print(classification_report(true_classes, predicted_classes))

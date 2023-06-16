import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras.metrics import TopKCategoricalAccuracy
from cutmix_keras import CutMixImageDataGenerator

import pickle as pkl

train_path = '/content/train/'
test_path = '/content/test/'
checkpoint_path = '/content/drive/MyDrive/CILSFinal/checkpoint.h5'
batch_size = 32
image_size = (299, 299)
epochs = 15

train_datagen = ImageDataGenerator(rescale=1./255,
                  shear_range=0.2,
                  zoom_range=0.2,
                  horizontal_flip=True,
                  validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator_1 = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

train_generator_2 = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

train_generator = CutMixImageDataGenerator(
  generator1=train_generator_1,
  generator2=train_generator_2,
  img_size=299,
  batch_size=batch_size)

validation_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=train_generator_1.classes)
class_weights = dict(enumerate(class_weights))

base_model = Xception(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
out = Dense(50, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)

# for layer in base_model.layers:
#     layer.trainable = False

model.summary()

model.compile(optimizer=Adam(),
       loss='categorical_crossentropy',
       metrics=['accuracy',TopKCategoricalAccuracy(k=5)])

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3)

history = model.fit(
        train_generator,
        steps_per_epoch=train_generator_1.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop, reduce_lr])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
plt.imsave('Accuracy_curve.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
plt.imsave('Loss_curve.png')

loss,top_1_accuracy,top_5_accuracy = model.evaluate(test_generator)
print(f'Top 1 Test accuracy: {top_1_accuracy}')
print(f'Top 5 Test accuracy: {top_5_accuracy}')

test_generator.reset()
pred = model.predict(test_generator, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
cm = confusion_matrix(test_generator.classes, predicted_class_indices)

sns.heatmap(cm, annot=False)
plt.imsave('confusion_matrix.png')

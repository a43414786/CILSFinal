# Import necessary libraries
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
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
from keras.models import load_model


train_path = './train'
test_path = './test'
checkpoint_path = './checkpoint.h5'
batch_size = 32
image_size = (299, 299)
epochs = 50

# Data generators with augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2 
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)

# Load the data from directory
train_generator = train_datagen.flow_from_directory(
        train_path,  
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

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
        class_mode='categorical')

# Calculate class weights for imbalance dataset
class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=train_generator.classes)
class_weights = dict(enumerate(class_weights))

# Load the Xception model
base_model = Xception(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer and a logistic layer with 50 classes 
x = Dense(1024, activation='relu')(x)
predictions = Dense(50, activation='softmax')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
history = History()

callbacks = [checkpoint, early_stop, reduce_lr, history]

# Train the model
history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size,
        class_weight=class_weights,
        callbacks=callbacks)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}')

# Predict the labels
test_generator.reset()
pred = model.predict(test_generator, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

# Get the confusion matrix
cm = confusion_matrix(test_generator.classes, predicted_class_indices)

# Plot the confusion matrix
sns.heatmap(cm, annot=True)
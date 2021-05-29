import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Conv2D, Dropout
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV

import os, cv2
f = open("../ai-unibuc-23-31-2021/train.txt")
lines = f.readlines()

train_images = []
train_labels = []

for line in lines:
    line = line.strip().split(",")
    image_name = line[0]
    image = cv2.imread(os.path.join("../ai-unibuc-23-31-2021/train/", image_name))
    image_label = int(line[1])
    if image_label==0:
        cv2.imwrite(os.path.join("../impartire/train/0", image_name), image)
    elif image_label==1:
        cv2.imwrite(os.path.join("../impartire/train/1", image_name), image)
    elif image_label==2:
        cv2.imwrite(os.path.join("../impartire/train/2", image_name), image)
    train_images.append(image)
    train_labels.append(image_label)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
    zoom_range=0.1,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
)

validation_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    directory= "../impartire/train",
    target_size=(50,50),
    batch_size= 32
)
validation_generator = validation_datagen.flow_from_directory(
    directory= "../impartire/validation",
    target_size=(50,50),
    shuffle=False,
)

model = Sequential()
model.add(keras.layers.experimental.preprocessing.RandomContrast(factor = 0.4, input_shape=(50, 50, 3,)))
model.add(Conv2D(32,5, activation="relu", input_shape=(50, 50, 3,)))
model.add(Conv2D(32,5,activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64,3, activation="relu"))
model.add(Conv2D(64,3,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(128,3,activation="relu"))
model.add(Conv2D(128,3,activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(256,3,activation="relu"))
model.add(Conv2D(256,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(48, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])

checkpoint_filepath = "checkpoint1"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    monitor = 'val_acc',
    mode = 'max',
    save_best_only = True
)

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=model_checkpoint_callback
)

model.load_weights("checkpoint1")

f = open("../ai-unibuc-23-31-2021/test.txt")
g = open("submission.txt", "w")
g.write("id,label\n")
lines = f.readlines()

for line in lines:
    line = line.strip()
    image_name = line
    image = cv2.imread(os.path.join("../ai-unibuc-23-31-2021/test/", image_name))
    image = np.array([image])
    pred_label = model.predict(image)
    pred_label = np.argmax(pred_label)

    g.write(image_name + "," + str(pred_label) + "\n")
    
g.close()
print("done writing")

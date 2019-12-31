
# Convolutional Neural Network- Classification of Cat vs. Dog Using Keras and Tensorflow backend
### Project from Machine Learning A-Z<sup>TM</sup> by Kirill Eremenko and Hadelin De Ponteves



## Let's first begin by importing the necessary libraries and packages for CNN


```python
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

```

    Using TensorFlow backend.


# Steps for initialization of CNN
### 1) Convolution operation/ ReLU layer-rectifier (reduce non-linearity)
### 2) Pooling of data/down sampling to reduce the size, parameters, preserved features, account for textual/spacial invaraince, minimize overfitting
### 3) Adding additional convolutional layers
### 4) Flattening of the data - becomes the input layer to NN
### 5) Full connection of the nuero networks
### 6) Completion of CNN


```python
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Adding a second convolutional layer
classifier.add(Conv2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 4 - Flattening
classifier.add(Flatten())

# Step 5 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Step 6 - Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```


# Fitting the CNN to the scaled images


```python
# Part 2 - Fitting the CNN to the images and avoid overfitting by using image augmentation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         nb_epoch = 25,
                         validation_data = test_set,
                         validation_steps = 62)
```

    Found 8000 images belonging to 2 classes.
    Found 2000 images belonging to 2 classes.


    Epoch 1/25
    250/250 [==============================] - 85s 338ms/step - loss: 0.6664 - acc: 0.5945 - val_loss: 0.6080 - val_acc: 0.6663
    Epoch 2/25
    250/250 [==============================] - 110s 439ms/step - loss: 0.5927 - acc: 0.6866 - val_loss: 0.5545 - val_acc: 0.7317
    Epoch 3/25
    250/250 [==============================] - 106s 426ms/step - loss: 0.5371 - acc: 0.7289 - val_loss: 0.5512 - val_acc: 0.7273
    Epoch 4/25
    250/250 [==============================] - 108s 431ms/step - loss: 0.5152 - acc: 0.7419 - val_loss: 0.4776 - val_acc: 0.7734
    Epoch 5/25
    250/250 [==============================] - 109s 436ms/step - loss: 0.4882 - acc: 0.7658 - val_loss: 0.4854 - val_acc: 0.7694
    Epoch 6/25
    250/250 [==============================] - 104s 417ms/step - loss: 0.4641 - acc: 0.7799 - val_loss: 0.4708 - val_acc: 0.7784
    Epoch 7/25
    250/250 [==============================] - 111s 444ms/step - loss: 0.4556 - acc: 0.7817 - val_loss: 0.4698 - val_acc: 0.7759
    Epoch 8/25
    250/250 [==============================] - 107s 426ms/step - loss: 0.4374 - acc: 0.7877 - val_loss: 0.4644 - val_acc: 0.7816
    Epoch 9/25
    250/250 [==============================] - 98s 391ms/step - loss: 0.4166 - acc: 0.8040 - val_loss: 0.4794 - val_acc: 0.7835
    Epoch 10/25
    250/250 [==============================] - 108s 434ms/step - loss: 0.3985 - acc: 0.8190 - val_loss: 0.4618 - val_acc: 0.7856
    Epoch 11/25
    250/250 [==============================] - 106s 424ms/step - loss: 0.3837 - acc: 0.8224 - val_loss: 0.4469 - val_acc: 0.8074
    Epoch 12/25
    250/250 [==============================] - 107s 429ms/step - loss: 0.3682 - acc: 0.8345 - val_loss: 0.4502 - val_acc: 0.8138
    Epoch 13/25
    250/250 [==============================] - 99s 397ms/step - loss: 0.3574 - acc: 0.8399 - val_loss: 0.4715 - val_acc: 0.7879
    Epoch 14/25
    250/250 [==============================] - 102s 408ms/step - loss: 0.3466 - acc: 0.8449 - val_loss: 0.4684 - val_acc: 0.7986
    Epoch 15/25
    250/250 [==============================] - 113s 451ms/step - loss: 0.3264 - acc: 0.8541 - val_loss: 0.4903 - val_acc: 0.8050
    Epoch 16/25
    250/250 [==============================] - 114s 456ms/step - loss: 0.3095 - acc: 0.8668 - val_loss: 0.4578 - val_acc: 0.8226
    Epoch 17/25
    250/250 [==============================] - 111s 442ms/step - loss: 0.2915 - acc: 0.8781 - val_loss: 0.4360 - val_acc: 0.8138
    Epoch 18/25
    250/250 [==============================] - 110s 440ms/step - loss: 0.2794 - acc: 0.8818 - val_loss: 0.4898 - val_acc: 0.8112
    Epoch 19/25
    250/250 [==============================] - 112s 448ms/step - loss: 0.2524 - acc: 0.8909 - val_loss: 0.5407 - val_acc: 0.7936
    Epoch 20/25
    250/250 [==============================] - 94s 376ms/step - loss: 0.2426 - acc: 0.9001 - val_loss: 0.5862 - val_acc: 0.8013
    Epoch 21/25
    250/250 [==============================] - 90s 359ms/step - loss: 0.2347 - acc: 0.9044 - val_loss: 0.4726 - val_acc: 0.8037
    Epoch 22/25
    250/250 [==============================] - 83s 334ms/step - loss: 0.2220 - acc: 0.9074 - val_loss: 0.5321 - val_acc: 0.7967
    Epoch 23/25
    250/250 [==============================] - 84s 336ms/step - loss: 0.2109 - acc: 0.9125 - val_loss: 0.5160 - val_acc: 0.8100
    Epoch 24/25
    250/250 [==============================] - 86s 344ms/step - loss: 0.2057 - acc: 0.9164 - val_loss: 0.6585 - val_acc: 0.7917
    Epoch 25/25
    250/250 [==============================] - 94s 375ms/step - loss: 0.1758 - acc: 0.9326 - val_loss: 0.6066 - val_acc: 0.8037

    <keras.callbacks.History at 0x183ad06b748>


# Conclusion 
#### Running the model took ~43 minutes with a classification accuracy of 93% and validation accuracy of 80%. The overfitting problem observed may be mitigated by increasing the number of images, adding more layers, adding batch normalization, or adding dropout layers.

#### Further improvements for the model can be accomplished by adding more convolutional layers, increase hidden layers, and increasing the resolution of the image file from (64,64) to higher resolution (i.e. 256,256) to extract more features. However, under those parameters, we should process the data using a GPU instead of a CPU that was used to allow for practical computational speeds/time.

#### Thank you for viewing my project - Tak Koyanagi

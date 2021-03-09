from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop

####################################################
# Simple Shapes Model
####################################################
def compile_model(model, optimizer, learning_rate=0.001):
    # Compiling the CNN
    if optimizer=='SGD':
        opt = SGD(lr=learning_rate, momentum=0.9, decay=1e-2 / NUM_EPOCHS)
    else:
        opt = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer = opt,
                loss = 'categorical_crossentropy', 
                metrics = ['accuracy'])

####################################################
# Simple Shapes Model
####################################################
def simple_shapes_model():
    # Initialising the CNN
    model = Sequential()
    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), padding='same', input_shape = (28, 28, 3), activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) # antes era 0.25
    # Adding a second convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) # antes era 0.25
    # Adding a third convolutional layer
    model.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5)) # antes era 0.25
    # Step 3 - Flattening
    model.add(Flatten())
    # Step 4 - Full connection
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(units = 3, activation = 'softmax'))
    return model
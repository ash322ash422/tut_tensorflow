import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


#Lets start by loading the Cifar10 data
(X, y), (X_test, y_test) = cifar10.load_data()


#Keep in mind the images are in RGB
#So we can normalise the data by diving by 255
#The data is in integers therefore we need to convert them to float first
X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0

#Then we convert the y values into one-hot vectors
#The cifar10 has only 10 classes, thats is why we specify a one-hot
#vector of width/class 10
y, y_test = to_categorical(y, 10), to_categorical(y_test, 10)

#Now we can go ahead and create our Convolution model
model = Sequential()
#We want to output 32 features maps. The kernel size is going to be
#3x3 and we specify our input shape to be 32x32 with 3 channels
#Padding=same means we want the same dimensional output as input
#activation specifies the activation function
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
                 activation='relu'))
#20% of the nodes are set to 0
model.add(Dropout(0.2))
#now we add another convolution layer, again with a 3x3 kernel
#This time our padding=valid this means that the output dimension can
#take any form
model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
#maxpool with a kernet of 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
#In a convolution NN, we neet to flatten our data before we can
#input it into the ouput/dense layer
model.add(Flatten())
#Dense layer with 512 hidden units
model.add(Dense(512, activation='relu'))
#this time we set 30% of the nodes to 0 to minimize overfitting
model.add(Dropout(0.3))
#Finally the output dense layer with 10 hidden units corresponding to
#our 10 classe
model.add(Dense(10, activation='softmax'))
#Few simple configurations
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])
#Run the algorithm!
model.fit(X, y, validation_data=(X_test, y_test), epochs=1,
          batch_size=512)

# model.save("cifar10_model.h5")  # This saves everything
model.save('model.keras')

#Now load the model
model = load_model("model.keras")

#Finally print the accuracy of our model!
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Make predictions
predictions = np.argmax(model.predict(X_test), axis=1)

# Print first 10 predictions
print("Predicted classes      :", predictions[:10])#Prints out a number
print("Actual classes(one-hot):",y_test[:10])
print("Actual classes         :", np.argmax(y_test[:10], axis=1))

#1 - airplane, 2 - automobile, 3 - bird, 4 - cat, 5 - deer, 6 - dog
#7 - frog, 8 - horse, 9 - ship, 10 - truck

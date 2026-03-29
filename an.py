import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


#Normalize pixel values (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

#Flatten 28×28 images into 784 features:
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

#Build ANN Model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes (0-9)
])


#Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Train Model
model.fit(x_train, y_train, epochs=10, batch_size=32)

#Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)


#Make Predictions
predictions = model.predict(x_test)
print("Predicted digit:", predictions[0].argmax())

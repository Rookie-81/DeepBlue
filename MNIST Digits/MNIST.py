import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist #28x28  hand written images 0---9
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())  #INPUT layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #HiddenLayer #1
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #HiddenLayer #2
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #OUTPUT Layer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train,epochs=3)
val_loss,val_acc = model.evaluate(x_test, y_test)
print(val_acc, val_loss)
model.save('Digit Reco.model')

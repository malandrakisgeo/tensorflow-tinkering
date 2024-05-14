import tensorflow as tf
import keras as ker;
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = ker.models.Sequential([
    ker.layers.Flatten(input_shape=(28, 28)),
    ker.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    ker.layers.BatchNormalization(),
    ker.layers.Dropout(0.2),
    ker.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    ker.layers.BatchNormalization(),
    ker.layers.Dropout(0.4),

    ker.layers.Dense(10)
])


predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)


model.evaluate(x_test,  y_test, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])



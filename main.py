import tensorflow as tf


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


train_images = tf.image.resize(train_images[..., tf.newaxis], (96, 96)) / 255.0
test_images = tf.image.resize(test_images[..., tf.newaxis], (96, 96)) / 255.0


batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

def conv_block(num_channels):
    blk = tf.keras.Sequential()
    blk.add(tf.keras.layers.BatchNormalization())
    blk.add(tf.keras.layers.Activation('relu'))
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same'))
    return blk

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = tf.keras.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def call(self, x):
        for blk in self.net.layers:
            y = blk(x)
            x = tf.concat([x, y], axis=-1)
        return x

def transition_block(num_channels):
    blk = tf.keras.Sequential()
    blk.add(tf.keras.layers.BatchNormalization())
    blk.add(tf.keras.layers.Activation('relu'))
    blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=1))
    blk.add(tf.keras.layers.AvgPool2D(pool_size=2, strides=2))
    return blk


net = tf.keras.Sequential()
net.add(tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=(96, 96, 1)))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('relu'))
net.add(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    num_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))


net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('relu'))
net.add(tf.keras.layers.GlobalAveragePooling2D())
net.add(tf.keras.layers.Dense(10))

lr, num_epochs = 0.1, 10
net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


net.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)


test_loss, test_accuracy = net.evaluate(test_dataset)
print(f"Loss: {test_loss:.3f}, Train Accuracy: {net.history.history['accuracy'][-1]:.3f}, Test Accuracy: {test_accuracy:.3f}")

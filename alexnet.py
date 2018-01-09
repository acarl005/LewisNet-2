import tensorflow as tf

class AlexNet:
    """A convolutional AlexNet network"""

    def __init__(self, X, dropout_rate, num_classes, trainable=True):
        self.X = X
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.trainable = trainable

        self._create()

    def _create(self):
        """Create the computation graph in tensorflow"""
        # 1st layer: Conv -> ReLU -> LRN -> MaxPool
        conv1 = tf.layers.conv2d(self.X, filters=96, kernel_size=(7, 7), strides=4, activation=tf.nn.relu, trainable=self.trainable, name="conv1") # 36x36
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name="norm1")
        pool1 = tf.layers.max_pooling2d(norm1, pool_size=(3, 3), strides=2, name="pool1") # 17x17

        # 2nd layer: Conv -> ReLU -> LRN -> MaxPool
        conv2 = tf.layers.conv2d(pool1, filters=256, kernel_size=(5, 5), padding="SAME", activation=tf.nn.relu, trainable=self.trainable, name="conv2") # 17x17
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name="norm2")
        pool2 = tf.layers.max_pooling2d(norm2, pool_size=(3, 3), strides=2, name="pool2") # 8x8

        # 3rd layer: Conv -> ReLU
        conv3 = tf.layers.conv2d(pool2, filters=384, kernel_size=(3, 3), padding="SAME", activation=tf.nn.relu, trainable=self.trainable, name="conv3")

        # 4th layer: Conv -> ReLU
        conv4 = tf.layers.conv2d(conv3, filters=384, kernel_size=(3, 3), padding="SAME", activation=tf.nn.relu, trainable=self.trainable, name="conv4")

        # 5th layer: Conv -> ReLU => MaxPool
        conv5 = tf.layers.conv2d(conv4, filters=256, kernel_size=(3, 3), padding="SAME", activation=tf.nn.relu, trainable=self.trainable, name="conv5") # 8x8
        pool5 = tf.layers.max_pooling2d(conv5, pool_size=(3, 3), strides=2, name="pool5") # 3x3

        fc6 = tf.layers.conv2d(pool5, filters=4096, kernel_size=(3, 3), activation=tf.nn.relu, trainable=self.trainable, name="fc6")
        dropout6 = tf.layers.dropout(fc6, rate=self.dropout_rate)

        fc7 = tf.layers.conv2d(dropout6, filters=4096, kernel_size=(1, 1), activation=tf.nn.relu, trainable=self.trainable, name="fc7")
        dropout7 = tf.layers.dropout(fc7, rate=self.dropout_rate)

        self.logits = tf.layers.conv2d(dropout7, filters=self.num_classes, kernel_size=(1, 1), trainable=self.trainable, name="fc8")


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer"""
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


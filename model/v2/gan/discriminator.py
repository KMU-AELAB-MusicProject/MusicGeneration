import tensorflow as tf

from model.v2.config import *


def discriminator(model_name='Default'):
    if model_name == 'Default':
        return Default()
    if model_name == 'NoCondition':
        return NoCondition()


def linear(x, output_size, bias_start=0.0, with_w=False, name='fc'):
    shape = x.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable(name="matrix", shape=[shape[1], output_size],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name="bias", shape=[output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


class Default(object):
    def __init__(self):
        pass

    def __call__(self, data, pre_phrase_feature, reuse=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            data = tf.reshape(data, [-1, data.shape[1], data.shape[2], 1])
            h = data

            with tf.variable_scope('chroma-feature', reuse=reuse):
                n_beats = h.get_shape()[1] // BEAT_RESOLUTION
                reshaped = tf.reshape(data, (-1, n_beats, BEAT_RESOLUTION, h.get_shape()[2], h.get_shape()[3]))

                summed = tf.reduce_sum(reshaped, 2)

                factor = int(h.get_shape()[2]) // 12
                remainder = int(h.get_shape()[2]) % 12
                reshaped = tf.reshape(summed[..., :(factor * 12), :], (-1, n_beats, factor, 12, 1))
                chroma = tf.reduce_sum(reshaped, 2)  # 4, 4, 12

                if remainder:
                    chroma += tf.reshape(tf.reduce_sum(summed[..., -remainder:, :], 2), (-1, n_beats, 1, 1))

                chroma = tf.layers.conv2d(chroma, 32, [3, 3], padding='same', strides=2, activation=tf.nn.relu)
                chroma = tf.layers.conv2d(chroma, 64, [3, 3], padding='same', strides=2, activation=tf.nn.leaky_relu)
                chroma = tf.reshape(tf.layers.average_pooling2d(chroma,
                                                                (chroma.get_shape()[1].value, chroma.get_shape()[2].value),
                                                                1),
                                    [-1, chroma.get_shape()[3].value])

            with tf.variable_scope('on/offset-feature', reuse=reuse):
                padded = tf.pad(data[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0)))
                on_off_set = tf.reduce_sum(data - padded, 2, True)  # 4, 48, 1
                on_off_set = tf.layers.conv2d(on_off_set, 16, [3, 1], padding='same', strides=2, activation=tf.nn.relu)
                on_off_set = tf.layers.conv2d(on_off_set, 32, [3, 1], padding='same', strides=2, activation=tf.nn.relu)
                on_off_set = tf.layers.conv2d(on_off_set, 64, [3, 1], padding='same', strides=2, activation=tf.nn.leaky_relu)
                on_off_set = tf.reshape(tf.layers.average_pooling2d(on_off_set,
                                                                    (on_off_set.get_shape()[1].value,
                                                                     on_off_set.get_shape()[2].value),
                                                                    1),
                                        [-1, on_off_set.get_shape()[3].value])

            with tf.variable_scope('phrase_feature', reuse=reuse):
                x = tf.reshape(data, [-1, data.get_shape()[1], data.get_shape()[2], 1])
                with tf.variable_scope('pitch-time'):
                    x1 = tf.layers.conv2d(x, 16, [1, 12], padding='same', strides=2, activation=tf.nn.relu)
                    x1 = tf.layers.conv2d(x1, 32, [4, 1], padding='same', strides=2, activation=tf.nn.relu)

                with tf.variable_scope('time-pitch', reuse=reuse):
                    x2 = tf.layers.conv2d(x, 16, [4, 1], padding='same', strides=2, activation=tf.nn.relu)
                    x2 = tf.layers.conv2d(x2, 32, [1, 12], padding='same', strides=2, activation=tf.nn.relu)

                x = tf.concat([x1, x2], axis=3)

                with tf.variable_scope('merged_feature', reuse=reuse):
                    x = tf.layers.conv2d(x, 64, [1, 1], padding='same')
                    x = tf.layers.conv2d(x, 128, [3, 3], padding='same', strides=2, activation=tf.nn.leaky_relu)
                    x = tf.layers.conv2d(x, 256, [5, 5], padding='same', strides=2, activation=tf.nn.leaky_relu)
                    x = tf.reshape(tf.layers.average_pooling2d(x, (x.get_shape()[1].value, x.get_shape()[2].value), 1),
                                   [-1, x.get_shape()[3].value])
                    x = tf.layers.dense(x, 64)

                x = tf.concat([chroma, on_off_set, x, pre_phrase_feature], axis=-1)

                x = tf.layers.dense(x, 32, activation=tf.nn.relu)
                x = tf.layers.dense(x, 1, activation=tf.nn.relu)
                x_flatten = tf.layers.flatten(x)
                x_linear = linear(x_flatten, 1, name='x_linear')

            return tf.nn.sigmoid(x_linear), x_linear


class NoCondition(object):
    def __init__(self, bcc_model, rbc_model):
        self.bcc_model = bcc_model
        self.rbc_model = rbc_model

    def __call__(self, data, beat_cnt, root_beat, step):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            data = tf.reshape(data, [-1, data.shape[1], data.shape[2], 1])
            step = tf.reshape(step, [-1, 1])
            h = data

            with tf.variable_scope('chroma-feature'):
                n_beats = h.get_shape()[1] // BEAT_RESOLUTION
                reshaped = tf.reshape(data, (-1, n_beats, BEAT_RESOLUTION, h.get_shape()[2], h.get_shape()[3]))

                summed = tf.reduce_sum(reshaped, 2)

                factor = int(h.get_shape()[2]) // 12
                remainder = int(h.get_shape()[2]) % 12
                reshaped = tf.reshape(summed[..., :(factor * 12), :], (-1, n_beats, factor, 12, 1))
                chroma = tf.reduce_sum(reshaped, 2)  # 4, 4, 12

                if remainder:
                    chroma += tf.reshape(tf.reduce_sum(summed[..., -remainder:, :], 2), (-1, n_beats, 1, 1))

                chroma = tf.layers.conv2d(chroma, 32, [3, 3], padding='same', strides=2, activation=tf.nn.relu)
                chroma = tf.layers.conv2d(chroma, 64, [3, 3], padding='same', strides=2, activation=tf.nn.leaky_relu)
                chroma = tf.reshape(tf.layers.average_pooling2d(chroma,
                                                                (chroma.get_shape()[1].value, chroma.get_shape()[2].value),
                                                                1),
                                    [-1, chroma.get_shape()[3].value])

            with tf.variable_scope('on/offset-feature'):
                padded = tf.pad(data[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0)))
                on_off_set = tf.reduce_sum(data - padded, 2, True)  # 4, 48, 1
                on_off_set = tf.layers.conv2d(on_off_set, 16, [3, 1], padding='same', strides=2, activation=tf.nn.relu)
                on_off_set = tf.layers.conv2d(on_off_set, 32, [3, 1], padding='same', strides=2, activation=tf.nn.relu)
                on_off_set = tf.layers.conv2d(on_off_set, 64, [3, 1], padding='same', strides=2, activation=tf.nn.leaky_relu)
                on_off_set = tf.reshape(tf.layers.average_pooling2d(on_off_set,
                                                                    (on_off_set.get_shape()[1].value,
                                                                     on_off_set.get_shape()[2].value),
                                                                    1),
                                        [-1, on_off_set.get_shape()[3].value])

            with tf.variable_scope('phrase_feature', reuse=tf.AUTO_REUSE):
                x = tf.reshape(data, [-1, data.get_shape()[1], data.get_shape()[2], 1])
                with tf.variable_scope('pitch-time'):
                    x1 = tf.layers.conv2d(x, 16, [1, 12], padding='same', strides=2, activation=tf.nn.relu)
                    x1 = tf.layers.conv2d(x1, 32, [4, 1], padding='same', strides=2, activation=tf.nn.relu)

                with tf.variable_scope('time-pitch'):
                    x2 = tf.layers.conv2d(x, 16, [4, 1], padding='same', strides=2, activation=tf.nn.relu)
                    x2 = tf.layers.conv2d(x2, 32, [1, 12], padding='same', strides=2, activation=tf.nn.relu)

                x = tf.concat([x1, x2], axis=3)

                with tf.variable_scope('merged_feature'):
                    x = tf.layers.conv2d(x, 64, [1, 1], padding='same')
                    x = tf.layers.conv2d(x, 128, [3, 3], padding='same', strides=2, activation=tf.nn.leaky_relu)
                    x = tf.layers.conv2d(x, 256, [5, 5], padding='same', strides=2, activation=tf.nn.leaky_relu)
                    x = tf.reshape(tf.layers.average_pooling2d(x, (x.get_shape()[1].value, x.get_shape()[2].value), 1),
                                   [-1, x.get_shape()[3].value])
                    x = tf.layers.dense(x, 64)

                x = tf.concat([chroma, on_off_set, x], axis=1)
                x = tf.layers.dense(x, 32, activation=tf.nn.leaky_relu)
                x = tf.layers.dense(x, 1)
                x_flatten = tf.layers.flatten(x)
                x_linear = linear(x_flatten, 1, name='x_linear')

            return tf.nn.sigmoid(x_linear), x_linear

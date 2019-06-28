import tensorflow as tf

from model.v2.config import *


def generator(model_name='Default'):
    if model_name == 'Default':
        return Default()
    if model_name == 'NoFeature':
        return NoFeature()
    if model_name =='AddArranger':
        return AddArranger()


def feature_extractor(phrase, training=True, reuse=True):
    with tf.variable_scope('generator_feature_extractor', reuse=reuse):
        x = tf.reshape(phrase, [-1, phrase.get_shape()[1], phrase.get_shape()[2], 1])
        with tf.variable_scope('pitch-time'):
            x1 = tf.layers.conv2d(x, 16, [1, 12], padding='same', strides=2, activation=tf.nn.relu)
            x1 = tf.layers.conv2d(x1, 32, [4, 1], padding='same', strides=2, activation=tf.nn.relu)

        with tf.variable_scope('time-pitch'):
            x2 = tf.layers.conv2d(x, 16, [4, 1], padding='same', strides=2, activation=tf.nn.relu)
            x2 = tf.layers.conv2d(x2, 32, [1, 12], padding='same', strides=2, activation=tf.nn.relu)

        x = tf.concat([x1, x2], axis=3)

        with tf.variable_scope('merged_feature'):
            x = tf.layers.conv2d(x, 64, [1, 1], padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, [3, 3], padding='same', strides=2, activation=tf.nn.relu)
            x = tf.layers.batch_normalization(
                tf.layers.conv2d(x, 96, [3, 3], padding='same', strides=2, activation=tf.nn.relu),
                training=training)
            x = tf.reshape(tf.layers.average_pooling2d(x, (x.get_shape()[1].value, x.get_shape()[2].value), 1),
                           [-1, x.get_shape()[3].value])

        return tf.layers.flatten(x)


def arranger(phrase):
    with tf.variable_scope('generator_arranger'):
        lstm_fw_cell = [tf.contrib.rnn.LSTMCell(num_units=PITCH_SIZE) for _ in range(2)]
        lstm_bw_cell = [tf.contrib.rnn.LSTMCell(num_units=PITCH_SIZE) for _ in range(2)]

        output, fw, bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, phrase, dtype=tf.float32)
        output = tf.reshape(output, [-1, BAR_SIZE * PHRASE_SIZE * 2, PITCH_SIZE * 2, 1])
        output = tf.layers.conv2d(output, 1, [3, 3], strides=[1, 2], padding='same', activation=tf.nn.leaky_relu)
        output = tf.gather(output, [i for i in range(BAR_SIZE * PHRASE_SIZE, BAR_SIZE * PHRASE_SIZE * 2)], axis=1)

        return tf.reshape(output, [-1, BAR_SIZE * PHRASE_SIZE, PITCH_SIZE])


class Default(object):
    def __init__(self):
        pass

    def __call__(self, z, pre_phrase, training=True, reuse=True):
        pre_phrase_feature = feature_extractor(pre_phrase, training=training, reuse=reuse)
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('phrase_generator', reuse=reuse):
                _x = tf.concat([z, pre_phrase_feature], axis=-1)
                _x = tf.reshape(_x, [-1, 1, 1, _x.get_shape()[-1]])

                with tf.variable_scope('pitch-time'):
                    x1 = tf.layers.conv2d_transpose(_x, 128, [1, 6], strides=[1, 6], activation=tf.nn.relu)  # (1, 6, 128)
                    x1 = tf.layers.conv2d_transpose(x1, 96, [6, 1], strides=[6, 1], activation=tf.nn.relu)  # (6, 6, 96)

                with tf.variable_scope('time-pitch'):
                    x2 = tf.layers.conv2d_transpose(_x, 128, [6, 1], strides=[6, 1], activation=tf.nn.relu)  # (6, 1, 128)
                    x2 = tf.layers.conv2d_transpose(x2, 96, [1, 6], strides=[1, 6], activation=tf.nn.relu)  # (6, 6, 96)

                x = tf.concat([x1, x2], axis=3)  # (6, 6, 192)

                with tf.variable_scope('merged_feature'):
                    x = tf.layers.conv2d(x, 128, [1, 1], padding='same')  # (6, 6, 128)
                    x = tf.layers.conv2d_transpose(x, 64, [3, 3], padding='same', strides=2,
                                                   activation=tf.nn.relu)  # (12, 12, 64)
                    x = tf.layers.conv2d_transpose(x, 32, [3, 3], padding='same', strides=2,
                                                   activation=tf.nn.relu)  # (24, 24, 32)
                    x = tf.layers.conv2d_transpose(x, 16, [3, 3], padding='same', strides=2,
                                                   activation=tf.nn.relu)  # (48, 48, 16)
                    x = tf.layers.conv2d_transpose(x, PHRASE_SIZE, [3, 3], padding='same', strides=2,
                                                   activation=tf.nn.relu)  # (96, 96, 4)
                phrase = tf.reshape(tf.transpose(x, perm=[0, 3, 1, 2]), [-1, BAR_SIZE * PHRASE_SIZE, PITCH_SIZE])

            return phrase, pre_phrase_feature


class NoFeature(object):
    def __init__(self):
        pass

    def __call__(self, z, pre_phrase, training=True, reuse=True):
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('phrase_generator', reuse=reuse):
                _x = tf.reshape(z, [-1, 1, 1, Z_SIZE])

                with tf.variable_scope('pitch-time'):
                    x1 = tf.layers.conv2d_transpose(_x, 32, [1, 6], strides=[1, 6], activation=tf.nn.relu)  # (1, 6, 32)
                    x1 = tf.layers.conv2d_transpose(x1, 32, [6, 1], strides=[6, 1], activation=tf.nn.relu)  # (6, 6, 32)

                with tf.variable_scope('time-pitch'):
                    x2 = tf.layers.conv2d_transpose(_x, 32, [6, 1], strides=[6, 1], activation=tf.nn.relu)  # (6, 1, 32)
                    x2 = tf.layers.conv2d_transpose(x2, 32, [1, 6], strides=[1, 6], activation=tf.nn.relu)  # (6, 6, 32)

                x = tf.concat([x1, x2], axis=3)  # (6, 6, 64)

                with tf.variable_scope('merged_feature'):
                    x = tf.layers.conv2d(x, 64, [1, 1], padding='same')  # (6, 6, 96)
                    x = tf.layers.conv2d_transpose(x, 64, [3, 3], padding='same', strides=2,
                                                   activation=tf.nn.leaky_relu)  # (12, 12, 64)
                    x = tf.layers.conv2d_transpose(x, 32, [3, 3], padding='same', strides=2,
                                                   activation=tf.nn.leaky_relu)  # (24, 24, 32)
                    x = tf.layers.conv2d_transpose(x, 16, [3, 3], padding='same', strides=2,
                                                   activation=tf.nn.relu)  # (48, 48, 16)
                    x = tf.layers.conv2d_transpose(x, 4, [3, 3], padding='same', strides=2,
                                                   activation=tf.nn.relu)  # (96, 96, 4)
                phrase = tf.reshape(tf.transpose(x, perm=[0, 3, 1, 2]), [-1, BAR_SIZE * PHRASE_SIZE, PITCH_SIZE])

            return phrase, phrase


class AddArranger(object):
    def __init__(self):
        pass

    def __call__(self, z, pre_phrase, bar_generator, phrase_generator, phrase_number, training=True, reuse=True):
        pre_phrase_feature = phrase_generator.encoder(pre_phrase)
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('phrase_generator', reuse=reuse):
                bars = [bar_generator.decoder(tf.gather(z, i, axis=1) + pre_phrase_feature +
                                              tf.gather(bar_generator.bar_number, tf.tile([i], tf.shape(z)[0:1])))
                        for i in range(PHRASE_SIZE)]
                phrase1 = tf.reshape(tf.concat(bars, axis=1), [-1, BAR_SIZE * PHRASE_SIZE, PITCH_SIZE, 1])
                phrase2 = tf.reshape(phrase_generator.decoder(tf.reduce_mean(z, axis=1) + pre_phrase_feature +
                                                              tf.gather(phrase_generator.phrase_number, phrase_number)),
                                     [-1, BAR_SIZE * PHRASE_SIZE, PITCH_SIZE, 1])

                phrase = tf.concat([phrase1, phrase2], axis=-1)

                phrase1 = tf.layers.conv2d(phrase, 2, [1, 12], padding='same', activation=tf.nn.relu)
                phrase2 = tf.layers.conv2d(phrase, 2, [12, 1], padding='same', activation=tf.nn.relu)

                phrase += phrase1 + phrase2

                phrase = tf.reshape(tf.layers.conv2d(phrase, 1, [3, 3], padding='same', activation=tf.nn.leaky_relu),
                                    [-1, BAR_SIZE * PHRASE_SIZE, PITCH_SIZE])

                phrase = arranger(tf.concat([pre_phrase, phrase], axis=1))

            return phrase, pre_phrase_feature

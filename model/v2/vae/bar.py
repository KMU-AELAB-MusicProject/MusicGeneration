import os
import random
import pickle
import numpy as np
import tensorflow as tf

from ..config import *


class Bar_VAE(object):
    def __init__(self, session, phrase_model, save_path):
        self.session = session
        self.save_path = save_path
        self.phrase_model = phrase_model

        self._set_placeholder()
        self._build_model()

        self.minimum_loss = 9999999999

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

        tf.set_random_seed(410)

    def _set_placeholder(self):
        self.bar = tf.placeholder(dtype=tf.float32, shape=[None, BAR_SIZE, PITCH_SIZE])
        self.pre_phrase = tf.placeholder(dtype=tf.float32, shape=[None, BAR_SIZE * PHRASE_SIZE, PITCH_SIZE])
        self.number = tf.placeholder(dtype=tf.int32, shape=[None])
        self.bar_number = tf.get_variable(
            name='bar_number',
            shape=[PHRASE_SIZE, POSITION_VECTOR_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

    def _build_model(self):
        self.network()

    def encoder(self, bar):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            x = tf.reshape(bar, [-1, BAR_SIZE, PITCH_SIZE, 1])
            encoder1_1 = tf.layers.conv2d(x, 16, [1, 4], padding='same', strides=2, activation=tf.nn.relu)
            encoder1_1 = tf.layers.conv2d(encoder1_1, 32, [4, 1], padding='same', strides=2, activation=tf.nn.relu)

            encoder1_2 = tf.layers.conv2d(x, 16, [4, 1], padding='same', strides=2, activation=tf.nn.relu)
            encoder1_2 = tf.layers.conv2d(encoder1_2, 32, [1, 4], padding='same', strides=2, activation=tf.nn.relu)

            encoder2 = tf.concat([encoder1_1, encoder1_2], axis=-1)

            encoder3 = tf.layers.conv2d(encoder2, 16, [3, 3], padding='same', strides=2, activation=tf.nn.relu)
            encoder4 = tf.layers.conv2d(encoder3, 32, [3, 3], padding='same', strides=2, activation=tf.nn.relu)
            encoder5 = tf.layers.conv2d(encoder4, 64, [3, 3], padding='same', strides=2, activation=tf.nn.relu)
            encoder6 = tf.layers.flatten(tf.layers.conv2d(encoder5, 128, [3, 3], padding='same', strides=2,
                                                          activation=tf.nn.leaky_relu))

            self.mean = tf.layers.dense(encoder6, Z_SIZE)
            self.log_var = tf.layers.dense(encoder6, Z_SIZE)

            with tf.variable_scope('reparameterization'):
                eps = tf.random_normal(tf.shape(self.log_var), dtype=tf.float32, mean=0., stddev=1.0)

        return self.mean + tf.exp(self.log_var * 0.5) * eps

    def decoder(self, z):
        with tf.variable_scope('generator_bar_decoder', reuse=tf.AUTO_REUSE):
            _x = tf.reshape(z, [-1, 1, 1, Z_SIZE])

            with tf.variable_scope('pitch-time'):
                x1 = tf.layers.conv2d_transpose(_x, 128, [1, 6], strides=[1, 6], activation=tf.nn.relu)  # (1, 6, 128)
                x1 = tf.layers.conv2d_transpose(x1, 64, [6, 1], strides=[6, 1], activation=tf.nn.relu)  # (6, 6, 64)

            with tf.variable_scope('time-pitch'):
                x2 = tf.layers.conv2d_transpose(_x, 128, [6, 1], strides=[6, 1], activation=tf.nn.relu)  # (6, 1, 128)
                x2 = tf.layers.conv2d_transpose(x2, 64, [1, 6], strides=[1, 6], activation=tf.nn.relu)  # (6, 6, 64)

            x = tf.concat([x1, x2], axis=3)  # (6, 6, 128)

            with tf.variable_scope('merged_feature'):
                x += tf.layers.conv2d(x, 128, [1, 1], padding='same', activation=tf.nn.leaky_relu)  # (6, 6, 128)
                x = tf.layers.conv2d_transpose(x, 64, [3, 3], padding='same', strides=2,
                                               activation=tf.nn.leaky_relu)  # (12, 12, 64)
                x = tf.layers.conv2d_transpose(x, 32, [3, 3], padding='same', strides=2,
                                               activation=tf.nn.leaky_relu)  # (24, 24, 32)
                x = tf.layers.conv2d_transpose(x, 16, [3, 3], padding='same', strides=2,
                                               activation=tf.nn.relu)  # (48, 48, 16)
                x += tf.layers.conv2d(x, 16, [1, 1], padding='same', activation=tf.nn.leaky_relu)  # (48, 48, 16)
                x = tf.layers.conv2d(x, 8, [1, 1], padding='same', activation=tf.nn.relu)  # (48, 48, 8)

            return tf.layers.conv2d_transpose(x, 1, [3, 3], padding='same', strides=2)  # (96, 96, 1)

    def network(self):
        z = self.encoder(self.bar)
        pre_z = self.phrase_model.encoder(self.pre_phrase)
        order = tf.gather(self.bar_number, self.number)

        self.logits = tf.reshape(self.decoder(z + pre_z + order), [-1, BAR_SIZE, PITCH_SIZE])

        cond = tf.less(tf.nn.tanh(self.logits), tf.zeros(tf.shape(self.logits)))
        self.outputs = tf.where(cond, tf.zeros(tf.shape(self.logits)), tf.ones(tf.shape(self.logits)))
        tmp_image = tf.reshape(self.outputs, [-1, BAR_SIZE, PITCH_SIZE, 1])
        board_image = tf.summary.image('bar_image', tmp_image, max_outputs=3)

        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.bar, logits=self.logits)
        reconstruction_loss = tf.reduce_sum(xentropy)

        latent_loss = -0.5 * tf.reduce_sum(1. + self.log_var - tf.square(self.mean) - tf.exp(self.log_var), axis=1)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bar')

        self.loss = tf.reduce_mean(reconstruction_loss + latent_loss)
        board_loss = tf.summary.scalar('pre_train_bar_loss', self.loss)

        self.merged = tf.summary.merge([board_image, board_loss])

        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss, var_list=vars)

    def load_model(self, path):
        self.saver.restore(self.session, path)

    def train(self):
        writer = tf.summary.FileWriter(ROOT_PATH + 'board/bar', self.session.graph)

        print('train_start_bar')

        file_number = [i for i in range(22)]
        for epoch in range(1, 60000 + 1):
            loss_epoch = 0.
            for i in random.sample(file_number, 6):
                bar_number = 0

                with open(os.path.join(BAR_VAE_TRAIN_DATA_PATH, 'bar_data{}.pkl'.format(i)), 'rb') as fp:
                    bar, phrase = pickle.load(fp)

                for j in range(0, len(bar), 256):
                    tmp_num = [(bar_number + i) % PHRASE_SIZE for i in range(len(bar[j:j + 256]))]
                    _, loss, outputs = self.session.run([self.train_op, self.loss, self.outputs],
                                                        feed_dict={self.bar: bar[j:j + 256],
                                                                   self.pre_phrase: phrase[j:j + 256],
                                                                   self.number: tmp_num})
                    bar_number = (tmp_num[-1] + 1) % PHRASE_SIZE
                    loss_epoch += loss

            summary = self.session.run(self.merged, feed_dict={self.bar: bar[:100],
                                                               self.pre_phrase: phrase[:100],
                                                               self.number: [i % 4 for i in range(100)]})
            writer.add_summary(summary, epoch)

            if loss_epoch < self.minimum_loss:
                ckpt_path = self.saver.save(self.session, os.path.join(self.save_path, 'bar_model'))
                self.minimum_loss = loss_epoch
                print(ckpt_path)

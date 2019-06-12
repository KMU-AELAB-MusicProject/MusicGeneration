import os
import pickle
import numpy as np
import tensorflow as tf

from model.v1.config import *
from model.v1.utils import gan_batch_iterator
from .discriminator import discriminator
from .generator import generator


class GAN(object):
    def __init__(self, sess, save_path, note_path, bar_generator, phrase_generator):
        self.sess = sess
        self.lambd = 1.0

        self.bar_generator = bar_generator
        self.phrase_generator = phrase_generator

        self._build_placeholder()
        self._build_model()
        self.generate_music()

        self.save_path = save_path
        self.note_path = note_path

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=GAN_EPOCH)

    def _build_placeholder(self):
        self.z = tf.placeholder(tf.float32, shape=[None, PHRASE_SIZE, Z_SIZE])
        self.real = tf.placeholder(tf.float32, shape=[None, BAR_SIZE * PHRASE_SIZE, PITCH_SIZE])
        self.batch_size = tf.placeholder(tf.int32)
        self.pre_phrase = tf.placeholder(tf.float32, shape=[None, BAR_SIZE * PHRASE_SIZE, PITCH_SIZE])

    def _build_model(self):
        self.generator = generator(GENERATOR)
        self.discriminator = discriminator(DISCRIMINATOR)
        self.network()

    def generate_music(self):
        self.logit, _ = self.generator(self.z, self.pre_phrase, self.bar_generator, self.phrase_generator)
        cond = tf.less(self.logit, tf.zeros(tf.shape(self.logit)))
        self.music = tf.where(cond, tf.zeros(tf.shape(self.logit)), tf.ones(tf.shape(self.logit)))

    def network(self):
        self.logits, pre_phrase_feature = self.generator(self.z, self.pre_phrase, self.bar_generator, self.phrase_generator, True, False)
        cond = tf.less(self.logits, tf.zeros(tf.shape(self.logits)))
        self.outputs = tf.where(cond, tf.zeros(tf.shape(self.logits)), tf.ones(tf.shape(self.logits)))

        logits_r = self.discriminator(self.real, pre_phrase_feature, reuse=False)
        logits_f = self.discriminator(self.outputs, pre_phrase_feature, reuse=True)

        differences = self.outputs - self.real
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1],
                                  minval=0., maxval=1.)
        interpolates = self.real + (alpha * differences)
        gradients = tf.gradients(self.discriminator(interpolates, pre_phrase_feature, reuse=True), [interpolates])[0]
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        with tf.variable_scope('train_generator'):
            self.g_loss = tf.losses.absolute_difference(labels=self.real, predictions=self.outputs) -\
                          tf.reduce_mean(logits_f)

            with tf.variable_scope('generator_optimizer'):
                self.g_op = tf.train.AdamOptimizer(learning_rate=0.00002
                                                   ,
                                                   beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_vars)

        with tf.variable_scope('train_discriminator'):
            loss_r = tf.reduce_mean(logits_r)
            loss_f = tf.reduce_mean(logits_f)

            loss = loss_f - loss_r

            self.d_loss = loss + self.lambd * gradient_penalty

            with tf.variable_scope('discriminator_optimizer'):
                self.d_op = tf.train.AdamOptimizer(learning_rate=0.00005,
                                                   beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_vars)

    def load_model(self, path):
        self.saver.restore(self.sess, path)

    def make_music(self, phrase_size=5):
        pre_phrase = [[[0 for _ in range(PITCH_SIZE)] for _ in range(BAR_SIZE * PHRASE_SIZE)]]

        music, tmp = [], []
        for i in range(0, phrase_size):
            pre_phrase, logit = self.sess.run([self.music, self.logit],
                                       feed_dict={self.z: np.random.uniform(-1, 1, size=(1, Z_SIZE)),
                                                  self.pre_phrase: pre_phrase, self.phrase_generator.phrase: pre_phrase})
            music.append(pre_phrase)
            tmp.append(logit)

        tmp = np.concatenate(np.concatenate(np.array(tmp), axis=0), axis=0)
        print(tmp.min(), tmp.max(), tmp.mean())

        return music

    def train(self):
        for epoch in range(1, GAN_EPOCH + 1):
            d_loss = 0.
            g_loss = 0.
            for i in range(13):
                with open(os.path.join(GAN_TRAIN_DATA_PATH, 'gan_data{}.pkl'.format(i)), 'rb') as fp:
                    data = pickle.load(fp)

                for phrase, pre_phrase in gan_batch_iterator(data, GAN_BATCH_SIZE):
                    z = np.random.uniform(-1, 1, size=(len(phrase),PHRASE_SIZE, Z_SIZE))

                    _, _, loss1, loss2 = self.sess.run([self.d_op, self.g_op, self.d_loss, self.g_loss],
                                                       feed_dict={self.z: z, self.real: phrase,
                                                                  self.pre_phrase: pre_phrase,
                                                                  self.batch_size: len(phrase), self.phrase_generator.phrase: pre_phrase})
                    d_loss += loss1
                    g_loss += loss2

            print('epoch {} - g_loss: {}, d_loss: {}'.format(epoch, g_loss, d_loss))

            ckpt_path = self.saver.save(self.sess, os.path.join(self.save_path, 'gan_model'), epoch)
            print(ckpt_path)

            with open(os.path.join(self.note_path, 'gan{}(10).pkl'.format(epoch)), 'wb') as fp:
                pickle.dump(self.make_music(10), fp)
            with open(os.path.join(self.note_path, 'gan{}(15).pkl'.format(epoch)), 'wb') as fp:
                pickle.dump(self.make_music(15), fp)

if __name__ == '__main__':
    pass

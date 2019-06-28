import os
import tensorflow as tf

from .config import *
from .vae.bar import Bar_VAE
from .vae.phrase import Phrase_VAE
from model.v2.gan import gan


class Model(object):
    def __init__(self, sess, save_path, load_gan_model, model_number, pre_train, note_path):
        self.sess = sess
        self.save_path = save_path
        self.load_gan_model = load_gan_model
        self.model_number = model_number
        self.note_path = note_path
        self.pre_train = pre_train

        self.build_model()

    def build_model(self):
        self.phrase_generator = Phrase_VAE(self.sess, os.path.join(ROOT_PATH + 'trained', 'phrase'))
        self.bar_generator = Bar_VAE(self.sess, self.phrase_generator, os.path.join(ROOT_PATH + 'trained', 'bar'))
        self.gan = gan.GAN(self.sess, self.save_path, self.note_path, self.bar_generator, self.phrase_generator)

    def train(self):
        if self.load_gan_model is not None:
            print((os.path.join(ROOT_PATH + 'trained', 'gan', self.load_gan_model,
                                'gan_model-{}'.format(self.model_number))))
            self.gan.load_model(os.path.join(ROOT_PATH + 'trained', 'gan', self.load_gan_model,
                                             'gan_model-{}'.format(self.model_number)))

            if not self.pre_train:
                print((os.path.join(ROOT_PATH + 'trained', 'bar', 'bar_model')))
                self.gan.load_model(os.path.join(ROOT_PATH, 'trained', 'bar', 'bar_model'))

                print((os.path.join(ROOT_PATH + 'trained', 'phrase', 'phrase_model')))
                self.gan.load_model(os.path.join(ROOT_PATH + 'trained', 'phrase', 'phrase_model'))

        if self.pre_train:
            self.phrase_generator.train()
            self.bar_generator.train()

        print('train gan')
        self.gan.train()


if __name__ == '__main__':
    model = Model(tf.Session(), False, False)
    model.train()

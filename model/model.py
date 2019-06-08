import os
import tensorflow as tf

from config import *
from .vae.bar import Bar_VAE
from .vae.phrase import Phrase_VAE
from .gan import gan


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
        self.bar_generator = Bar_VAE(self.sess, os.path.join('/home/algorithm/musicGeneration3/trained', 'bar'))
        self.phrase_generator = Phrase_VAE(self.sess, os.path.join('/home/algorithm/musicGeneration3/trained', 'phrase'))
        self.gan = gan.GAN(self.sess, self.save_path, self.note_path)

    def train(self):
        if self.load_gan_model is not None:
            if self.pre_train is not None:
                print((os.path.join('/home/algorithm/musicGeneration3/trained', 'bar', 'bar_model')))
                self.gan.load_model(os.path.join('/home/algorithm/musicGeneration3/trained', 'bar', 'bar_model'))

                print((os.path.join('/home/algorithm/musicGeneration3/trained', 'phrase', 'phrase_model')))
                self.gan.load_model(os.path.join('/home/algorithm/musicGeneration3/trained', 'phrase', 'phrase_model'))

            print((os.path.join('/home/algorithm/musicGeneration3/trained', 'gan', self.load_gan_model, 'gan_model-{}'.format(self.model_number))))
            self.gan.load_model(os.path.join('/home/algorithm/musicGeneration3/trained', 'gan', self.load_gan_model,
                                             'gan_model-{}'.format(self.model_number)))
        self.bar_generator.train()
        self.phrase_generator.train()
        # print('train gan')
        # self.gan.train()


if __name__ == '__main__':
    model = Model(tf.Session(), False, False)
    model.train()

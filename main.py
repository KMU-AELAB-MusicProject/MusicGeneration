import os
import argparse

import tensorflow as tf

from model.v1.model import Model
from config import *


def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_from', type=str, default=None, help="The train number number to start train.")
    parser.add_argument('--model_number', type=str, default=str(GAN_EPOCH), help="The epoch number to start train.")
    parser.add_argument('--pre_train', help="Train Classifier model before train vae.", action='store_true')
    # parser.add_argument('--gpu', '--gpu_device_num', type=str, default="0", help="The GPU device number to use.")

    return parser.parse_args()


def main():
    args = parse_arguments()
    if not os.path.exists(os.path.join(MUSIC_PATH)):
        os.mkdir(os.path.join(MUSIC_PATH))
    if not os.path.exists(os.path.join(MODEL_PATH)):
        os.mkdir(os.path.join(MODEL_PATH))
    if not os.path.exists(os.path.join(MODEL_PATH, 'gan')):
        os.mkdir(os.path.join(MODEL_PATH, 'gan'))

    if args.start_from:
        _max = 0
        for dir_name in os.listdir(os.path.join(MODEL_PATH, 'gan')):
            name = args.start_from + '-'
            if (name in dir_name) and ('-' not in dir_name.split(name)[1]):
                print(dir_name.split(name), dir_name)
                _max = max(int(dir_name.split(name)[1]), _max)
        os.mkdir(os.path.join(MODEL_PATH, 'gan', '{}-{}'.format(args.start_from, str(_max + 1))))
        os.mkdir(os.path.join(MUSIC_PATH, '{}-{}'.format(args.start_from, str(_max + 1))))
        save_path = os.path.join(MODEL_PATH, 'gan', '{}-{}'.format(args.start_from, str(_max + 1)))
        note_path = os.path.join(MUSIC_PATH, '{}-{}'.format(args.start_from, str(_max + 1)))

    else:
        _max = 0
        for dir_name in os.listdir(os.path.join(MODEL_PATH, 'gan')):
            _max = max(int(dir_name.split('-')[0]), _max)
        os.mkdir(os.path.join(MODEL_PATH, 'gan', str(_max + 1)))
        os.mkdir(os.path.join(MUSIC_PATH, str(_max + 1)))
        save_path = os.path.join(MODEL_PATH, 'gan', str(_max + 1))
        note_path = os.path.join(MUSIC_PATH, str(_max + 1))

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    model = Model(tf.Session(), save_path, args.start_from, args.model_number, args.pre_train, note_path)
    model.train()

if __name__ == '__main__':
    main()

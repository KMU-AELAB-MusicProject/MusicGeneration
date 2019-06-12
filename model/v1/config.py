CLASSIFIER_TRAIN_DATA_PATH = '/home/algorithm/music_project/data/classification_data'
GAN_TRAIN_DATA_PATH = '/home/algorithm/music_project/data/gan_data'
BAR_VAE_TRAIN_DATA_PATH = '/home/algorithm/music_project/data/bar_data'
PHRASE_VAE_TRAIN_DATA_PATH = '/home/algorithm/music_project/data/phrase_data'
MODEL_PATH = '/home/algorithm/music_project/trained'
MUSIC_PATH = '/home/algorithm/music_project/music'

PHRASE_SIZE = 4
BAR_SIZE = 96
PITCH_SIZE = 96 # 128
BEAT_RESOLUTION = 24
TIME_SIGNATURES = ['4/4']
MINIMUM_NOTE = 8


CLASSIFIER_EPOCH = 500
BATCH_SIZE = 128
BEAT_COUNT_CLASS_SIZE = 12
ROOT_BEAT_CLASS_SIZE = 3

GAN_EPOCH = 1000
GAN_BATCH_SIZE = 100  # music count
GENERATED_PHRASE_CNT = 10
Z_SIZE = 192

GENERATOR = 'AddArranger'
DISCRIMINATOR = 'Default'

ALPHA = 0.0
THRESHOLD = 0.0

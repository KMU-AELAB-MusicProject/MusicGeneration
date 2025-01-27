"""This script converts a collection of MIDI files to multitrack pianorolls.
"""
import os
import sys
import time
import pickle
import joblib
import random
import warnings
import pretty_midi
import multiprocessing

import numpy as np
from pypianoroll import Multitrack
from google_drive_downloader import GoogleDriveDownloader as gdd

from config import *


phrase_step = [i for i in range(331)]

def get_midi_info(pm):
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time
    else:
        first_beat_time = pm.estimate_beat_start()

    tc_times, tempi = pm.get_tempo_changes()

    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                   pm.time_signature_changes[0].denominator)
    else:
        time_sign = None

    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'constant_time_signature': time_sign,
        'constant_tempo': tempi[0] if len(tc_times) == 1 else None
    }

    return midi_info


def data_maker():
    gan_data = [[], [], []] # train-phrase, pre-phrase, train-phrase-number
    bar_data = [[], []]
    size = 0
    file_num = 0

    index = 0
    total_size = len(os.listdir(NP_FILE_PATH))
    sys.stdout.write('\rProgress: |{0}| {1:0.2f}% '.format('-' * 30, index * 100 / total_size))

    for file in os.listdir(NP_FILE_PATH):
        size += 1
        with open(os.path.join(NP_FILE_PATH, file), 'rb') as f:
            np_data = pickle.load(f)
            note_data, number_data = np_data

            ### pitch selection ###
            data = np.array([[[0 for _ in range(PITCH_SIZE)] for _ in range(BAR_SIZE * PHRASE_SIZE)]])
            data = np.append(data, np.take(note_data, [i for i in range(17, 113)], axis=-1), axis=0)

            # phrase/pre-phrase/phrase-number
            gan_data[0].extend(data[1:])
            gan_data[1].extend(data[:-1])
            gan_data[2].extend(number_data)

            for d in data[1:]:
                for i in range(0, len(d), BAR_SIZE):
                    bar_data[0].append(d[i:i + BAR_SIZE])
                    bar_data[1].append(d)

        if size == 60:
            with open(os.path.join(DATA_PATH, 'gan_data', 'gan_data{}.pkl'.format(file_num)), 'wb') as fp:
                pickle.dump(gan_data, fp)
                del gan_data

            with open(os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.pkl'.format(file_num)), 'wb') as fp:
                pickle.dump(bar_data, fp)
                del bar_data

            size = 0
            file_num += 1
            gan_data = [[], [], []]
            bar_data = [[], []]

        index += 1
        sys.stdout.flush()
        sys.stdout.write('\rProgress: |{0}{1}| {2:0.2f}% '.format('#' * (30 * index // total_size),
                                                                  '-' * (30 * (total_size - index) // total_size),
                                                                  index * 100 / total_size))
    else:
        if size != 0:
            with open(os.path.join(DATA_PATH, 'gan_data', 'gan_data{}.pkl'.format(file_num)), 'wb') as fp:
                pickle.dump(gan_data, fp)
            with open(os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.pkl'.format(file_num)), 'wb') as fp:
                pickle.dump(bar_data, fp)
                del bar_data

            sys.stdout.flush()
            sys.stdout.write('\rProgress: |{0}| {1:0.2f}% \n'.format('#' * 30, index * 100 / total_size))


def converter(file):
    phrase_size = BAR_SIZE * PHRASE_SIZE
    try:
        midi_md5 = os.path.splitext(os.path.basename(os.path.join(MIDI_FILE_PATH, file)))[0]
        multitrack = Multitrack(beat_resolution=BEAT_RESOLUTION, name=midi_md5)

        pm = pretty_midi.PrettyMIDI(os.path.join(MIDI_FILE_PATH, file))
        multitrack.parse_pretty_midi(pm)
        midi_info = get_midi_info(pm)

        length = multitrack.get_max_length()
        padding_size = phrase_size - (length % phrase_size) if length % phrase_size else 0
        tmp = length // phrase_size + (1 if length % phrase_size else 0)

        multitrack.binarize(0)
        data = multitrack.get_merged_pianoroll(mode='max')

        if padding_size:
            data = np.concatenate((data, np.array([[0 for _ in range(128)] for _ in range(padding_size)])), axis=0)

        data_by_phrase = []
        phrase_number = [330] + [i for i in range(tmp - 2, -1, -1)]
        for i in range(0, len(data), phrase_size):
            data_by_phrase.append(data[i:i + phrase_size])

        with open(os.path.join(NP_FILE_PATH, '{}_{}.pkl'.format(tmp, midi_md5)), 'wb') as fp:
            pickle.dump([np.array(data_by_phrase), phrase_number], fp)

        return (midi_md5, midi_info)

    except Exception as e:
        print(e)
        return None


def main():
    if not os.path.exists(os.path.join(DATA_PATH, 'np')):
        os.mkdir(os.path.join(DATA_PATH, 'np'))
    if not os.path.exists(os.path.join(DATA_PATH, 'bar_data')):
        os.mkdir(os.path.join(DATA_PATH, 'bar_data'))
    if not os.path.exists(os.path.join(DATA_PATH, 'phrase_data')):
        os.mkdir(os.path.join(DATA_PATH, 'phrase_data'))
    if not os.path.exists(os.path.join(DATA_PATH, 'gan_data')):
        os.mkdir(os.path.join(DATA_PATH, 'gan_data'))
    if not os.path.exists(os.path.join(DATA_PATH, 'midi')):
        gdd.download_file_from_google_drive(file_id='1L854vE7ghnI8uD-gR5McDZ71Wt-g9iH4',
                                            dest_path=os.path.join(DATA_PATH, 'midi.zip'),
                                            unzip=True)

    midi_info = {}

    warnings.filterwarnings('ignore')

    files = list(filter(lambda x: '.mid' in x, os.listdir(MIDI_FILE_PATH)))

    # if multiprocessing.cpu_count() > 1:
    #      kv_pairs = joblib.Parallel(n_jobs=multiprocessing.cpu_count() - 1, verbose=5)(joblib.delayed(converter)(file) for file in files)
    #      for kv_pair in kv_pairs:
    #         if kv_pair is not None:
    #             midi_info[kv_pair[0]] = kv_pair[1]
    # else:
    #     for file in files:
    #         kv_pair = converter(file)
    #         if kv_pair is not None:
    #             midi_info[kv_pair[0]] = kv_pair[1]

    with open(os.path.join(DATA_PATH, 'midi_info.pkl'), 'wb') as fp:
        pickle.dump(midi_info, fp)

    data_maker()


if __name__ == "__main__":
    main()

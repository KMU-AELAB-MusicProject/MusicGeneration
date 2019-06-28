import pickle


def data_spliter(data_path, test_data_rate=0.3):
    with open(data_path, 'rb') as fp:
        tmp_data = pickle.load(fp)  # data shape -> [[data_1, target_1]...[data_n, target_n]]

    data, target = tmp_data
    size = len(data)
    train_data, train_target, test_data, test_target = [], [], [], []

    for i in range(int(size * test_data_rate)):
        test_data.append(data[i])
        test_target.append(target[i])

    for i in range(int(size * test_data_rate), size):
        train_data.append(data[i])
        train_target.append(target[i])

    return train_data, train_target, test_data, test_target


def batch_iterator(data, target, size):
    for i in range(0, len(data), size):
        yield [data[i:i + size], target[i:i + size], len(target[i:i + size])]


def gan_batch_iterator(data, size):
    phrase, pre_phrase, phrase_number = data
    for i in range(0, len(phrase), size):
        yield [phrase[i:i + size], pre_phrase[i:i + size], phrase_number[i:i + size]]

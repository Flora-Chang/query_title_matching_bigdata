# encoding: utf-8
import numpy as np
import json
from util import FLAGS
import tensorflow as tf

def strQ2B(ustring):
    '''全转半'''
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def get_vocab_dict(input_file=FLAGS.vocab_path):
    words_dict = {}
    cnt = 0
    with open(input_file) as f:
        for word in f:
            words_dict[word.strip()] = cnt
            cnt += 1
    return words_dict

def get_id2word(input_file=FLAGS.vocab_path):
    id2word = [0 for i in range(4200001)]
    cnt = 0
    with open(input_file) as f:
        for word in f:
            id2word[cnt] = word
            cnt += 1
    return id2word

def termid_to_id(input_file=FLAGS.termid_path):
    termid_dict = {}
    cnt =  0
    with open(input_file, "r") as f:
        for termid in f:
            termid_dict[termid.strip()] = cnt
            cnt += 1
        termid_dict["<unk>"] = cnt
        cnt += 1
        return termid_dict


def get_word_vector(input_file=FLAGS.vectors_path):
    word_vectors = []
    with open(input_file) as f:
        for line in f:
            line = [float(v) for v in line.strip().split()]
            word_vectors.append(line)
    return word_vectors

# output batch_major data
def batch(inputs, threshold_length):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    '''
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
        max_sequence_length = min(max_sequence_length, threshold_length)
    '''
    max_sequence_length = threshold_length
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int64)  # == PAD
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            if j >= threshold_length:
                sequence_lengths[i] = max_sequence_length
                break
            inputs_batch_major[i, j] = element
    # [batch_size, max_time] -> [max_time, batch_size]
    # inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_batch_major

class LoadTrainData(object):
    def __init__(self, vocab_dict, data_path, query_len_threshold, title_len_threshold, batch_size=64):
        self.vocab_dict = vocab_dict
        self.batch_size = batch_size
        self.title_len_threshold = title_len_threshold  # 句子长度限制
        self.query_len_threshold = query_len_threshold
        self.data = open(data_path, 'r').readlines()
        self.batch_index = 0
        print("len data: ", len(self.data))

    def _word_2_id(self, word):
        if word in self.vocab_dict.keys():
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict['<unk>']
        return res

    def next_batch(self, shuffle=True):
        self.batch_index = 0
        self.cnt = 0
        data = np.array(self.data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size / self.batch_size) + 1
        print("training_set: ", data_size, num_batches_per_epoch)

        if shuffle:
            np.random.shuffle(data)
        shuffled_data = data

        while self.batch_index < num_batches_per_epoch \
                and (self.batch_index + 1) * self.batch_size <= data_size:
            queries = []
            titles = []
            labels = []
            ranks = []
            batch_feature_local = []
            start_index = self.batch_index * self.batch_size
            self.batch_index += 1
            end_index = min(self.batch_index * self.batch_size, data_size)
            batch_data = shuffled_data[start_index:end_index]

            for line in batch_data.tolist():
                line = line.strip().split('\t')
                self.cnt +=1
                label = int(line[3])
                if label >=2:
                    label=1
                else:
                    label = 0

                ori_query = line[1].split()
                query = list(map(self._word_2_id, ori_query))
                ori_title = line[2].split()
                title = list(map(self._word_2_id, ori_title))
                rank = int(line[0].strip())
                queries.append(query)
                titles.append(title)
                labels.append(label)
                ranks.append(rank)
                '''
                local_match = np.zeros(shape=[self.query_len_threshold, self.title_len_threshold], dtype=np.int32)
                for i in range(min(self.query_len_threshold, len(ori_query))):
                    for j in range(min(self.title_len_threshold, len(title))):
                        if ori_query[i] == ori_title[j]:
                            local_match[i, j] = 1
                batch_feature_local.append(local_match)  # [batch_size, query_length, title_length]
                '''

            queries = batch(queries, self.query_len_threshold)
            titles = batch(titles, self.title_len_threshold)
            yield  queries, titles, labels,ranks
        print("self.cnt:", self.cnt)


class LoadTestData(object):
    def __init__(self, vocab_dict, data_path, query_len_threshold, title_len_threshold, batch_size=64):
        self.vocab_dict = vocab_dict
        self.query_len_threshold = query_len_threshold
        self.title_len_threshold = title_len_threshold
        self.index = 0
        self.data = open(data_path, encoding="utf-8").readlines()
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.cnt = 0

    def _word_2_id(self, word):
        if word in self.vocab_dict:
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict["<unk>"]
            #print(word, res)
        return res

    def next_batch(self):

        while (self.index ) * self.batch_size < self.data_size:
            if (self.index + 1) * self.batch_size <= self.data_size:
                batch_data = self.data[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            else:
                batch_data = self.data[self.index * self.batch_size: self.data_size]
            self.index += 1
            queries = []
            titles = []
            labels = []
            ranks = []
            batch_feature_local = []
            for line in batch_data:
                self.cnt += 1
                line = line.strip().split('\t')
                label = int(line[3])
                if label >=2:
                    label = 1
                else:
                    label=0
                '''
                if label == 2:
                    label = 1
                else:
                    label = 0
                '''
                ori_query = line[1].split()
                query = list(map(self._word_2_id, ori_query))
                ori_title = line[2].split()
                title = list(map(self._word_2_id, ori_title))
                rank = int(line[0].strip())
                queries.append(query)
                titles.append(title)
                labels.append(label)
                ranks.append(rank)
               
            queries = batch(queries, self.query_len_threshold)
            titles = batch(titles, self.title_len_threshold)
            yield  queries, titles, labels, ranks

        print("self.cnt:", self.cnt)

class LoadClickData(object):
    def __init__(self, vocab_dict, data_path, query_len_threshold, title_len_threshold, shuffle=True, batch_size=64):
        self.vocab_dict = vocab_dict
        self.query_len_threshold = query_len_threshold
        self.title_len_threshold = title_len_threshold
        self.index = 0
        self.data = open(data_path, 'r').readlines()
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.cnt = 0

    def _word_2_id(self, word):
        if word in self.vocab_dict:
            res = self.vocab_dict[word]
        else:
            res = self.vocab_dict["<unk>"]
        return res

    def next_batch(self):
        self.batch_index = 0
        self.cnt = 0
        self.index = 0
        #data = np.array(self.data)
        data_size = len(self.data)
        num_batches_per_epoch = int(data_size / self.batch_size) + 1
        print("training_set: ", data_size, num_batches_per_epoch)
        '''
        if self.shuffle:
            np.random.shuffle(self.data)
        shuffled_data = self.data
        '''
        while (self.index ) * self.batch_size < self.data_size:
            if (self.index + 1) * self.batch_size <= self.data_size:
                batch_data = self.data[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            else:
                batch_data = self.data[self.index * self.batch_size: self.data_size]
            self.index += 1
            queries = []
            pos_titles = []
            neg_titles = []
            titles = []
            labels = [0] * self.batch_size
            ranks = [0] * self.batch_size
            for line in batch_data:
                self.cnt += 1
                line = line.strip().split('\t')
                ori_query = line[0].split()
                query = list(map(self._word_2_id, ori_query))
                ori_title1 = line[1].split()
                ori_title2 = line[2].split()
                title1 = list(map(self._word_2_id, ori_title1))
                title2 = list(map(self._word_2_id, ori_title2))
                queries.append(query)
                pos_titles.append(title1)
                neg_titles.append(title2)
                '''
                if(self.cnt<=2):
                    print("ori_query: ", ori_query)
                    print("query: ", ori_query)
                    print("ori_title1: ", ori_title1)
                    print("title1: ", title1)
                    print("ori_title2: ", ori_title2)
                    print("title2: ", title2)
                '''

            queries = batch(queries, self.query_len_threshold)
            queries = queries.repeat(2, axis=0)
            pos_titles = batch(pos_titles, self.title_len_threshold)
            neg_titles = batch(neg_titles, self.title_len_threshold)
            for i, j in zip(pos_titles, neg_titles):
                titles.append([i, j])
            titles = np.array(titles, dtype=np.int64).reshape([-1, self.title_len_threshold])

            yield queries, titles, labels, ranks

        print("self.cnt:", self.cnt)

class LoadTfRecord(object):
    def __init__(self):
        pass

    def load_tfrecord(self, tfrecord_paths):
        # Even when reading in multiple threads, share the filename queue.
        Qlen = 20
        Alen = 20
        filename_queue = tf.train.string_input_producer(tfrecord_paths)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'query': tf.FixedLenFeature(Qlen, tf.int64),
                                               'pos_title': tf.FixedLenFeature(Alen, tf.int64),
                                               'neg_title': tf.FixedLenFeature(Alen, tf.int64),
                                           })
        query = features['query']
        pos_title = features['pos_title']
        neg_title = features['neg_title']
        print(query)
        print(pos_title)
        print(neg_title)

        query_, pos_title_, neg_title_ = tf.train.shuffle_batch(
            [query, pos_title, neg_title],
            batch_size=128,
            capacity=10000,
            num_threads=5,
            enqueue_many=False,
            min_after_dequeue=5000)

        return  query_, pos_title_, neg_title_

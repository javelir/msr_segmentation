# -*- coding: utf-8 -*-
""" Read segment data and output BMES data iterator
"""

import logging

import numpy as np

from random import shuffle
from collections import defaultdict


class AnnotatedData(object):
    """ Annotated data iterator
      Build data iterator based on annotated (segmented) data.
        path: location of input file containing segmented sentences.
        sep: separation mark
        encoding: file encoding
        limit: input row number limit from given file
      Members:
        source_len: number of source sentences
        dictionary: word-number dictionary
        target_dictionary: label-number dictionary
        reverse_target_dictionary: number-word dictionary
        pad_id: id(number) for padding unit
    """
    pad_id = 0
    def __init__(self, path, sep=' ', encoding='utf-8', limit=None, ):
        with open(path, mode='r', encoding=encoding) as instream:
            raw = instream.read().splitlines()
        if limit:
            raw = raw[:limit]
        self.source_len = len(raw)
        self._remaining_indexes = list(range(self.source_len))
        shuffle(self._remaining_indexes)
        self._separator = sep
        self.dictionary = {}
        self.target_dictionary = {key:val for val, key in enumerate('bmes', start=1)}
        # self.target_dictionary = {key:val for val, key in enumerate('bmes', start=0)}
        self.reverse_target_dictioanry = {val:key for key, val in self.target_dictionary.items()}
        self.build_dataset(inputs=raw)

    def build_dataset(self, inputs):
        """ Build dataset from raw string
          inputs are segmented sentences separated by '\n', with words in each sentence
          separated by given separator as self._separator.
        """
        letter_dic = defaultdict(int)
        raw_sources = []
        raw_targets = []
        lengths = []
        for line in inputs:
            _source = []
            _target = []
            _length = 0
            for word in line.split(self._separator):
                _length += len(word)
                if len(word) == 1:
                    _source.append(word)
                    letter_dic[word] += 1
                    _target.append('s')
                    continue
                word_len = len(word)
                for idx, letter in enumerate(word, start=1):
                    # must start from 1 depending on the following conditions
                    _source.append(letter)
                    letter_dic[letter] += 1
                    if idx == 1:
                        _target.append('b')
                    elif idx == word_len:
                        _target.append('e')
                    else:
                        _target.append('m')
            raw_sources.append(_source)
            raw_targets.append(_target)
            lengths.append(_length)

        revordered_dic = sorted(letter_dic.items(), key=lambda x:x[1], reverse=True)
        # letter_dictionary = {key:idx for idx, (key, _) in enumerate(revordered_dic, start=1)}
        letter_dictionary = {key:idx for idx, (key, _) in enumerate(revordered_dic, start=0)}
        self.raw_sources = raw_sources
        self.raw_targets = raw_targets
        self.training_sources = [
            [letter_dictionary.get(x) for x in sent]
            for sent in self.raw_sources
        ]
        self.training_targets = [
            [self.target_dictionary.get(x) for x in target]
            for target in self.raw_targets
        ]
        self.letter_dictionary = letter_dictionary
        self.reverse_words_dictionary = {val:key for key, val in letter_dictionary.items()}
        logging.info(
            'Training dataset build: %d observations with vocab size=%d.',
            self.source_len, len(self.letter_dictionary))
        self.lengths = lengths
        self.vocab_size = len(self.reverse_words_dictionary)
        
    def get_batch(self, size):
        """ x """
        indexes = self._gen_batch_indexes(size=size)
        return [self.training_sources[idx] for idx in indexes],\
               [self.training_targets[idx] for idx in indexes],\
               [self.lengths[idx] for idx in indexes]

    def get_padded_batch(self, size):
        """ x """
        indexes = self._gen_batch_indexes(size=size)
        lengths = [self.lengths[idx] for idx in indexes]
        max_length = max(lengths)
        _sources = []
        _targets = []
        _lengths = []
        for idx in indexes:
            _source = self.training_sources[idx]
            _target = self.training_targets[idx]
            _sources.append(_source + [self.pad_id] * (max_length - len(_source)))
            _targets.append(_target + [self.pad_id] * (max_length - len(_target)))
        return np.array(_sources), np.array(_targets), np.array(lengths)

    def get_letter_batch(self, size):
        """ x """
        indexes = self._gen_batch_indexes(size=size)
        return [self.raw_sources[idx] for idx in indexes],\
               [self.raw_targets[idx] for idx in indexes],\
               [self.lengths[idx] for idx in indexes]

    def _gen_batch_indexes(self, size):
        """ x """
        if len(self._remaining_indexes) < size:
            self._remaining_indexes = list(range(self.source_len))
            shuffle(self._remaining_indexes)
        return [self._remaining_indexes.pop() for _ in range(size)]

    def words_to_ids(self, words):
        """ x """
        return np.array([self.letter_dictionary.get(word) for word in words])

    def ids_to_words(self, ids):
        """ x """
        return [self.reverse_words_dictionary.get(x, 'UNK') for x in ids]

    def ids_to_targets(self, ids):
        """ x """
        return [self.reverse_target_dictioanry.get(x, 'UNK') for x in ids]

    def targets_to_ids(self, targets):
        """ x """
        return np.array([self.target_dictionary.get(targ) for targ in targets])


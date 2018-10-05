# -*- coding: utf-8 -*-
""" Read segment data and output BMES data iterator
"""

import os
import logging
import re
import pickle
import json

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
    unk = "__unk__"
    rgx_non_ch = re.compile("^[-0-9a-zA-Z\\.]+$")
    def __init__(self, path=None, sep=' ', encoding='utf-8', limit=None, min_count=5):
        """
         Load data into memory and keep given number of lines if limit given,
         build dictionary and reverse dictionary for taggs, build dataset
        """
        self.logger = logging.getLogger(__name__)
        self.source_len = 0
        self.vocabulary_size = 0
        self._remaining_indexes = []
        self._separator = sep
        self.target_dictionary = {}
        self.reverse_target_dictionary = {}
        self.letter_dictionary = {}
        self.reverse_words_dictionary = {}
        self.lengths = []
        self.raw_sources = []
        self.raw_targets = []
        self.training_sources = []
        self.training_targets = []
        if path:
            self.load_from_path(path=path, limit=limit, encoding=encoding, min_count=min_count)

    def load_from_path(self, path, limit, encoding, min_count):
        with open(path, mode='r', encoding=encoding) as instream:
            raw = instream.read().splitlines()
        if limit:
            raw = raw[:limit]
        self.source_len = len(raw)
        self._remaining_indexes = list(range(self.source_len))
        shuffle(self._remaining_indexes)
        self.target_dictionary = {key:val for val, key in enumerate('bmes', start=1)}
        self.target_dictionary["padding"] = 0
        # self.target_dictionary = {key:val for val, key in enumerate('bmes', start=0)}
        self.reverse_target_dictioanry = {val:key for key, val in self.target_dictionary.items()}
        self.build_dataset(inputs=raw, min_count=min_count)

    @classmethod
    def load(cls, target_dir):
        obj = cls()
        path = os.path.join(target_dir, "source_dictionary.json")
        with open(path, "r", encoding="utf-8") as fin:
            obj.letter_dictionary = json.load(fin)
        obj.reverse_words_dictionary = {val:key for key,val in obj.letter_dictionary.items()}
        path = os.path.join(target_dir, "target_dictionary.json")
        with open(path, "r", encoding="utf-8") as fin:
            obj.target_dictionary = json.load(fin)
        obj.reverse_target_dictionary = {val:key for key,val in obj.target_dictionary.items()}
        path = os.path.join(target_dir, "dataset.txt")
        with open(path, "r", encoding="utf-8") as fin:
            for _line in fin:
                line = _line.strip().split("\t")
                if line[0] == "vocabulary_size":
                    obj.vocabulary_size = int(line[1])
                if line[0] == "num_sents":
                    obj.source_len = int(line[1])
                if line[0] == "source":
                    obj.raw_sources.append(line[2:])
                    obj.training_sources.append(obj.words_to_ids(line[2:]))
                    obj.lengths.append(len(line) - 2)
                if line[0] == "target":
                    obj.raw_targets.append(line[2:])
                    obj.training_targets.append(obj.targets_to_ids(line[2:]))
        return obj

    def save(self, target_dir):
        path = os.path.join(target_dir, "dataset.txt")
        with open(path, "w+", encoding="utf-8") as fout:
            fout.write(f"vocabulary_size\t{self.vocabulary_size}\n")
            fout.write(f"num_sents\t{self.source_len}\n")
            for idx in range(self.source_len):
                one_source = "\t".join(self.raw_sources[idx])
                one_target = "\t".join(self.raw_targets[idx])
                fout.write(f"source\t{idx}\t{one_source}\n")
                fout.write(f"target\t{idx}\t{one_target}\n")
        path = os.path.join(target_dir, "source_dictionary.json")
        with open(path, "w+", encoding="utf-8") as fout:
            json.dump(self.letter_dictionary, fout)
        path = os.path.join(target_dir, "target_dictionary.json")
        with open(path, "w+", encoding="utf-8") as fout:
            json.dump(self.target_dictionary, fout)

    def preprocess_string(self, line):
        target = '“'
        count = line.count(target)
        #print(f"count={count}, line={line} {target==line[0]} target={target}")
        if count % 2 == 1:
            line = re.sub(target, "", line)
        return line.strip().lower().split(self._separator)

    def build_dataset(self, inputs, min_count=5):
        """ Build dataset from raw string
          inputs are segmented sentences separated by '\n', with words in each sentence
          separated by given separator as self._separator.
        """
        letter_dic = defaultdict(int)
        raw_sources = []
        raw_targets = []
        lengths = []
        for line_idx, line in enumerate(inputs, start=1):
            _source = []
            _target = []
            _length = 0
            for word in self.preprocess_string(line):
                # consider english word and numbers
                if self.rgx_non_ch.search(word):
                    word_len = 1
                else:
                    word_len = len(word)
                _length += word_len
                if word_len == 1:
                    _source.append(word)
                    letter_dic[word] += 1
                    _target.append('s')
                    continue
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
            if line_idx % 10000 == 0:
                self.logger.info(f"Finish processing {line_idx} lines")

        self.logger.info(f"Number of raw input characters={len(letter_dic)}")
        revordered_dic = sorted(letter_dic.items(), key=lambda x:x[1], reverse=True)
        letter_dictionary = {key:idx for idx, (key, _) in enumerate(revordered_dic, start=1) if idx >= min_count}
        letter_dictionary[self.unk] = 0
        self.raw_sources = raw_sources
        self.raw_targets = raw_targets
        self.training_sources = [
            [letter_dictionary.get(x, 0) for x in sent]
            for sent in self.raw_sources
        ]
        self.training_targets = [
            [self.target_dictionary[x] for x in target]
            for target in self.raw_targets
        ]
        self.letter_dictionary = letter_dictionary
        self.reverse_words_dictionary = {val:key for key, val in letter_dictionary.items()}
        self.reverse_target_dictionary = {val:key for key,val in self.target_dictionary.items()}
        self.logger.info(f"Training dataset build: {self.source_len} observations with vocab size={len(self.letter_dictionary)}.")
        self.lengths = lengths
        self.vocab_size = len(self.reverse_words_dictionary)
        
    def get_batch(self, size):
        """ Generate a batch of training sets without padding """
        indexes = self._gen_batch_indexes(size=size)
        return [self.training_sources[idx] for idx in indexes],\
               [self.training_targets[idx] for idx in indexes],\
               [self.lengths[idx] for idx in indexes]

    def get_padded_batch(self, size):
        """ Generate a batch of training sets with padding  """
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
        """ Generate (non-repeat for an epoch) random indexes of given size """
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
        """ Convert source ids to words """
        #print(ids)
        return [self.reverse_words_dictionary.get(x, self.unk) for x in ids]

    def ids_to_targets(self, ids):
        """ x """
        return [self.reverse_target_dictionary.get(x, self.unk) for x in ids]

    def targets_to_ids(self, targets):
        """ x """
        return np.array([self.target_dictionary.get(targ) for targ in targets])

    def print_couples(self, _sources, _targets, lengths):
        """ Print out in a nice form """
        if len(_sources) != len(_targets) or len(_sources) != len(lengths):
            raise ValueError("Unequal lengths of inputs")
        words = [self.ids_to_words(ids=x) for x in _sources]
        marks = [self.ids_to_targets(ids=x) for x in _targets]
        #print(f"lengths={lengths}, _sources={_sources}, words={words}, _targets={_targets}, marks={marks}")
        data = zip(lengths, _sources, words, _targets, marks)
        for _len, _sid, _w, _tid, _m in data:
            #print(f"_len={_len}, _sid={_sid}, _tid={_tid}, _m={_m}")
            print(f"length={_len}")
            elem_data = zip(_sid, _w, _tid, _m)
            for elem in elem_data:
                print("\t".join([str(x) for x in elem]))
        return


if __name__ == "__main__":
    from config import BasicConfig
    annotated_data = AnnotatedData(path=BasicConfig.msr_utf8, limit=1, min_count=0)
    print(annotated_data.raw_sources)
    print(annotated_data.training_sources)
    print(annotated_data.training_targets)
    annotated_data.save("../data/")
    #sources, targets, lengths = annotated_data.get_batch(size=1)
    #annotated_data.print_couples(_sources=sources, _targets=targets, lengths=lengths)
    new_annotated_data = AnnotatedData.load("../data/")
    print(new_annotated_data.raw_sources)
    print(new_annotated_data.training_sources)
    print(new_annotated_data.training_targets)
    sources, targets, lengths = new_annotated_data.get_batch(size=1)
    new_annotated_data.print_couples(_sources=sources, _targets=targets, lengths=lengths)

# -*- coding: utf-8 -*-
""" Sequence tagging """

import os
import logging
import numpy as np
import tensorflow as tf


class TrainingParms(object):
    """ x """
    emb_size = 100
    hidden_size = 128
    nbatches = 1000
    batch_size = 128
    burn_out = 100
    lr = 0.001
    decay_gap = 100
    lr_decay = 0.9
    nepoch_no_improve = 10
    dropout = 0.5
    model_path = './model/'

    # def __init__(self, vocab_size, num_targets=4):
    def __init__(self, vocab_size, num_targets=5):
        self._vocab_size = vocab_size
        self._num_targets = num_targets

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def num_targets(self):
        return self._num_targets


class ChineseSegmentationModel(object):
    """ x """
    def __init__(self, parms, data_generator):
        self.parms = parms
        self.data_generator = data_generator

    def build_graph(self):
        """ x """
        self.build_placeholders()
        self.build_embeddings()
        self.build_forward_op()
        self.build_loss()
        self.build_optimizer()

    def build_placeholders(self):
        """ x """
        self.source_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name='source_ids',
        )
        logging.info('Placeholder source_ids:%s', self.source_ids.name)
        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name='sequence_lengths',
        )
        logging.info('Placeholder sequence_lengths:%s', self.sequence_lengths.name)
        self.target_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name='target_ids',
        )
        logging.info('Placeholder target_ids:%s', self.target_ids.name)
        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=[],
            name='dropout',
        )
        self.lr = tf.placeholder(
            dtype=tf.float32,
            shape=[],
            name='learning_rate',
        )

    def build_embeddings(self):
        """ x """
        embeddings = tf.get_variable(
            name='embeddings',
            dtype=tf.float32,
            shape=[self.parms.vocab_size, self.parms.emb_size],
        )
        self.char_embeddings = tf.nn.embedding_lookup(
            params=embeddings,
            ids=self.source_ids,
            name='char_embeddings',
        )

    def build_forward_op(self):
        """ x """
        # cell_list = [
        #     tf.contrib.rnn.BasicLSTMCell(
        #         num_units=self.parms.num_units,
        #         forget_bias=forget_bias,
        #     )
        #     for _ in range(self.parms.num_layers)
        # ]
        fw_cell = tf.contrib.rnn.LSTMCell(num_units=self.parms.hidden_size,)
        bw_cell = tf.contrib.rnn.LSTMCell(num_units=self.parms.hidden_size,)
        # batch_size * time_step * output_size
        (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=self.char_embeddings,
            sequence_length=self.sequence_lengths,
            dtype=tf.float32,
        )
        output = tf.concat([fw_output, bw_output], axis=-1)
        output = tf.nn.dropout(
            x=output, keep_prob=self.dropout,
        )
        output_weight = tf.get_variable(
            name='output_weight',
            dtype=tf.float32,
            shape=[2 * self.parms.hidden_size, self.parms.num_targets],
        )
        output_bias = tf.get_variable(
            name='output_bias',
            dtype=tf.float32,
            shape=[self.parms.num_targets],
            initializer=tf.zeros_initializer(),
        )
        nsteps = tf.shape(output)[1]
        output = tf.reshape(
            tensor=output,
            shape=[-1, 2 * self.parms.hidden_size],
        )
        yhat = tf.matmul(output, output_weight) + output_bias
        self.logits = tf.reshape(
            tensor=yhat,
            shape=[-1, nsteps, self.parms.num_targets],
        )
        self.labels_pred = tf.cast(
            x=tf.argmax(
                self.logits, axis=-1
            ),
            dtype=tf.int32,
        )

    def build_loss(self):
        """ x """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.target_ids,
            logits=self.logits,
        )
        mask = tf.sequence_mask(self.sequence_lengths)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar('loss', self.loss)

    def build_optimizer(self):
        """ x """
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        _grads, _vars = zip(*optimizer.compute_gradients(self.loss))
        # grads, gnorm = tf.clip_by_global_norm(
        #     t_list=_grads,
        #     clip_norm=self.parms.clip,
        # )
        # grads = tf.clip_by_value(
        #     t=_grads, clip_value_min=-5.0, clip_value_max=5.0
        # )
        # self.train_op = optimizer.apply_gradients(zip(grads, _vars))
        self.train_op = optimizer.minimize(self.loss)

    def init_session(self):
        """ x """
        self.sess = tf.Session()
        self.sess.run(
            tf.global_variables_initializer()
        )

    def train(self):
        """ x """
        best_score = 0
        nepoch_no_improve = 0
        for _batch in range(self.parms.nbatches):
            sents, tags, lens = self.data_generator.get_padded_batch(size=self.parms.batch_size)
            score = self.run_batch(sents=sents, tags=tags, lens=lens)
            if _batch > self.parms.burn_out and _batch % self.parms.decay_gap == 0:
                self.parms.lr *= self.parms.lr_decay
            if score >= best_score:
                nepoch_no_improve = 0
            else:
                nepoch_no_improve += 1
                if nepoch_no_improve >= self.parms.nepoch_no_improve:
                    print('No improve stopping')
            print('Batch: ', _batch, 'Score: ', score)

    def run_batch(self, sents, tags, lens):
        """ x """
        _, train_loss = self.sess.run(
            fetches=[self.train_op, self.loss],
            feed_dict={
                self.source_ids:sents,
                self.sequence_lengths:lens,
                self.target_ids:tags,
                self.lr:self.parms.lr,
                self.dropout:self.parms.dropout,
            }
        )
        return train_loss

    def trial_batch(self, sents, tags, lens, target):
        """ x """
        result = self.sess.run(
            fetches=target,
            feed_dict={
                self.source_ids:sents,
                self.sequence_lengths:lens,
                self.target_ids:tags,
                self.lr:self.parms.lr,
                self.dropout:self.parms.lr,
            }
        )
        return result

    # def save_session(self):
    #     """ x """
    #     if not os.path.exists(self.parms.model_path):
    #         os.makedirs(self.parms.model_path)
    #     self.saver.save(self.sess, self.parms.model_path)

    def predict(self, sent):
        """ x """
        ids = self.data_generator.words_to_ids(words=sent)
        seqlength = np.array([len(ids)])
        labels_pred = self.sess.run(
            fetches=self.labels_pred,
            feed_dict={
                self.source_ids:np.array([ids]),
                self.sequence_lengths:seqlength,
                self.dropout:0.5
            }
        )
        targets = self.data_generator.ids_to_targets(ids=labels_pred[0])
        return targets



def test_project(_):
    from data_iterator import AnnotatedData
    from config import Config
    # data_generator = AnnotatedData(path=Config.test_utf8, limit=100)
    data_generator = AnnotatedData(path=Config.pku_utf8, limit=10000)
    parms = TrainingParms(vocab_size=data_generator.vocab_size)
    parms.lr = parms.lr
    parms.lr_decay = 0.999
    parms.nbatches = 1000
    parms.batch_size = 128
    parms.burn_out = 300
    model = ChineseSegmentationModel(parms=parms, data_generator=data_generator)
    model.build_graph()
    model.init_session()
    model.train()
    target_sents = [
        #"今天的天气真的好棒呀！",
        "我们相信通过这些国家和地区的努力以及有关的国际合作，情况会逐步得到缓解。",
        "继续把建设有中国特色社会主义事业推向前进。"
    ]
    for target_sent in target_sents:
        for x, y in zip(target_sent, model.predict(target_sent)):
            print(x, y)


if __name__ == '__main__':
    test_project(None)

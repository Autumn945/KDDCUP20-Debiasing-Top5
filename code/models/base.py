if __name__ == '__main__':
    raise Exception

import tensorflow as tf
import numpy as np
import utils
from utils import args
from utils import Object
import dataset

eps = 1e-7
inf = 1e31

class DEBUG:
    fit_show_list = []
    inf_show_list = []
    block = False
    @staticmethod
    def print_when_fit(*v, block=True):
        if not args.debug:
            return
        if block:
            DEBUG.block = True
        for _v in v:
            DEBUG.fit_show_list.append(_v)

    @staticmethod
    def print_when_inference(*v, block=True):
        if not args.debug:
            return
        if block:
            DEBUG.block = True
        for _v in v:
            DEBUG.inf_show_list.append(_v)

    @staticmethod
    def when_run(v):
        if not args.debug:
            return
        if v:
            print('--- debug ---')
        for _v in v:
            try:
                print(_v.tolist())
            except Exception:
                print(_v)

        if DEBUG.block:
            input('----- continue -----')

class UTILS:
    mid_pop = 5

    @staticmethod
    def get_session():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1.0,
            visible_device_list=args.get('gpu', '0'),
            allow_growth=True,
        )
        config = tf.ConfigProto(gpu_options=gpu_options)
        session = tf.Session(config=config)
        return session

    @staticmethod
    def minimizer(loss):
        loss += tf.losses.get_regularization_loss()
        if args.opt.lower() == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=args.lr)
        elif args.opt.lower() == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
        elif args.opt.lower() == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=args.lr)
        elif args.opt.lower() == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=args.lr, momentum=0.9)
        elif args.opt.lower() == 'momentum5':
            opt = tf.train.MomentumOptimizer(learning_rate=args.lr, momentum=0.5)
        elif args.opt.lower() == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate=args.lr)
            # RMSprop
        else:
            raise Exception(f'unknown opt: {args.opt}')

        minimizer_op = opt.minimize(loss)
        return minimizer_op

    @staticmethod
    def l2_loss(name):
        alpha = args.get(f'l2_{name}', 0)
        if alpha < 1e-10:
            return None
        return lambda x: alpha * tf.nn.l2_loss(x)

    @staticmethod
    def mask_logits(logits, mask):
        mask = tf.cast(mask, tf.float32)
        logits = logits * mask - (1 - mask) * 1e12
        return logits

    @staticmethod
    def LSTM(seq, seq_length=None, name='text'):
        k = seq.get_shape().as_list()[-1]
        with tf.variable_scope(f'LSTM_{name}', reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.LSTMCell(k)
            # ret: outputs, state
            outputs, state = tf.nn.dynamic_rnn(cell, seq, sequence_length=seq_length, dtype=tf.float32)
            return state.h

    @staticmethod
    def GRU(seq, seq_length=None, mask=None, name='text'):
        k = seq.get_shape().as_list()[-1]
        if seq_length is None and mask is not None:
            seq_length = tf.reduce_sum(tf.cast(mask, tf.int32), -1)
        with tf.variable_scope(f'GRU_{name}', reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.GRUCell(k)
            # ret: outputs, state
            outputs, state = tf.nn.dynamic_rnn(cell, seq, sequence_length=seq_length, dtype=tf.float32)
            return outputs, state

    @staticmethod
    def BiRNN(seq, seq_length=None, name='text'):
        k = args.dim_k
        with tf.variable_scope(f'BiGRU_{name}', reuse=tf.AUTO_REUSE):
            cell_f = tf.nn.rnn_cell.GRUCell(k)
            cell_b = tf.nn.rnn_cell.GRUCell(k)
            # ret: outputs, state
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_f, cell_b, seq, sequence_length=seq_length, dtype=tf.float32)
        return outputs

    @staticmethod
    def Mean(seq, seq_length=None, mask=None, name=None):
        # seq: (None, L, k), seq_length: (None, ), mask: (None, L)
        # ret: (None, k)
        if seq_length is None and mask is None:
            with tf.variable_scope('Mean'):
                return tf.reduce_sum(seq, -2)

        with tf.variable_scope('MaskMean'):
            if mask is None:
                mask = tf.sequence_mask(seq_length, maxlen=tf.shape(seq)[1], dtype=tf.float32)
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, -1)  # (None, L, 1)
            seq = seq * mask
            seq = tf.reduce_sum(seq, -2)  # (None, k)
            seq = seq / (tf.reduce_sum(mask, -2) + eps)
        return seq

    @staticmethod
    def MLP(x, fc, activation, name):
        with tf.variable_scope(f'MLP_{name}'):
            for i in range(len(fc)):
                x = tf.layers.dense(x, fc[i], activation=activation, name=f'dense_{i}')
        return x

    @staticmethod
    def gate(a, b, name):
        with tf.variable_scope(name):
            alpha = tf.layers.dense(tf.concat([a, b], -1), 1, activation=tf.nn.sigmoid, name='gateW')
            ret = alpha * a + (1 - alpha) * b
        return ret

    @staticmethod
    def dot(a, b):
        return tf.reduce_sum(a * b, -1)

    @staticmethod
    def Embedding(node, n, begin_idx=0, k=None, name='node', emb_w=None, activation_regularization=True):
        # node: [BS]
        with tf.variable_scope(f'Emb_{name}'):
            if emb_w is None:
                if k is None:
                    k = args.dim_k
                emb_w = tf.get_variable(
                    name='emb_w',
                    shape=[n, k],
                    # initializer=tf.random_normal_initializer(0.0, 0.05),
                    # regularizer=UTILS.l2_loss(f'emb_{name}'),
                )
            t = tf.gather(emb_w, node)
            mask = tf.greater_equal(node, begin_idx)
            # mask = tf.cast(mask, tf.float32)
            t = t * tf.cast(tf.expand_dims(mask, -1), tf.float32)
            if activation_regularization:
                regularization = UTILS.l2_loss(f'emb_{name}')
                if regularization is not None:
                    tf.losses.add_loss(regularization(t), tf.GraphKeys.REGULARIZATION_LOSSES)
        return t, mask, emb_w

    # @staticmethod
    # def dense(x, k, use_bias=True, name='dense', **kwargs):
    #     init = tf.glorot_normal_initializer()
    #     ret = tf.layers.dense(x, k, use_bias=use_bias, name=name, kernel_initializer=init, **kwargs)
    #     return ret

    @staticmethod
    def save_npy(ar, fn):
        fn = f'{utils.data_dir}/{args.ds}/{fn}'
        np.save(fn, ar)
    @staticmethod
    def load_npy(fn):
        fn = f'{utils.data_dir}/{args.ds}/{fn}'
        return np.load(fn)

class Base:
    args = Object()
    need_train = True
    def __init__(self, data: dataset.Data):
        self.tmp_vars = Object()

        self.data = data

        # self.save_dir = f'{utils.save_dir}/{args.run_name}'
        self.save_dir = f'{utils.save_dir}/{args.msg}'

        with self.data.tf_graph.as_default():
            tf.set_random_seed(args.seed)
            self.compile()
        self.fit_step = 0

    def compile(self):
        self.emb_l2_norm_op = None
        self.sess = UTILS.get_session()
        self.make_io()
        self.make_model()
        self.sess.run(tf.global_variables_initializer())
        if self.emb_l2_norm_op is not None:
            self.sess.run(self.emb_l2_norm_op)

    def make_io(self):
        self.is_on_train = tf.placeholder(tf.bool, [], 'is_on_train')

        train_data = self.data.train_batch_repeat
        train_data_iter = train_data.make_one_shot_iterator()
        self.train_data_handle = self.sess.run(train_data_iter.string_handle())
        self.data_handle = tf.placeholder(tf.string, [], 'data_handle')
        data_iter = tf.data.Iterator.from_string_handle(
            self.data_handle,
            train_data.output_types,
            train_data.output_shapes,
        )
        self.input_dict = data_iter.get_next()
        self.input_dict = Object(**self.input_dict)

    def get_metric_v(self, x, predict_v):
        # [BS,]
        true_item = x.ans
        true_item_a1 = tf.expand_dims(true_item, -1)
        # [BS, M], [BS, 1]
        eq = tf.cast(tf.equal(predict_v.top_items, true_item_a1), tf.int32)
        # [BS,]
        m = tf.reduce_max(eq, -1)
        idx = tf.cast(tf.argmax(eq, -1), tf.int32)
        rank = idx + m - 1
        ndcg = tf.log(2.0) * tf.cast(m, tf.float32) / tf.log(2.0 + tf.cast(idx, tf.float32))
        hit_rate = tf.cast(m, tf.float32)
        ret = Object(
            ndcg=ndcg,
            hit_rate=hit_rate,
            user=x.user,
            true_item=true_item,
            phase=x.phase,
            top_items=predict_v.top_items,
            top_scores=predict_v.top_scores,
            rank=rank,
            q_ts=x.q_ts,
        )
        return ret

    def get_predict_v(self, x, score):
        # [N]
        item_pop = tf.cast(self.data.item_pop, tf.float32)
        item_pop_log = tf.log(item_pop + np.e)

        item_deg_self = tf.cast(tf.gather(self.data.item_deg_self_per_phase, x.phase), tf.float32)
        item_pop_self_log = tf.log(item_deg_self + np.e)

        if args.mode_pop == 'log':
            score = score * args.alpha_pop_base + score / item_pop_log
        elif args.mode_pop == 'log_mdeg':
            score = score * item_pop_self_log / item_pop_log
        elif args.mode_pop == 'log_mdeg_only':
            score = score * item_pop_self_log
        elif args.mode_pop == 'linear':
            item_pop = tf.cast(self.data.item_pop, tf.float32) + 1.0
            score = score * args.alpha_pop_base + score / item_pop
        elif args.mode_pop == 'log_md':
            item_pop_self_log = tf.log(item_deg_self + np.e + 10.0)
            score = score * item_pop_self_log / item_pop_log

        if args.mode_rare in {'log', 'linear', 'base'}:
            if args.mode_rare == 'log':
                rare_weight_pop = 1.0 / tf.log(tf.cast(self.data.item_pop, tf.float32) + np.e)
            elif args.mode_rare == 'linear':
                rare_weight_pop = 1.0 / tf.cast(self.data.item_pop, tf.float32)
            elif args.mode_rare == 'base':
                rare_weight_pop = 0.0
            else:
                raise Exception
            # [N]
            rare_weight = float(args.alpha_rare)
            rare_weight = rare_weight + rare_weight_pop
            rare_weight *= float(args.alpha_rare_mul)

            is_rare = self.is_rare(x)
            score = tf.where(is_rare, score * rare_weight + float(args.alpha_rare_base), score)

        score = UTILS.mask_logits(score, x.score_mask)

        tf.summary.histogram('score', score)
        top_items, top_scores = self.topk_idx(score, x)

        if args.dump_all:
            self.tmp_vars.update(all_scores=score, item_seq=x.seq, ts_seq=x.ts, q_ts=x.q_ts)
            ret = Object(user=x.user, phase=x.phase, top_items=top_items, top_scores=top_scores)
            ret.update(**self.tmp_vars)
            return ret

        return Object(user=x.user, phase=x.phase, top_items=top_items, top_scores=top_scores)

    def make_model(self):
        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE, regularizer=UTILS.l2_loss('all')):
            x = self.input_dict
            self.train_op, self.train_v, self.predict_v = self.forward(x)
            self.metric_v = self.get_metric_v(x, self.predict_v)
            self.metric_v.update(loss=self.train_v.loss)

        network_var_list = tf.trainable_variables(scope='^Network/')
        if network_var_list:
            args.log.log('trainable_variables:')
            for v in network_var_list:
                args.log.log(f'network: {v}')
                tf.summary.histogram(v.name, v)
            self.saver = tf.train.Saver(var_list=tf.trainable_variables())
            # self.saver_emb = tf.train.Saver(var_list=tf.trainable_variables(scope='^Network/Emb_'))

    def fit(self):
        data = {
            self.is_on_train: True,
            self.data_handle: self.train_data_handle,
        }
        tb_v = []
        if args.run_tb:
            tb_v = [self.all_summary]
        debug_v = DEBUG.fit_show_list
        all_v = [self.train_op, self.train_v, debug_v, tb_v]
        _, train_v, debug_v, tb_v = self.sess.run(all_v, data)
        if self.emb_l2_norm_op is not None:
            self.sess.run(self.emb_l2_norm_op)
        if tb_v:
            self.tbfw.add_summary(tb_v[0], self.fit_step)
        DEBUG.when_run(debug_v)
        self.fit_step += 1
        return train_v

    def inference(self, data, out_obj):
        with self.data.tf_graph.as_default():
            data_iter = data.make_one_shot_iterator()
            data_handle = self.sess.run(data_iter.string_handle())
            data = {
                self.is_on_train: False,
                self.data_handle: data_handle,
            }
        while True:
            try:
                ret_value, debug_v = self.sess.run([out_obj, DEBUG.inf_show_list], data)
                DEBUG.when_run(debug_v)
                yield ret_value
            except tf.errors.OutOfRangeError:
                break

    def metric(self, data):
        for v in self.inference(data, self.metric_v):
            yield v

    def predict(self, data):
        for v in self.inference(data, self.predict_v):
            yield v

    def save(self, s):
        if not self.need_train:
            return
        name = f'{self.save_dir}/model_{s}.ckpt'
        self.saver.save(self.sess, name)
    def restore(self, s):
        if not self.need_train:
            return
        name = f'{self.save_dir}/model_{s}.ckpt'
        self.saver.restore(self.sess, name)
    def restore_from_other(self, run_name):
        save_dir = f'{utils.save_dir}/{run_name}'
        s = 0
        if not self.need_train:
            return
        import os
        if not os.path.isdir(save_dir):
            args.log.log('download from hdfs')
            sh = f'$HADOOP_HOME/bin/hadoop fs -get save/{utils.project_name}/{run_name} {utils.save_dir}/'
            print(os.system(sh))
        name = f'{save_dir}/model_{s}.ckpt'
        self.saver.restore(self.sess, name)
        # if args.restore_train:
        #     self.saver_emb.restore(self.sess, name)
        # else:
        #     self.saver.restore(self.sess, name)


    def forward(self, x):
        raise NotImplementedError

    def is_rare(self, x):
        is_rare = tf.gather(self.data.is_rare_per_phase, x.phase)
        return is_rare

    def topk_idx(self, prob, x):
        rare_k = args.nb_rare_k
        if rare_k < 0:
            topk = tf.nn.top_k(prob, args.nb_topk)
            return topk.indices, topk.values

        is_rare = self.is_rare(x)
        prob_rare = UTILS.mask_logits(prob, is_rare)
        prob_rich = UTILS.mask_logits(prob, tf.logical_not(is_rare))

        topk_rare = tf.nn.top_k(prob_rare, rare_k).indices
        topk_rich = tf.nn.top_k(prob_rich, args.nb_topk - rare_k).indices
        topk = tf.concat([topk_rich, topk_rare], -1)
        # [BS, N], [BS, L] --> [BS, L]
        top_prob = tf.batch_gather(prob, topk)
        sort_topk = tf.nn.top_k(top_prob, args.nb_topk)
        sort_idx = sort_topk.indices
        top_values = sort_topk.values

        sorted_topk = tf.batch_gather(topk, sort_idx)
        return sorted_topk, top_values

    def before_train(self):
        pass

    def after_train(self):
        pass


class GNN:
    def __init__(self, neighbors, item_emb_w, is_on_train):
        self.neighbors = neighbors
        self.item_emb_w = item_emb_w
        self.is_on_train = is_on_train
        with tf.variable_scope('GNN') as self.scope:
            with tf.variable_scope('GNN_agg') as self.agg_scope:
                pass

    @staticmethod
    def gather_adj(neighbors, idx):
        # [N, M] + [BS, L] --> [BS, L, M]
        ret = tf.gather(neighbors, idx)
        return ret

    def node_embedding(self, node):
        # node: [BS, M]
        emb, mask, _ = UTILS.Embedding(node, args.nb_items, begin_idx=3, name='item', emb_w=self.item_emb_w)
        return emb, mask

    def node_list_aggregate(self, node_2d, ans, depth=0):
        # node_2d: [BS, M]
        bs = tf.shape(node_2d)[0]
        m = tf.shape(node_2d)[1]
        node = tf.reshape(node_2d, [bs * m])
        if ans is not None:
            # [BS, ] --> [BS*m, ]
            ans_2d = tf.expand_dims(ans, -1)
            zeros_2d = tf.zeros(shape=[bs, m - 1], dtype=tf.int32)
            ans = tf.concat([ans_2d, zeros_2d], -1)
            ans = tf.reshape(ans, [bs * m, 1])
        emb, mask = self.single_node_aggregate(node, ans, depth)
        k = args.dim_k
        emb = tf.reshape(emb, [bs, m, k])
        mask = tf.reshape(mask, [bs, m])
        return emb, mask

    def single_node_aggregate(self, node, ans, depth=0):
        # node: [BS,]
        if depth > args.gnn_depth:
            raise Exception
        if depth == args.gnn_depth:
            emb, mask = self.node_embedding(node)
            return emb, mask
        with tf.variable_scope(f'GNN_depth_{depth}'):
            cur, cur_mask = self.single_node_aggregate(node, None, depth + 1)

            nxt_list = []
            for i in range(2):
                nxt = self.gather_adj(self.neighbors[i], node)
                if ans is not None and depth == 0 and i == 0:
                    # nxt: [BS*L, N], ans: [BS*L, 1]
                    zeros = tf.zeros_like(nxt, dtype=tf.int32)
                    # DEBUG.print_when_fit(nxt)
                    nxt = tf.where(tf.equal(nxt, ans), zeros, nxt)
                    # DEBUG.print_when_fit(nxt, block=True)
                nxt, nxt_mask = self.node_list_aggregate(nxt, None, depth + 1)
                nxt_list.append([nxt, nxt_mask])

            h = self.aggregate2(cur, nxt_list)  # [BS, L, k]
            return h, cur_mask

    def aggregate2(self, cur, nxt_list):
        # [BS, k], [ [nxt, mask], ...]
        with tf.variable_scope(self.agg_scope):
            nxt_forward = self.agg(cur, *nxt_list[0], 'agg_forward')
            nxt_backward = self.agg(cur, *nxt_list[0], 'agg_backward')
            nxt = nxt_forward + nxt_backward
        ret = cur + nxt
        if args.gnn_use_bn:
            ret = tf.layers.batch_normalization(ret, name='BN', training=self.is_on_train)
        return ret

    def agg(self, cur, nxt, mask, name):
        # [BS, k], [BS, M, k], [BS, M]
        ret = UTILS.Mean(nxt, mask=mask)
        return ret



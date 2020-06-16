from .base import Base
from .Emb import Emb, MUE
from .base import UTILS
from .base import DEBUG
import tensorflow as tf
from utils import args
import numpy as np


class ATT(MUE):
    def dist_weight(self, x, seq_mask):
        with tf.variable_scope('pos_enc'):
            maxlen = args.seq_length
            emb_w = tf.get_variable(
                name='dist_att_w',
                shape=[maxlen, ],
                initializer=tf.zeros_initializer(),
            )
            t = emb_w

        w = UTILS.mask_logits(t, seq_mask)
        w = tf.nn.softmax(w, -1)
        return w

    def seg_ts(self, ts):
        ts_unit = 0.05
        ts = ts / ts_unit + 1.0
        ts_seg = tf.log(ts) / np.log(10.0) * 10.0
        ts_seg = tf.cast(ts_seg, tf.int32)
        return ts_seg

    def ts_diff_weight(self, x, seq_mask):
        # [BS, L]
        ts_diff = tf.expand_dims(x.q_ts, -1) - x.ts
        ts_seg = self.seg_ts(ts_diff)
        maxlen = 40
        ts_seg = tf.where(tf.less(ts_seg, maxlen), ts_seg, tf.ones_like(ts_seg, dtype=tf.int32) * (maxlen - 1))

        with tf.variable_scope('ts_enc'):
            emb_w = tf.get_variable(
                name='ts_diff_att_w',
                shape=[maxlen, ],
                initializer=tf.zeros_initializer(),
            )
            self.tmp_vars.update(att_w=emb_w)
            t = tf.gather(emb_w, ts_seg)

        # DEBUG.print_when_inference(emb_w, block=True)
        w = UTILS.mask_logits(t, seq_mask)
        w = tf.nn.softmax(w, -1)
        # w = tf.exp(w)
        # DEBUG.print_when_inference(w, block=True)
        return w

    def get_att_weight(self, x, seq_mask):
        raise NotImplementedError

    def att(self, x, seq, seq_mask):
        w = self.get_att_weight(x, seq_mask)
        # [BS, L, k] x [BS, L] --> [BS, k]
        att_out = tf.reduce_sum(seq * tf.expand_dims(w, -1), -2)
        return att_out

    def get_out_rep(self, x):
        seq_emb, seq_mask = self.item_embedding(x.seq)
        user = self.get_user_emb(x)
        att = self.att(x, seq_emb, seq_mask)
        ret = user + att

        # [BS, L + 1, k]
        bs = tf.shape(user)[0]
        L = args.seq_length
        k = args.dim_k
        N = args.nb_items
        tmp = tf.concat([tf.expand_dims(user, 1), seq_emb], 1)
        tmp = tf.reshape(tmp, [bs * (L + 1), k])
        # [BS * (L + 1), N]
        sim = tf.matmul(tmp, self.item_emb_matrix, transpose_b=True)
        sim = tf.reshape(sim, [bs, L + 1, N])
        self.tmp_vars.update(sim=sim)

        return ret


class ATT_ts(ATT):
    def get_att_weight(self, x, seq_mask):
        return self.ts_diff_weight(x, seq_mask)

class ATT_pos(ATT):
    def get_att_weight(self, x, seq_mask):
        return self.dist_weight(x, seq_mask)

class ATT_ts_pos(ATT):
    def get_att_weight(self, x, seq_mask):
        return self.ts_diff_weight(x, seq_mask) + self.dist_weight(x, seq_mask)

class ATT_ts_m_pos(ATT):
    def get_att_weight(self, x, seq_mask):
        return self.ts_diff_weight(x, seq_mask) * self.dist_weight(x, seq_mask)

class ATT_ts_E(ATT_ts):
    def get_out_rep(self, x):
        seq_emb, seq_mask = self.item_embedding(x.seq)
        emb = seq_emb[:, 0, :]
        user = self.get_user_emb(x)
        att = self.att(x, seq_emb, seq_mask)
        ret = user + att + emb
        return ret

class ATT_ts_G(ATT_ts):
    def gate(self, x):
        ts_diff = x.q_ts - x.ts[:, 0]
        ts_seg = self.seg_ts(ts_diff)
        maxlen = 40
        ts_seg = tf.where(tf.less(ts_seg, maxlen), ts_seg, tf.ones_like(ts_seg, dtype=tf.int32) * (maxlen - 1))

        emb_w = tf.get_variable(
            name='ts_diff_get_w',
            shape=[maxlen, ],
            initializer=tf.zeros_initializer(),
        )
        t = tf.gather(emb_w, ts_seg)



    def get_out_rep(self, x):
        seq_emb, seq_mask = self.item_embedding(x.seq)
        emb = seq_emb[:, 0, :]
        user = self.get_user_emb(x)
        att = self.att(x, seq_emb, seq_mask)
        ret = user + att + emb
        return ret



# TODO: 时间段 loss 权重
# TODO: 时间差 attention


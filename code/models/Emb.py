import tensorflow as tf
import numpy as np
from utils import args
from utils import Object
from . import Base
from . import UTILS

class Emb(Base):
    def sample_neg(self, ans, pred_score):
        BS = tf.shape(ans)[0]
        def sample(minval, maxval):
            _neg = tf.random_uniform(
                shape=(BS, args.nb_neg),
                minval=minval,
                maxval=maxval,
                dtype=tf.int32,
                name='sample_neg',
            )
            # DEBUG.print_when_fit(_neg, block=True)
            return _neg

        if args.mode_neg_sample == 'all':
            neg = sample(3, args.nb_items)
        else:
            if args.mode_neg_sample == 'all2':
                base_score = 0.0
            elif args.mode_neg_sample == 'pred':
                base_score = tf.cast(self.data.dp.score_mask, tf.float32)
            elif args.mode_neg_sample == 'rare':
                base_score = tf.cast(self.data.is_rare, tf.float32)
            elif args.mode_neg_sample == 'not_rare':
                base_score = tf.cast(tf.logical_not(self.data.is_rare), tf.float32)
            elif args.mode_neg_sample == 'pop':
                base_score = tf.cast(self.data.item_pop, tf.float32)
                base_score = tf.log(base_score + 1.0)
            elif args.mode_neg_sample == 'deg':
                base_score = tf.cast(self.data.item_deg, tf.float32)
                base_score = tf.log(base_score + 1.0)
            elif args.mode_neg_sample == 'deg_self':
                base_score = tf.cast(self.data.item_deg_self, tf.float32)
                base_score = tf.log(base_score + 1.0)
            elif args.mode_neg_sample == 'pred_top':
                base_score = pred_score
                base_score = tf.nn.sigmoid(base_score)
            else:
                raise Exception
            random_score = tf.random_uniform(shape=tf.shape(pred_score), minval=0.0, maxval=args.alpha_neg_sample, dtype=tf.float32)
            neg = tf.nn.top_k(base_score + random_score, args.nb_neg).indices
        pos = tf.expand_dims(ans, -1)
        # neg: [BS, nb_neg]
        # pos: [BS, 1]
        neg = tf.where(tf.equal(neg, pos), tf.zeros_like(neg, dtype=tf.int32), neg)
        return neg

    def float201(self, f):
        m = tf.reduce_max(f, -1)
        return f / m

    def non_linear_feat_emb(self, feat):
        # [N,]
        feat = tf.cast(feat, tf.float32)
        f1 = self.float201(feat)
        f2 = self.float201(tf.log(feat))
        f3 = self.float201(tf.sqrt(feat))
        f = tf.stack([f1, f2, f3], axis=-1)
        return f

    def _get_item_emb_matrix(self):
        if hasattr(self, 'item_emb_matrix'):
            return self.item_emb_matrix
        self.item_emb_matrix = self.get_item_emb_matrix()
        return self.item_emb_matrix

    def _get_user_emb_matrix(self):
        if hasattr(self, 'user_emb_matrix'):
            return self.user_emb_matrix
        self.user_emb_matrix = self.get_user_emb_matrix()
        return self.user_emb_matrix

    def get_item_emb_matrix(self):
        mode = args.mode_item_emb_init
        if mode == 'txt':
            assert args.dim_k == 128
            init = self.data.item_feat[:, 0, :]
        elif mode == 'img':
            assert args.dim_k == 128
            init = self.data.item_feat[:, 1, :]
        elif mode == 'all':
            assert args.dim_k == 256
            init = tf.reshape(self.data.item_feat, [args.nb_items, 256])
        else:
            assert mode == 'none'
            init = None

        if init is None:
            raw_id_emb = tf.get_variable('item_emb_w', shape=[args.nb_items, args.dim_k])
        else:
            raw_id_emb = tf.get_variable('item_emb_w', initializer=init)

        return raw_id_emb

    def get_user_emb_matrix(self):
        raw_id_emb = tf.get_variable('user_emb_w', shape=[args.nb_users, args.dim_k])
        return raw_id_emb

    def item_embedding(self, item_id):
        emb_w = self._get_item_emb_matrix()
        t = tf.gather(emb_w, item_id)
        mask = tf.greater_equal(item_id, 3)
        t = t * tf.cast(tf.expand_dims(mask, -1), tf.float32)
        return t, mask

    def user_embedding(self, user_id):
        emb_w = self._get_user_emb_matrix()
        t = tf.gather(emb_w, user_id)
        return t

    # def node_embedding(self, node_id):
    #     item_emb_w = self._get_item_emb_matrix()
    #     user_emb_w = self._get_user_emb_matrix()
    #     emb_w = tf.concat([item_emb_w, user_emb_w], -1)
    #     t = tf.gather(emb_w, node_id)
    #     mask = tf.greater_equal(node_id, 3)
    #     t = t * tf.cast(tf.expand_dims(mask, -1), tf.float32)
    #     return t, mask

    def merge_pos_neg_for_softmax(self, ans, neg):
        # [BS,], [BS, N] --> [BS, N+1], [BS, N+1]
        ans_a1 = tf.expand_dims(ans, -1)
        out = tf.concat([ans_a1, neg], -1)
        out_label = tf.concat([tf.ones_like(ans_a1), tf.zeros_like(neg)], -1)
        return out, out_label

    def get_out_rep(self, x):
        # [BS, L, K]
        seq_emb, seq_mask = self.item_embedding(x.seq)
        return seq_emb[:, 0, :]


    def calculate_score_for_train(self, x, pred_emb, item_ids, activation_regularization=True):
        # [BS, k], [BS, N, k]
        # item_emb, _, _ = UTILS.Embedding(item_ids, args.nb_items, begin_idx=3, name='item')
        item_emb, _ = self.item_embedding(item_ids)
        if args.mode_pred == 'cosine':
            pred_emb = tf.nn.l2_normalize(pred_emb, -1)
            item_emb = tf.nn.l2_normalize(item_emb, -1)

        pred_emb = tf.expand_dims(pred_emb, -2)
        score = tf.reduce_sum(pred_emb * item_emb, -1)

        if args.mode_pop_bias != 'none':

            if args.mode_pop_bias == 'add':
                pop_bias_w = tf.get_variable('pop_bias', shape=[args.nb_items], dtype=tf.float32,
                                             initializer=tf.zeros_initializer())
                pop_bias = tf.gather(pop_bias_w, item_ids)
                score += pop_bias
            elif args.mode_pop_bias == 'mul':
                pop_bias_w = tf.get_variable('pop_bias', shape=[args.nb_items], dtype=tf.float32,
                                             initializer=tf.ones_initializer())
                pop_bias = tf.gather(pop_bias_w, item_ids)
                score *= pop_bias
            else:
                raise Exception

            if activation_regularization:
                regularization = UTILS.l2_loss('bias')
                if regularization is not None:
                    tf.losses.add_loss(regularization(pop_bias), tf.GraphKeys.REGULARIZATION_LOSSES)

        score = score / args.alpha_score_div
        return score

    def calculate_score_for_pred(self, x, pred_emb, item_emb):
        # [BS, k], [N, k]
        # if args.mode_pred == 'cosine' and args.use_cosine_for_pred:
        if args.use_cosine_for_pred:
            pred_emb = tf.nn.l2_normalize(pred_emb, -1)
            item_emb = tf.nn.l2_normalize(item_emb, -1)

        score = tf.matmul(pred_emb, item_emb, transpose_b=True)

        if args.mode_pop_bias != 'none' and args.use_pop_bias_for_pred:
            if args.mode_pop_bias == 'add':
                pop_bias_w = tf.get_variable('pop_bias', shape=[args.nb_items], dtype=tf.float32,
                                             initializer=tf.zeros_initializer())
                score += pop_bias_w
            elif args.mode_pop_bias == 'mul':
                pop_bias_w = tf.get_variable('pop_bias', shape=[args.nb_items], dtype=tf.float32,
                                             initializer=tf.ones_initializer())
                score *= pop_bias_w
            else:
                raise Exception

        return score

    def is_current_phase(self, x):
        unit = 54.5
        min_ts = tf.cast(x.phase, tf.float32) * unit
        max_ts = min_ts + (unit * 4) + 1
        return tf.logical_and(tf.greater_equal(x.q_ts, min_ts), tf.less_equal(x.q_ts, max_ts))

    def is_current_day(self, x):
        unit = 54.5
        day = tf.cast(x.phase + 3, tf.float32)
        min_ts = 10.0 + day * unit
        max_ts = min_ts + unit + 1
        return tf.logical_and(tf.greater_equal(x.q_ts, min_ts), tf.less_equal(x.q_ts, max_ts))

    def is_long_term(self, x):
        ts_diff = x.q_ts - x.ts[:, 0]
        return tf.greater_equal(ts_diff, 3.0)

    def get_loss_weight(self, x):
        mode = args.mode_loss_weight
        alpha = args.alpha_loss_weight
        if mode == 'none':
            weights = 1.0
        elif mode == 'pop':
            item_pop = tf.gather(self.data.item_pop, x.ans)
            weights = 1.0 / tf.log(tf.cast(item_pop, tf.float32) + np.e + alpha)
        elif mode == 'ts':
            ts_seg = tf.cast(x.q_ts / args.ts_unit, tf.int32)
            weights = tf.gather(self.data.ts_weight, ts_seg)
        elif mode == 'rare':
            is_rare = self.is_rare(x)
            is_rare = tf.batch_gather(is_rare, tf.expand_dims(x.ans, -1))
            is_rare = tf.squeeze(is_rare, -1)
            weights = tf.cast(is_rare, tf.float32) * (alpha - 1.0) + 1.0
        elif mode == 'phase':
            cond = self.is_current_phase(x)
            weights = tf.cast(cond, tf.float32) * (alpha - 1.0) + 1.0
        elif mode == 'day':
            cond = self.is_current_day(x)
            weights = tf.cast(cond, tf.float32) * (alpha - 1.0) + 1.0
        else:
            raise Exception(mode)

        return weights

    def get_loss_softmax(self, x, seq_rep, neg):
        # [BS, N]
        sample, sample_label = self.merge_pos_neg_for_softmax(x.ans, neg)
        sample_logits = self.calculate_score_for_train(x, seq_rep, sample)
        weights = self.get_loss_weight(x)
        loss = tf.losses.softmax_cross_entropy(sample_label, sample_logits, weights=weights)
        return loss

    def get_loss_pairwise(self, x, seq_rep, neg):
        pos = tf.expand_dims(x.ans, -1)
        pos_logits = self.calculate_score_for_train(x, seq_rep, pos)
        neg_logits = self.calculate_score_for_train(x, seq_rep, neg)
        # [BS, N]
        diff_logits = neg_logits - pos_logits

        # loss = tf.nn.relu(diff_logits + args.alpha_diff)
        # loss = tf.reduce_sum(loss, -1)

        if args.mode_loss_weight == 'warp':
            item_rank = tf.reduce_sum(tf.cast(tf.greater(diff_logits + args.alpha_diff, 0.0), tf.int32), -1)
            weights = tf.gather(self.data.thjs, item_rank)
        else:
            weights = self.get_loss_weight(x)
        weights = tf.expand_dims(weights, -1)
        loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(diff_logits), diff_logits, weights=weights)
        # loss = tf.reduce_sum(loss * weights)
        return loss

    def forward(self, x):
        # user: [BS,], seq: [BS, L], ts: [BS, L]

        # [BS, k]
        seq_rep = self.get_out_rep(x)
        if args.use_bl:
            seq_rep = tf.layers.dense(seq_rep, args.dim_k, name='bl', kernel_regularizer=UTILS.l2_loss('bl'), use_bias=False)
        dropout = tf.where(self.is_on_train, args.dropout, 1.0)
        seq_rep = tf.nn.dropout(seq_rep, dropout)

        score = self.calculate_score_for_pred(x, seq_rep, self.item_emb_matrix)

        # [BS, M]
        neg = self.sample_neg(x.ans, score)
        if args.mode_loss == 'softmax':
            loss = self.get_loss_softmax(x, seq_rep, neg)
        elif args.mode_loss == 'pairwise':
            loss = self.get_loss_pairwise(x, seq_rep, neg)
        else:
            a = float(args.mode_loss)
            loss_1 = self.get_loss_softmax(x, seq_rep, neg)
            loss_2 = self.get_loss_pairwise(x, seq_rep, neg)
            loss = loss_1 * a + loss_2 * (1.0 - a)


        tf.summary.scalar('loss', loss)
        train_op = UTILS.minimizer(loss)
        train_v = Object(loss=loss)

        return train_op, train_v, self.get_predict_v(x, score)

class UserEmb(Emb):
    def get_user_emb(self, x):
        # user, _, _ = UTILS.Embedding(x.user, args.nb_users, begin_idx=0, name='user')
        user = self.user_embedding(x.user)
        user_dropout = tf.where(self.is_on_train, args.user_dp, 1.0)
        user = tf.nn.dropout(user, user_dropout)
        user = user * args.user_rate
        return user

    def get_out_rep(self, x):
        seq_emb, seq_mask = self.item_embedding(x.seq)
        return self.get_user_emb(x)

class LastItem(Emb):
    args = Emb.args.cp_update(seq_length=1)
    def get_out_rep(self, x):
        seq_emb, seq_mask = self.item_embedding(x.seq)
        emb = seq_emb[:, 0, :]
        return emb

class SeqMean(Emb):
    def get_out_rep(self, x):
        seq_emb, seq_mask = self.item_embedding(x.seq)
        mean = UTILS.Mean(seq_emb, mask=seq_mask)
        return mean

class MUE(UserEmb):
    def get_out_rep(self, x):
        seq_emb, seq_mask = self.item_embedding(x.seq)
        emb = seq_emb[:, 0, :]
        user = self.get_user_emb(x)
        mean = UTILS.Mean(seq_emb, mask=seq_mask)
        ret = mean + user + emb
        return ret

class MUE_LT(MUE):
    def get_out_rep(self, x):
        seq_emb, seq_mask = self.item_embedding(x.seq)
        emb = seq_emb[:, 0, :]
        user = self.get_user_emb(x)
        mean = UTILS.Mean(seq_emb, mask=seq_mask)
        ret_short = mean + user + emb
        ret_long = mean + user
        ret = tf.where(self.is_long_term(x), ret_long, ret_short)
        return ret

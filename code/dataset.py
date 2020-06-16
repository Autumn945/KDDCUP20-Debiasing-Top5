import numpy as np
import sys
import os
import json
import tensorflow as tf

from utils import args, tqdm
import utils
from collections import defaultdict

data_home = utils.data_dir

version = 'v3'
NUM_PHASE = 10


class DataProcess:
    def load_json(self, fn):
        with open(f'{self.home}/{fn}.json', 'r') as f:
            return json.load(f)

    def __init__(self):
        self.home = f'{data_home}/{args.ds}'
        metadata = self.load_json('metadata')
        args.update(**metadata)

        tmp = [6, 11, 15, 18, 22, 24, 26, 29, 32, 33]
        if args.ds == 'v8':
            tmp = tmp[:7]
            global NUM_PHASE
            NUM_PHASE = 7
        # tmp = [t + 2 for t in tmp]
        metadata['mid_deg_per_phase'] = tmp

        self.mid_deg_per_phase = metadata['mid_deg_per_phase']
        args.update(mid_deg_per_phase=str(self.mid_deg_per_phase))

        self.user2item_seq = self.load_json('user2item_seq')
        self.user2ts_seq = self.load_json('user2ts_seq')
        self.vali_puiqa = self.load_json('vali_puiqa')
        self.test_puiqa = self.load_json('test_puiqa')
        # uids_list, vids_list
        self.raw_id_list = self.load_json('ids_list')

        try:
            self.item_feat = np.load(f'{self.home}/item_feat.npy')
        except Exception as e:
            print(e)
            self.item_feat = np.zeros([args.nb_items, 2, 128])

        self.item_deg_per_phase = self.load_json('item_deg')
        self.item_deg_per_phase = np.array(self.item_deg_per_phase, dtype=int)
        self.item_deg_self_per_phase = self.load_json('item_deg_self')
        self.item_deg_self_per_phase = np.array(self.item_deg_self_per_phase, dtype=int)

        self.item_is_half = self.item_deg_per_phase <= np.array(self.mid_deg_per_phase)[:, None]

        self.score_mask_per_phase = np.ones([NUM_PHASE, args.nb_items], dtype=int)
        self.score_mask_per_phase[self.item_deg_self_per_phase == 0] = 0

        self.vali_user2ppa = {}
        for phase, user, pos, ts, ans in self.vali_puiqa:
            self.vali_user2ppa[user] = (phase, pos, ans)

        self.test_user2ppa = {}
        for phase, user, pos, ts, ans in self.test_puiqa:
            self.test_user2ppa[user] = (phase, pos, ans)

        self.train_users = list(range(args.nb_users))
        self.vali_users = sorted(self.vali_user2ppa.keys())
        self.test_users = sorted(self.test_user2ppa.keys())

        self.item_pop = [0] * args.nb_items
        self.user_pop = [0] * args.nb_users

        self.for_phase = -1
        if len(args.mode_pred_phase) == 1:
            self.for_phase = int(args.mode_pred_phase)

        args.update(for_phase=self.for_phase)

        nb_train = 0
        for user, item_list in enumerate(self.user2item_seq):
            for i, item in enumerate(item_list):
                if item == 1:
                    vali_phase, vali_pos, vali_ans = self.vali_user2ppa[user]
                    if args.use_unused_vali and args.mode_pred_phase != 'all' and str(vali_phase) not in args.mode_pred_phase:
                        item = vali_ans

                if item >= 3:
                    self.item_pop[item] += 1
                    self.user_pop[user] += 1
                    nb_train += 1
                    if args.mode_resample != 'none':
                        if args.mode_resample == 'user':
                            if user in self.vali_user2ppa or user in self.test_user2ppa:
                                nb_train += args.alpha_resample
                        elif args.mode_resample == 'rare':
                            # assert len(args.mode_pred_phase) == 1
                            # assert phase == int(args.mode_pred_phase)
                            phase = int(args.mode_pred_phase)
                            item_deg = self.item_deg_per_phase[phase][item]
                            mid_deg = self.mid_deg_per_phase[phase]
                            if item_deg <= mid_deg:
                                nb_train += args.alpha_resample
                        elif args.mode_resample == 'phase':
                            phase = int(args.mode_pred_phase)
                            q_ts = self.user2ts_seq[user][i]
                            unit = 54.5
                            min_ts = phase * unit
                            max_ts = min_ts + (unit * 4) + 1
                            if min_ts <= q_ts < max_ts:
                                nb_train += args.alpha_resample
                        elif args.mode_resample == 'day':
                            phase = int(args.mode_pred_phase)
                            q_ts = self.user2ts_seq[user][i]
                            unit = 54.5
                            min_ts = 10.0 + (phase + 3) * unit
                            max_ts = min_ts + unit + 1
                            if min_ts <= q_ts < max_ts:
                                nb_train += args.alpha_resample

        for item in range(3):
            self.item_pop[item] += 1

        self.item_pop = np.array(self.item_pop, dtype=int)
        self.item_pop[self.item_pop == 0] = 1
        assert np.min(self.item_pop) >= 1
        self.item_pop_log = np.log(self.item_pop + np.e)

        self.item_pop_inv = 1.0 / self.item_pop
        self.item_pop_log_inv = 1.0 / self.item_pop_log

        self.user_pop = np.array(self.user_pop, dtype=int)
        args.update(nb_train=nb_train)

        if args.gnn:
            pass
        if args.get('run_data', False) and args.run_test:
            pass
            # self.run_test()
            # input()

    def run_test(self):
        pass


    def build_graph_item_item(self):
        from tqdm import tqdm
        G_forward = [defaultdict(int) for _ in range(args.nb_items)]
        G_backward = [defaultdict(int) for _ in range(args.nb_items)]
        nb_edges = 0
        for u, item_list in tqdm(enumerate(self.user2item_seq), desc='build edges'):
            n = len(item_list)
            for i in range(1, n):
                a, b = item_list[i - 1], item_list[i]
                if a >= 3 and b >= 3:
                    G_forward[a][b] += 1
                    G_backward[b][a] += 1
                    if G_forward[a][b] == args.gnn_min_edge_cnt:
                        nb_edges += 1
        args.update(nb_edges=nb_edges)
        neighbors = [[], []]
        maxn = args.gnn_adj_length
        for item in tqdm(range(args.nb_items), desc='sample neighbors'):
            nxt_forward = self.sample_neighbors(G_forward[item], maxn)
            nxt_backward = self.sample_neighbors(G_backward[item], maxn)
            neighbors[0].append(nxt_forward)
            neighbors[1].append(nxt_backward)
        self.neighbors = neighbors

    def build_graph_user_item(self):
        from tqdm import tqdm
        user2item = [defaultdict(int) for _ in range(args.nb_users)]
        item2user = [defaultdict(int) for _ in range(args.nb_items)]
        for user, item_list in tqdm(enumerate(self.user2item_seq), desc='build edges'):
            for item in item_list:
                if item >= 3:
                    user2item[user][item] += 1
                    item2user[item][user] += 1

        maxn = args.gnn_adj_length
        user_neighbors = []
        for user in range(args.nb_users):
            items = self.sample_neighbors(user2item[user], maxn)
            user_neighbors.append(items)

        item_neighbors = []
        for item in range(args.nb_items):
            users = self.sample_neighbors(item2user[item], maxn)
            item_neighbors.append(users)

        self.neighbors = [user_neighbors, item_neighbors]

    def sample_neighbors(self, d, maxn):
        nxt_w = sorted(d.items(), key=lambda x: (-x[1], x[0]))
        nxt_node = [nxt for nxt, w in nxt_w if w >= args.gnn_min_edge_cnt]
        if len(nxt_node) < maxn:
            nxt_node = nxt_node + [0] * (maxn - len(nxt_node))
        else:
            nxt_node = nxt_node[:maxn]
        return nxt_node

class Data:
    def __init__(self):
        self.dp = DataProcess()
        self.tf_graph = tf.Graph()
        self.load_data()
        with self.tf_graph.as_default():
            self.make_tf_data()

    def load_data(self):
        self.raw_ids = utils.Object(**self.dp.raw_id_list)
        self.tv_cache = {}


    def make_tf_data(self):
        self.train = self.get_dataset(self.dp.train_users, 'train').prefetch(args.batch_size * 2)
        self.train_batch_single = Data.padded_batch(self.train, batch_size=1)
        if args.run_test:
            buffer_size = 10
        else:
            buffer_size = int(1e4)
        # buffer_size = min(args.nb_train, args.batch_size * 100)
        # self.train_batch = Data.padded_batch(self.train.shuffle(buffer_size=buffer_size, seed=345), args.batch_size)
        self.train_batch = Data.padded_batch(self.train.shuffle(buffer_size=buffer_size, seed=args.seed), args.batch_size)
        self.train_batch_repeat = self.train_batch.repeat(None)

        self.vali = self.get_dataset(self.dp.vali_users, 'vali')
        self.vali_batch = Data.padded_batch(self.vali, args.batch_size)

        self.test = self.get_dataset(self.dp.test_users, 'test')
        self.test_batch = Data.padded_batch(self.test, args.batch_size)

        # item_deg_per_phase
        # mid_deg_per_phase
        # item_deg_self_per_phase
        # item_pop

        # [N, ]
        self.item_pop = tf.constant(self.dp.item_pop, dtype=tf.int32)
        # [N, 2, 128]
        self.item_feat = tf.constant(self.dp.item_feat, dtype=tf.float32)

        self.item_deg_per_phase = tf.constant(self.dp.item_deg_per_phase, dtype=tf.int32)
        self.mid_deg_per_phase = tf.constant(self.dp.mid_deg_per_phase, dtype=tf.int32)

        self.is_rare_per_phase = tf.less_equal(self.item_deg_per_phase, tf.expand_dims(self.mid_deg_per_phase, -1))

        self.item_deg_self_per_phase = tf.constant(self.dp.item_deg_self_per_phase, dtype=tf.int32)

        ds = 1.0 / np.arange(1, 1000)
        self.thjs = np.cumsum(ds)
        self.thjs = tf.constant(self.thjs, dtype=tf.float32)
        # print(self.thjs[:30])

        if args.gnn:
            self.neighbors = tf.constant(self.dp.neighbors, dtype=tf.int32)

    def get_score_mask(self, phase, user, ans):
        mask = self.dp.score_mask_per_phase[phase].copy()
        item_seq = self.dp.user2item_seq[user]
        mask[item_seq] = 0
        mask[ans] = 1
        return mask

    def get_input_seq(self, user, pred_pos, name):
        if name in ('train', 'vali') and (user, pred_pos) in self.tv_cache:
            return self.tv_cache[(user, pred_pos)]
        item_seq = self.dp.user2item_seq[user]
        ts_seq = self.dp.user2ts_seq[user]
        if name == 'train':
            ans = item_seq[pred_pos]
            if ans == 1:
                vali_phase, vali_pos, vali_ans = self.dp.vali_user2ppa[user]
                if args.use_unused_vali and args.mode_pred_phase != 'all' and str(vali_phase) not in args.mode_pred_phase:
                    ans = vali_ans

            if ans < 3:
                return None
        elif name == 'vali':
            phase, vali_pos, ans = self.dp.vali_user2ppa[user]
            assert vali_pos == pred_pos
            assert item_seq[vali_pos] == 1
        elif name == 'test':
            phase, test_pos, ans = self.dp.test_user2ppa[user]
            assert test_pos == pred_pos
            assert item_seq[test_pos] == 2
            if user in self.dp.vali_user2ppa:
                vali_phase, vali_pos, vali_ans = self.dp.vali_user2ppa[user]
                item_seq = list(item_seq)
                item_seq[vali_pos] = vali_ans
        else:
            raise Exception

        q_ts = ts_seq[pred_pos]
        pre_ts = ts_seq[pred_pos - 1] if pred_pos else -100

        assert args.data_dt_less_than < 0 or args.data_dt_greater_than < 0
        # noinspection PyChainedComparisons
        if args.data_dt_less_than > 0 and name != 'train' and not q_ts - pre_ts < args.data_dt_less_than:
            return None

        if args.data_dt_greater_than > 0 and name != 'train' and not q_ts - pre_ts >= args.data_dt_greater_than:
            return None

        _item_seq, _ts_seq = [], []
        # _item_seq = []
        for i in range(pred_pos)[::-1]:
            item, ts = item_seq[i], ts_seq[i]
            # item = item_seq[i]
            # if _item_seq and ts_seq[i + 1] - ts > args.seq_max_dt:
            if q_ts - ts > args.seq_max_dt:
                break

            if item == 1:
                vali_phase, vali_pos, vali_ans = self.dp.vali_user2ppa[user]
                if args.use_unused_vali and args.mode_pred_phase != 'all' and str(vali_phase) not in args.mode_pred_phase:
                    item = vali_ans

            if item >= 3 and item != ans:
                _item_seq.append(item)
                _ts_seq.append(ts)

            if len(_item_seq) >= args.seq_length:
                break

        if not _item_seq:
            _item_seq, _ts_seq = [0], [-100]

        out = utils.Object(
            seq=_item_seq,
            ts=_ts_seq,
        )
        self.tv_cache[(user, pred_pos)] = out
        return out

    def gen_data_from_user(self, user, name):
        item_seq = self.dp.user2item_seq[user]
        ts_seq = self.dp.user2ts_seq[user]
        if name == 'train':
            score_mask = np.ones(1, dtype=float)
            phase = 5
            if len(args.mode_pred_phase) == 1:
                phase = int(args.mode_pred_phase)

            for pred_pos in range(1, len(item_seq)):
                ans = item_seq[pred_pos]
                q_ts = ts_seq[pred_pos]
                out = self.get_input_seq(user, pred_pos, name)
                if out is None:
                    continue
                out = out.cp_update(
                    user=user,
                    q_ts=q_ts,
                    ans=ans,
                    phase=phase,
                    score_mask=score_mask,
                )
                yield out
                if args.mode_resample != 'none':
                    # assert len(args.mode_pred_phase) == 1
                    # assert phase == int(args.mode_pred_phase)
                    resample_n = 0
                    if args.mode_resample == 'user':
                        if user in self.dp.vali_user2ppa or user in self.dp.test_user2ppa:
                            resample_n = args.alpha_resample
                    elif args.mode_resample == 'rare':
                        item_deg = self.dp.item_deg_per_phase[phase][ans]
                        mid_deg = self.dp.mid_deg_per_phase[phase]
                        if item_deg <= mid_deg:
                            resample_n = args.alpha_resample
                    elif args.mode_resample == 'phase':
                        unit = 54.5
                        min_ts = phase * unit
                        max_ts = min_ts + (unit * 4) + 1
                        if min_ts <= q_ts < max_ts:
                            resample_n = args.alpha_resample
                    elif args.mode_resample == 'day':
                        unit = 54.5
                        min_ts = 10.0 + (phase + 3) * unit
                        max_ts = min_ts + unit + 1
                        if min_ts <= q_ts < max_ts:
                            resample_n = args.alpha_resample

                    for _ in range(resample_n):
                        yield out

        else:
            if name == 'vali':
                phase, pred_pos, ans = self.dp.vali_user2ppa[user]
            elif name == 'test':
                phase, pred_pos, ans = self.dp.test_user2ppa[user]
            else:
                raise Exception

            if args.mode_pred_phase != 'all' and str(phase) not in args.mode_pred_phase:
                return

            out = self.get_input_seq(user, pred_pos, name)
            if out is None:
                return

            q_ts = ts_seq[pred_pos]
            score_mask = self.get_score_mask(phase, user, ans)
            out = out.cp_update(
                user=user,
                q_ts=q_ts,
                ans=ans,
                phase=phase,
                score_mask=score_mask,
            )
            yield out

    def gen_data_for_debug(self):
        from tqdm import tqdm
        for user in tqdm(self.dp.train_users):
            d = self.gen_data_from_user(user, 'train')
            yield from d

    def run_test(self):
        for x in self.gen_data_for_debug():
            pass

    def get_dataset(self, users, name):
        # rdm = np.random.RandomState(428)
        def gen():
            for user in users:
                d = self.gen_data_from_user(user, name)
                yield from d

        # TODO speed up !!!

        output_types = dict(
            user=tf.int32,
            seq=tf.int32,
            ts=tf.float32,
            # future_seq=tf.int32,
            # future_ts=tf.float32,
            q_ts=tf.float32,
            ans = tf.int32,
            phase = tf.int32,
            score_mask=tf.int32,
        )
        output_shapes = dict(
            user=tf.TensorShape([]),
            seq=tf.TensorShape([None]),
            ts=tf.TensorShape([None]),
            # future_seq=tf.TensorShape([None]),
            # future_ts=tf.TensorShape([None]),
            q_ts=tf.TensorShape([]),
            ans=tf.TensorShape([]),
            phase=tf.TensorShape([]),
            score_mask=tf.TensorShape([None]),
        )
        dataset = tf.data.Dataset.from_generator(
            gen,
            output_types=output_types,
            output_shapes=output_shapes,
        )
        return dataset

    @staticmethod
    def padded_batch(data, batch_size):
        padded_shapes = dict(
            user=tf.TensorShape([]),
            seq=tf.TensorShape([None]),
            ts=tf.TensorShape([None]),
            # future_seq=tf.TensorShape([None]),
            # future_ts=tf.TensorShape([None]),
            q_ts=tf.TensorShape([]),
            ans=tf.TensorShape([]),
            phase=tf.TensorShape([]),
            score_mask=tf.TensorShape([None]),
        )
        return data.padded_batch(batch_size, padded_shapes=padded_shapes)

    def convert_predictions(self, predictions: dict):
        uids_list = self.dp.raw_id_list['uids_list']
        vids_list = self.dp.raw_id_list['vids_list']
        uid2user = dict(zip(uids_list, range(len(uids_list))))
        vid2item = dict(zip(vids_list, range(len(vids_list))))
        ret = {}
        for uid, vids in predictions.items():
            items = [vid2item[vid] for vid in vids]
            user = uid2user[uid]
            ret[user] = items
        return ret

    def metric(self, predictions: dict):
        num_cases_full_per_phase = [0.0 for _ in range(NUM_PHASE)]
        ndcg_50_full_per_phase = [0.0 for _ in range(NUM_PHASE)]
        ndcg_50_half_per_phase = [0.0 for _ in range(NUM_PHASE)]
        num_cases_half_per_phase = [0.0 for _ in range(NUM_PHASE)]
        hitrate_50_full_per_phase = [0.0 for _ in range(NUM_PHASE)]
        hitrate_50_half_per_phase = [0.0 for _ in range(NUM_PHASE)]
        # pbar = tqdm(desc='metric', total=len(self.dp.vali_user2ppa))
        for user, (phase, pos, ans) in self.dp.vali_user2ppa.items():
            # pbar.update(1)
            if user not in predictions:
                continue

            item_degree = self.dp.item_deg_per_phase[phase][ans]
            median_item_degree = self.dp.mid_deg_per_phase[phase]
            # median_item_degree = [6, 11, 15, 18, 22, 24, 26][phase]

            rank = 0
            N = len(predictions[user])
            while rank < N and predictions[user][rank] != ans:
                rank += 1
            num_cases_full_per_phase[phase] += 1.0
            if rank < N:
                ndcg_50_full_per_phase[phase] += 1.0 / np.log2(rank + 2.0)
                hitrate_50_full_per_phase[phase] += 1.0
            if item_degree <= median_item_degree:
                num_cases_half_per_phase[phase] += 1.0
                if rank < N:
                    ndcg_50_half_per_phase[phase] += 1.0 / np.log2(rank + 2.0)
                    hitrate_50_half_per_phase[phase] += 1.0

        # pbar.close()

        score = np.zeros(6, dtype=float)

        for phase in range(NUM_PHASE):
            if num_cases_full_per_phase[phase] - num_cases_half_per_phase[phase] > 0:
                hitrate_pop = hitrate_50_full_per_phase[phase] - hitrate_50_half_per_phase[phase]
                hitrate_pop /= num_cases_full_per_phase[phase] - num_cases_half_per_phase[phase]
                ndcg_pop = ndcg_50_full_per_phase[phase] - ndcg_50_half_per_phase[phase]
                ndcg_pop /= num_cases_full_per_phase[phase] - num_cases_half_per_phase[phase]
            else:
                hitrate_pop = 0.0
                ndcg_pop = 0.0

            if num_cases_full_per_phase[phase]:
                ndcg_50_full_per_phase[phase] /= num_cases_full_per_phase[phase]
                hitrate_50_full_per_phase[phase] /= num_cases_full_per_phase[phase]

            if num_cases_half_per_phase[phase]:
                ndcg_50_half_per_phase[phase] /= num_cases_half_per_phase[phase]
                hitrate_50_half_per_phase[phase] /= num_cases_half_per_phase[phase]


            _score = np.array([
                hitrate_50_full_per_phase[phase],
                ndcg_50_full_per_phase[phase],
                hitrate_50_half_per_phase[phase],
                ndcg_50_half_per_phase[phase],
                hitrate_pop,
                ndcg_pop,
            ], dtype=float)

            if args.show_detail and num_cases_full_per_phase[phase]:
                print(f'phase: {phase}, vali: {format_metric(_score)}, full num: {num_cases_full_per_phase[phase]:.0f}, half num: {num_cases_half_per_phase[phase]:.0f}')

            score += _score

        return score


def format_metric(vs):
    return ','.join(['{:.4f}'] * len(vs)).format(*vs)

def show():
    data = Data()

    d = data.train.make_one_shot_iterator().get_next()
    d = utils.Object(**d)
    from models import UTILS
    sess = UTILS.get_session()

    cnt = 0
    hit = np.zeros(10)
    pbar = tqdm()
    while True:
        try:
            x = sess.run(d)
        except tf.errors.OutOfRangeError:
            break
        cnt += 1
        pbar.update(1)
        # for i, v in enumerate(x.seq):
        #     neighbors = data.dp.neighbors[0][v]
        #     if x.ans in neighbors:
        #         hit[i] += 1
        if cnt >= 10000: break
    pbar.close()
    print(hit, cnt, hit / cnt)


def test():
    data = Data()
    data.run_test()


def main():
    print('hello world, dataset.py')
    import main as main_md
    args.update(**main_md.parse_args())
    args.update(ds='v3s2')
    test()
    # show()


if __name__ == '__main__':
    main()

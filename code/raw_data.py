import numpy as np
import json
import os
from collections import defaultdict
import utils
from tqdm import tqdm

class RawData:
    item_mask = 'MASK'
    vali_unknown = 'V_UNK'
    test_unknown = 'T_UNK'

    def __init__(self, version, nb_phase, home=None):
        home = '../data'

        self.version = version
        self.nb_phase = nb_phase
        self.home = home
        self.load_data()

    def load_feat(self, feat_fn):
        item2txt = {}
        item2img={}
        with open(feat_fn) as f:
            for line in tqdm(f, desc='load feat', total=108916):
                line = f'[{line.strip()}]'
                item, txt, img = json.loads(line)
                item = str(item)
                item2txt[item] = txt
                item2img[item] = img
        return item2txt, item2img

    def load_data(self):
        prename = 'underexpose_'
        train_dir = f'{self.home}/{prename}train'
        test_dir = f'{self.home}/{prename}test'

        train_fn = f'{train_dir}/{prename}train_click-{{}}.csv'
        test_fn = f'{test_dir}/{prename}test_click-{{}}/{prename}test_click-{{}}.csv'
        test_time_fn = f'{test_dir}/{prename}test_click-{{}}/{prename}test_qtime-{{}}.csv'

        feat_fn = f'{train_dir}/{prename}item_feat.csv'

        self.item2txt, self.item2img = self.load_feat(feat_fn)


        # {uid: [(ts1, vid1), (ts2, vid2), ...]}
        seq_per_phase = [{} for _ in range(self.nb_phase)]
        # {uid: (p, ts, vid)}
        vali_pqa = {}
        test_pqa = {}

        # {vid: cnt}
        item_deg_per_phase = [defaultdict(int) for _ in range(self.nb_phase)]
        item_deg_self_per_phase = [defaultdict(int) for _ in range(self.nb_phase)]

        vali_or_test_uids = set()

        rdm = np.random.RandomState(123456)
        for p in range(self.nb_phase):
            train_seq = self.get_seq_from_fn(train_fn.format(p))
            test_seq = self.get_seq_from_fn(test_fn.format(p, p))
            assert len(set(train_seq.keys()) & set(test_seq.keys())) == 0

            all_seq = {}
            all_seq.update(train_seq)
            all_seq.update(test_seq)

            for _p in range(p, self.nb_phase):
                for uid, seq in all_seq.items():
                    for vid in RawData.split_seq(seq)[1]:
                        item_deg_per_phase[_p][vid] += 1

            for uid, seq in all_seq.items():
                for vid in RawData.split_seq(seq)[1]:
                    item_deg_self_per_phase[p][vid] += 1

            if p in {7, 8, 9}:
                vali_or_test_uids |= set(test_seq.keys())
                candidate_users = sorted(set(train_seq.keys()) - vali_or_test_uids)
                prob = len(test_seq) / len(candidate_users)
            else:
                prob = -1
                candidate_users = []

            vali_users = []
            for user in candidate_users:
                assert user not in vali_pqa
                if rdm.random_sample() < prob:
                    vali_users.append(user)

            vali_or_test_uids |= set(vali_users)
            for uid in vali_users:
                seq = all_seq[uid]
                q, a = seq[-1]
                vali_pqa[uid] = (p, q, a)
                seq[-1] = (q, RawData.vali_unknown)

            if p in {7, 8, 9}:
                with open(test_time_fn.format(p, p)) as f:
                    for line in f:
                        uid, ts = line[:-1].split(',')
                        all_seq[uid].append((ts, RawData.test_unknown))
                        test_pqa[uid] = (p, ts, RawData.test_unknown)

            seq_per_phase[p] = all_seq

            log = f'phase: {p}, #train: {len(train_seq)}, #vali: {len(vali_pqa)}, #test: {len(test_pqa)}'
            print(log)

        self.all_seq_per_phase = seq_per_phase
        self.vali_pqa = vali_pqa
        self.test_pqa = test_pqa
        self.item_deg_per_phase = item_deg_per_phase
        self.item_deg_self_per_phase = item_deg_self_per_phase

        self.mid_deg_per_phase = [0 for _ in range(self.nb_phase)]
        deg_per_phase = [[] for _ in range(self.nb_phase)]
        for uid, (p, q, a) in vali_pqa.items():
            deg = self.item_deg_per_phase[p][a]
            deg_per_phase[p].append(deg)

        for p in range(self.nb_phase):
            deg_list = sorted(deg_per_phase[p])
            n = len(deg_list)
            self.mid_deg_per_phase[p] = deg_list[n // 2] if n else 0

        print(self.mid_deg_per_phase)

    def save_validation_ans(self):
        save_home = f'{utils.data_dir}/raw_data_{self.version}'

        RawData.check_path(save_home)
        f_qtime = open(f'{save_home}/vali_time.txt', 'w')
        f_ans = open(f'{save_home}/validation_ans.txt', 'w')
        for uid, (p, ts, vid) in self.vali_pqa.items():
            line_time = f'{p},{uid},{ts},{vid}\n'
            f_qtime.write(line_time)
            deg = self.item_deg_per_phase[p][vid]
            line_ans = f'{p},{uid},{vid},{deg}\n'
            f_ans.write(line_ans)

        f_qtime.close()
        f_ans.close()

    def save_data(self):
        train_seq = self.merge_seq_dict(self.all_seq_per_phase)
        """ filter item, pop < 5 """
        uids = set()
        vids = set()
        ts_set = set()
        for uid, seq in train_seq.items():
            uids.add(uid)
            for ts, vid in seq:
                if vid not in (RawData.vali_unknown, RawData.test_unknown):
                    vids.add(vid)
                ts_set.add(ts)
        min_ts = float(min(ts_set))
        max_ts = float(max(ts_set))
        ts_rescale = 1e6

        print('ts and diff:')
        print(min_ts, max_ts, max_ts - min_ts)

        for u, (p, ts, vid) in self.vali_pqa.items():
            assert u in uids
            assert vid not in (RawData.vali_unknown, RawData.test_unknown)
            vids.add(vid)

        uids_list = sorted(uids, key=lambda x: int(x))
        vids_list = [RawData.item_mask, RawData.vali_unknown, RawData.test_unknown] + sorted(vids, key=lambda x: int(x))

        metadata = dict(
            nb_users=len(uids_list),
            nb_items=len(vids_list),
            nb_train=len(train_seq),
            nb_vali=len(self.vali_pqa),
            nb_test=len(self.test_pqa),
            mid_deg_per_phase=self.mid_deg_per_phase,
        )

        uid2int = dict(zip(uids_list, range(len(uids_list))))
        vid2int = dict(zip(vids_list, range(len(vids_list))))

        item_deg_per_phase = [[0] * len(vids_list) for _ in range(self.nb_phase)]
        item_deg_self_per_phase = [[0] * len(vids_list) for _ in range(self.nb_phase)]
        for p in range(self.nb_phase):
            for vid in vids_list:
                item = vid2int[vid]
                item_deg_per_phase[p][item] = self.item_deg_per_phase[p][vid]
                item_deg_self_per_phase[p][item] = self.item_deg_self_per_phase[p][vid]

        user2ts_seq = [[] for _ in range(len(uids_list))]
        user2item_seq = [[] for _ in range(len(uids_list))]
        vali_puiqa = []
        test_puiqa = []
        for uid, seq in train_seq.items():
            user = uid2int[uid]
            ts_seq = user2ts_seq[user]
            item_seq = user2item_seq[user]
            for_vali_cnt = 0
            for_test_cnt = 0
            for i, (ts, vid) in enumerate(seq):
                if vid == RawData.vali_unknown:
                    p, q, a = self.vali_pqa[uid]
                    q = (float(q) - min_ts) * ts_rescale
                    a = vid2int[a]
                    vali_puiqa.append([p, user, i, q, a])
                    for_vali_cnt += 1
                elif vid == RawData.test_unknown:
                    p, q, a = self.test_pqa[uid]
                    q = (float(q) - min_ts) * ts_rescale
                    a = vid2int[a]
                    test_puiqa.append([p, user, i, q, a])
                    for_test_cnt += 1
                item = vid2int[vid]
                ts_seq.append((float(ts) - min_ts) * ts_rescale)
                item_seq.append(item)

            assert for_vali_cnt <= 1
            assert for_test_cnt <= 1

        print(json.dumps(metadata, sort_keys=True, indent=2))

        item_feat = np.zeros(shape=[len(vid2int), 2, 128], dtype=np.float32)
        miss_feat_cnt = 0
        # for vid in tqdm(vids_list, desc='put item feat', gui=True):
        for vid in vids_list:
            if vid in self.item2txt:
                i = vid2int[vid]
                txt = self.item2txt[vid]
                img = self.item2img[vid]
                item_feat[i, 0] = txt
                item_feat[i, 1] = img
            else:
                miss_feat_cnt += 1
        print(f'miss feat cnt: {miss_feat_cnt}')

        save_home = f'{utils.data_dir}/{self.version}'

        RawData.check_path(save_home)
        print(f'save_home: {save_home}')

        RawData.save_json(dict(uids_list=uids_list, vids_list=vids_list), f'{save_home}/ids_list.json')
        RawData.save_json(metadata, f'{save_home}/metadata.json')
        RawData.save_json(user2item_seq, f'{save_home}/user2item_seq.json')
        RawData.save_json(user2ts_seq, f'{save_home}/user2ts_seq.json')
        RawData.save_json(test_puiqa, f'{save_home}/test_puiqa.json')
        RawData.save_json(vali_puiqa, f'{save_home}/vali_puiqa.json')
        RawData.save_json(item_deg_per_phase, f'{save_home}/item_deg.json')
        RawData.save_json(item_deg_self_per_phase, f'{save_home}/item_deg_self.json')
        np.save(f'{save_home}/item_feat.npy', item_feat)

        return save_home

    @staticmethod
    def check_path(path):
        if not os.path.isdir(path):
            print(f'Not found dir: {path}, creating it.')
            os.mkdir(path)
    @staticmethod
    def save_json(obj, fn):
        with open(fn, 'w') as f:
            json.dump(obj, f, sort_keys=True, indent=2)
    @staticmethod
    def load_json(fn):
        with open(fn) as f:
            return json.load(f)

    @staticmethod
    def get_seq_from_fn(fn):
        user_seq = defaultdict(list)
        with open(fn) as f:
            for line in f:
                uid, item, ts = line[:-1].split(',')
                user_seq[uid].append((ts, item))
        for uid, seq in user_seq.items():
            seq = sorted(seq)
            user_seq[uid] = seq
        return user_seq

    @staticmethod
    def make_seq(ts, vid):
        seq = sorted(set(zip(ts, vid)))
        return seq

    @staticmethod
    def split_seq(seq):
        ts = [float(ts) for ts, v in seq]
        vid = [v for ts, v in seq]
        return ts, vid

    @staticmethod
    def merge_seq_with_uid(seq_list, u):
        all_seq = []
        for seq in seq_list:
            all_seq += seq[u]
        all_seq = sorted(set(all_seq))
        return all_seq

    def merge_seq_dict(self, seq_list):
        all_seq = defaultdict(list)
        for seq in seq_list:
            for uid, s in seq.items():
                all_seq[uid].extend(s)
        # cnt = 0
        for uid, seq in all_seq.items():
            seq = sorted(set(seq))
            if uid in self.vali_pqa:
                vali_p, vali_qtime, vali_ans = self.vali_pqa[uid]
                seq = [s for s in seq if s[1] != vali_ans]

            all_seq[uid] = seq

        # print(cnt)
        return all_seq


def main():
    version = 'v3'

    r = RawData(version, nb_phase=10)
    r.save_validation_ans()
    r.save_data()


if __name__ == '__main__':
    main()

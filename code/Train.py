import time
import numpy as np

from utils import args, tqdm
import utils
import dataset

def tolist(ar):
    if type(ar) == list:
        return ar
    return ar.tolist()

def dict_mean(list_obj):
    keys = sorted(list_obj[0].keys())
    msg = []
    for k in keys:
        if list_obj[0][k].shape:
            v = np.concatenate([obj[k] for obj in list_obj], axis=0)
        else:
            v = [obj[k] for obj in list_obj]
        v = np.mean(v)
        msg.append(f'{k}:{v:.4f}')
    return ', '.join(msg)

class Train:
    def __init__(self, Model, data: dataset.Data):
        self.data = data
        self.build_model(Model)
        self.best_vali = None
        self.metric_weights = [0, 1, 0, 0]

    def build_model(self, Model):
        self.model = Model(self.data)

    def train_loop(self):
        brk = 0
        vali_best_w = -1
        for ep in range(args.epochs):
            pbar = tqdm(total=args.nb_vali_step, desc='training', leave=False)
            try:
                train_v = []
                t0 = time.time()
                for _ in range(args.nb_vali_step):
                    # dict
                    v = self.model.fit()
                    train_v.append(v)
                    pbar.update(1)
            finally:
                pbar.close()
            train_time = time.time() - t0
            train_msg = dict_mean(train_v)

            vali_v, vali_str, vali_loss = self.metric('vali')

            vali_w = np.sum([v * w for v, w in zip(vali_v, self.metric_weights)])

            if vali_w > vali_best_w:
                vali_best_w = vali_w
                self.best_vali = vali_v
                self.model.save(0)
                brk = 0
            else:
                brk += 1
            red = (brk == 0)

            msg = f'#{ep + 1}/{args.epochs} {train_msg}, brk: {brk}, vali: {vali_str}, {vali_loss:.4f}'
            if args.show_test and args.nb_test > 0:
                _, test_str, test_loss = self.metric('test')
                msg = f'{msg}, test: {test_str}'
            vali_time = time.time() - t0 - train_time
            msg = f'{msg}, time: {train_time:.0f}s,{vali_time:.0f}s'

            args.log.log(msg, red=red)

            if ep < args.min_train_epochs:
                brk = 0
            if brk >= args.early_stopping:
                break
        if args.epochs == 0:
            self.model.save(0)

        self.model.restore(0)

    def train(self):
        self.model.before_train()
        if self.model.need_train:
            self.train_loop()
        self.model.after_train()

    def final_test(self):
        if self.best_vali is None:
            vali, vali_str, vali_loss = self.metric('vali')
        else:
            vali_str = format_metric(self.best_vali)
        return vali_str

    def run_test(self):
        pass

    def metric(self, name):
        if name == 'vali':
            data = self.data.vali_batch
        elif name == 'test':
            data = self.data.test_batch
        elif name == 'train':
            data = self.data.train_batch
        else:
            raise Exception(f'unknown name: {name}')

        cnt = 0
        max_phase = 11
        full_hitrate_per_phase = [[] for _ in range(max_phase)]
        full_ndcg_per_phase = [[] for _ in range(max_phase)]
        half_hitrate_per_phase = [[] for _ in range(max_phase)]
        half_ndcg_per_phase = [[] for _ in range(max_phase)]
        pbar = tqdm(desc='predicting...', leave=False)
        loss_list = []
        for mv in self.model.metric(data):
            for i in range(len(mv.ndcg)):
                pbar.update(1)
                # if mv.future_seq[0][0] == 0: continue
                phase = mv.phase[i]
                item_deg = self.data.dp.item_deg_per_phase[phase][mv.true_item[i]]
                mid_deg = self.data.dp.mid_deg_per_phase[phase]
                if item_deg <= mid_deg:
                    half_hitrate_per_phase[phase].append(mv.hit_rate[i])
                    half_ndcg_per_phase[phase].append(mv.ndcg[i])
                full_hitrate_per_phase[phase].append(mv.hit_rate[i])
                full_ndcg_per_phase[phase].append(mv.ndcg[i])

            loss_list.append(mv.loss)
            cnt += 1
            if args.run_test and cnt > 10:
                break
        pbar.close()
        result = np.zeros(4, dtype=float)
        for p in range(max_phase):
            if half_hitrate_per_phase[p]:
                m1 = np.mean(full_hitrate_per_phase[p])
                m2 = np.mean(full_ndcg_per_phase[p])
                m3 = np.mean(half_hitrate_per_phase[p])
                m4 = np.mean(half_ndcg_per_phase[p])
                m = np.array([m1, m2, m3, m4])
                if args.show_detail:
                    print(f'phase: {p}, vali: {format_metric(m)}, full num: {len(full_hitrate_per_phase[p])}, half num: {len(half_hitrate_per_phase[p])}')

                result += m

        loss_mean = np.mean(loss_list)
        return result, format_metric(result), loss_mean

    def predict(self, name, vali_str=None):
        if name == 'vali':
            data = self.data.vali_batch
        elif name == 'test':
            data = self.data.test_batch
        elif name == 'train':
            data = self.data.train_batch
        else:
            raise Exception(f'unknown name: {name}')

        user_list = []
        top_items_list = []
        pbar = tqdm(desc='predicting...', leave=False)
        for pv in self.model.predict(data):
            pbar.update(1)
            user_list.extend(tolist(pv.user))
            top_items_list.extend(tolist(pv.top_items))
        if vali_str is None:
            fn = f'{utils.for_submit_dir}/{args.run_name}_{name}.csv'
        else:
            fn = f'{utils.for_submit_dir}/{args.run_name}_{name}_{vali_str}.csv'

        # fn = f'{utils.result_dir}/pred_{name}.csv'
        with open(fn, 'w') as f:
            for user, top_items in zip(user_list, top_items_list):
                user = str(self.data.raw_ids.uids_list[user])
                top_items = [str(self.data.raw_ids.vids_list[item]) for item in top_items]
                line = ','.join([user] + top_items)
                f.write(line + '\n')
        pbar.close()
        return fn

    def dump_features_all_item(self, name):
        if name == 'vali':
            data = self.data.vali_batch
        elif name == 'test':
            data = self.data.test_batch
        elif name == 'train':
            data = self.data.train_batch
        else:
            raise Exception(f'unknown name: {name}')

        users = []
        items = []
        logits = []

        from run_for_fuse import all_res
        fn_list = all_res.keys()

        user2items = None
        for fn in fn_list:
            _users, _items_list, _ = utils.load_pkl(f'{utils.for_fuse_dir}/{fn}_{name}')
            if user2items is None:
                user2items = {}
                for _u, _items in zip(_users, _items_list):
                    user2items[_u] = set(_items)
            else:
                assert set(user2items.keys()) == set(_users)
                for _u, _items in zip(_users, _items_list):
                    user2items[_u] |= set(_items)

        pbar = tqdm(desc=f'dump {name}, predicting...', leave=False)
        for pv in self.model.predict(data):
            pbar.update(1)
            users.extend(pv.user.tolist())
            for i in range(len(pv.user)):
                user = pv.user[i]
                _items_i = sorted(user2items[user])
                items.append(_items_i)
                logits.append(pv.all_scores[i, _items_i].tolist())

        pbar.close()

        feat = [users, items, logits]

        fn = f'{utils.for_fuse_dir}/union_{args.msg}_{name}'

        print(f'{utils.get_time_str()} dump file {fn}')
        utils.save_pkl(feat, fn)
        print(f'{utils.get_time_str()} dump file {fn} over')

        return fn

    def dump_features(self, name):
        if name == 'vali':
            data = self.data.vali_batch
        elif name == 'test':
            data = self.data.test_batch
        elif name == 'train':
            data = self.data.train_batch
        else:
            raise Exception(f'unknown name: {name}')

        users = []
        items = []
        logits = []

        pbar = tqdm(desc=f'dump {name}, predicting...', leave=False)
        for pv in self.model.predict(data):
            pbar.update(1)
            users.extend(pv.user.tolist())
            _items = pv.top_items.tolist()
            _scores = pv.top_scores.tolist()
            items.extend(_items)
            logits.extend(_scores)

            if args.run_test and pbar.n > 10:
                break

        pbar.close()

        feat = [users, items, logits]

        fn = f'{utils.for_fuse_dir}/{args.msg}_{name}'

        print(f'{utils.get_time_str()} dump file {fn}')
        utils.save_pkl(feat, fn)
        print(f'{utils.get_time_str()} dump file {fn} over')

        return fn


def format_metric(vs):
    return ','.join(['{:.4f}'] * len(vs)).format(*vs)

def main():
    print('hello world, Train.py')


if __name__ == '__main__':
    main()

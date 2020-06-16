import json
import sys, os
import time

from tqdm import tqdm as h_tqdm
import pickle

def load_pkl(fn):
    with open(f'{fn}.pkl', 'rb') as f:
        return pickle.load(f)

def save_pkl(obj, fn):
    with open(f'{fn}.pkl', 'wb') as f:
        pickle.dump(obj, f)

def check_path(home):
    if not os.path.isdir(home):
        print(f'Not found dir: {home}, creating it.')
        os.mkdir(home)


def red_str(s, tofile=False):
    s = str(s)
    if tofile:
        # s = f'**{s}**'
        pass
    else:
        s = f'\033[1;31;40m{s}\033[0m'
    return s


def get_time_str():
    # return time.strftime('%Y-%m-%d_%H.%M.%S') + str(time.time() % 1)[1:6]
    return time.strftime('%Y%m%d_%H%M%S') + str(time.time() % 1)[2:6]

def sys_run(cmd, retry=3):
    for _ in range(retry):
        ret = os.system(cmd)
        if ret == 0:
            return
    print(f'[ERROR!!!] run {cmd} failed')


class Logger:
    def __init__(self, fn=None, verbose=1):
        self.pre_time = time.time()
        self.fn = fn
        self.verbose = verbose

    def __str__(self):
        return str(self.fn)

    def log(self, s='', end='\n', red=False):
        if self.fn == -1:
            return
        s = str(s)
        if self.verbose == 1:
            p = red_str(s) if red else s
            print(p, end=end)
        elif self.verbose == 2:
            p = red_str(s, tofile=True) if red else s
            print(p, end=end)
        now_time = time.time()
        s = s + end
        if now_time - self.pre_time > 30 * 60:
            s = get_time_str() + '\n' + s
            self.pre_time = now_time
        if self.fn is not None:
            with open(self.fn, 'a') as f:
                fs = red_str(s, tofile=True) if red else s
                f.write(fs)
        sys.stdout.flush()


class Graph:
    def __init__(self):
        self.adj = {}
        self._nb_edges = 0

    def add_edge(self, a, b, ts):
        e = (a, b)
        self.adj.setdefault(a, [])
        self.adj[a].append([b, ts])
        self._nb_edges += 1

    def get_adj(self, a):
        return self.adj.get(a, [])

    def nb_edges(self):
        return self._nb_edges


class Object(dict):
    def __update(self):
        self.__dict__ = self

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__update()

    def cp_update(self, **kwargs):
        obj = Object(**self)
        obj.update(**kwargs)
        return obj

    def filter(self, *keys):
        ret = Object()
        for k in keys:
            ret.set(k, self.get(k))
        return ret

    def json(self, line=False):
        # d = {}
        # for k, v in self.items():
        #     try:
        #         s = json.dumps(v, sort_keys=True)
        #     except Exception:
        #         s = str(v)
        #     d[k] = s
        d = self
        indent = None if line else 2
        return json.dumps(d, indent=indent, sort_keys=True, default=lambda x: str(x))


class my_tqdm:
    def __init__(self, desc, total, leave, interval=30):
        self.desc = desc
        self.total = total
        self.cnt = 0
        self.leave = leave
        self.interval = interval
        if self.leave:
            print(f'>>> [begin] {self.desc}...')
            sys.stdout.flush()
        self.bt = time.time()
        self.pre_time = self.bt

    def update(self, n):
        self.cnt += n
        now = time.time()
        if now - self.pre_time > self.interval:
            t = now - self.bt
            v = self.cnt / t
            self.pre_time = now
            print(f'    [wait] {self.desc} cnt: {self.cnt}, time: {t:.0f}s, v: {v:.2f}it/s')
            sys.stdout.flush()

    def close(self):
        t = time.time() - self.bt
        v = self.cnt / t
        if self.leave:
            print(f'--- [end] {self.desc} cnt: {self.cnt}, time: {t:.0f}s, v: {v:.2f}it/s')
            sys.stdout.flush()

def load_gz(fn):
    import gzip
    user_list = []
    logits_list = []
    with gzip.open(fn, 'rt') as f:
        for line in f:
            user, *items = line[:-1].split(',')
            user_list.append(user)
            logits_dict = {}
            for item in items:
                item_id, logits = item.split(':')
                logits = float(logits)
                logits_dict[item_id] = logits
            logits_list.append(logits_dict)
    return user_list, logits_list


def tqdm(verbose=None, desc='tqdm', leave=True, total=None):
    if verbose is None:
        verbose = args.get('verbose', 1)
    if verbose == 1:
        return h_tqdm(desc=desc, total=total, leave=leave, ncols=90, ascii=True)
    return my_tqdm(desc=desc, total=total,  leave=leave)


args = Object()
# hdp: log/, results/, data/
project_name = 'KDDCUP20'
run_time_dir = '../user_data/tmp_data'
check_path(run_time_dir)

data_dir = f'{run_time_dir}/data'
log_dir = f'{run_time_dir}/log'
for_fuse_dir = f'{run_time_dir}/for_fuse'
save_dir = f'../user_data/model_data'
for_submit_dir = '../prediction_result'

check_path(data_dir)
check_path(log_dir)
check_path(save_dir)
check_path(for_submit_dir)
check_path(for_fuse_dir)


def main():
    print('hello world, logger.py')



if __name__ == '__main__':
    main()

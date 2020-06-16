import utils
from utils import args
import main as mmmmain
import dataset
from tqdm import tqdm
from collections import defaultdict

def load_pkl(fn):
    user2ans = {}
    users, items_list, logits_list = utils.load_pkl(fn)
    for user, items, logits in zip(users, items_list, logits_list):
        item2logits = dict(zip(items, logits))
        user2ans[user] = item2logits

    return user2ans

max_score_per_ans = None
score_diff_per_ans = None
# user2item_set = None

def fuse(ans_list, weight_list, name):
    users = sorted(ans_list[0].keys())
    user2tops = {}

    global max_score_per_ans, score_diff_per_ans
    if max_score_per_ans is None:
        max_score_per_ans = []
        score_diff_per_ans = []
        for i in range(len(ans_list)):
            max_score, min_score = None, None
            for user in tqdm(users, desc='count min max', ncols=90, ascii=True):
                logits_dict = ans_list[i][user]

                for item, score in logits_dict.items():
                    if max_score is None:
                        max_score = score
                        min_score = score
                    else:
                        max_score = max(max_score, score)
                        min_score = min(min_score, score)

            print(f'ans {i}, min ~ max: {min_score} ~ {max_score}')

            score_diff = max_score - min_score
            max_score_per_ans.append(max_score)
            score_diff_per_ans.append(score_diff)

    if name == 'vali':
        user2ppa = data.dp.vali_user2ppa
    else:
        user2ppa = data.dp.test_user2ppa

    is_first_fuse = args.get('is_first_fuse', True)
    for user in tqdm(users, desc='fuse', ncols=90, ascii=True, leave=is_first_fuse):
        phase, pos, ans = user2ppa[user]
        final_ans = defaultdict(float)
        for ans_i, (ans, weight) in enumerate(zip(ans_list, weight_list)):
            if weight < 1e-6:
                continue

            logits_dict = ans[user]
            for item, score in logits_dict.items():
                pop_inv = data.dp.item_pop_inv[item]
                pop_log_inv = data.dp.item_pop_log_inv[item]
                s = score / score_diff_per_ans[ans_i]
                if args.mode_pop == 'log':
                    s *= (1.0 + pop_log_inv)

                if args.mode_rare == 'linear':
                    if data.dp.item_is_half[phase, item] and alpha_rare[ans_i] > 0:
                        s *= (alpha_rare[ans_i] + pop_inv)

                w = weight

                final_ans[item] += w * s

        tops = sorted(final_ans.keys(), key=lambda _item: (-final_ans[_item], _item))[:args.nb_topk]
        user2tops[user] = tops

    args.update(is_first_fuse=False)
    args.update(show_detail=False)
    return user2tops


def metric(ans_list, weight_list):
    pred = fuse(ans_list, weight_list, 'vali')
    m = data.metric(pred)
    m_str = dataset.format_metric(m)
    print(f'{weight_list}: {m_str}')
    return m


def get_ans_list(fn_list, name):
    ans_list = []
    for fn in fn_list:
        fn = f'{utils.for_fuse_dir}/{fn}_{name}'
        ans = load_pkl(fn)
        print('len:', len(ans))
        ans_list.append(ans)

    return ans_list



def dfs_w_list(w_list, i):
    if i >= len(w_list):
        yield [None] * len(w_list)
        return

    for ret in dfs_w_list(w_list, i + 1):
        for w in w_list[i]:
            ret = list(ret)
            ret[i] = w
            yield ret

def run_fuse_vali(fn_list):
    ans_list = get_ans_list(fn_list, 'vali')
    best_v = 0
    best_weight = None

    w_list = [
        [10],
        [2.5],
        [19],
        [3],
    ]
    for weight_list in dfs_w_list(w_list, 0):
        print(weight_list)

    for weight_list in dfs_w_list(w_list, 0):
        m = metric(ans_list, weight_list)
        v = m[1]
        if v > best_v:
            best_v = v
            best_weight = list(weight_list)

    print(best_v, best_weight)
    return best_weight


def write_ans(fn_list, weight_list):
    print('weights:', weight_list)
    uids_list = data.dp.raw_id_list['uids_list']
    vids_list = data.dp.raw_id_list['vids_list']

    ans_list = get_ans_list(fn_list, 'test')
    user2tops = fuse(ans_list, weight_list, 'test')
    fn = f'{utils.for_submit_dir}/result.csv'
    with open(fn, 'w') as f:
        for user in sorted(user2tops.keys()):
            tops = user2tops[user]

            user = uids_list[user]
            tops = [vids_list[v] for v in tops]

            line = ','.join([user] + tops)
            f.write(line + '\n')
    print(fn)


data: dataset.Data = None
alpha_rare = [1.5] * 2 + [1.1] * 2

def run():
    global data
    data = dataset.Data()

    prefix = 'union_'
    fn_list = [
        f'{prefix}id_att_3',
        f'{prefix}id_last',

        f'{prefix}c_att_5',
        f'{prefix}c_last',
    ]

    print('fn list:')
    print(fn_list)
    print('alpha_rare:')
    print(alpha_rare)
    best_weight = run_fuse_vali(fn_list)
    print(write_ans(fn_list, weight_list=best_weight))


def main():
    args.update(mmmmain.parse_args())
    args.update(ds='v3')
    run()


if __name__ == '__main__':
    main()

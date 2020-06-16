import sys
import os
import utils

def run(name, dim_k, dump='dump', add_cmd=''):
    res = all_res[name]
    model = 'ATT_ts' if res.split('_')[1] == 'att' else 'LastItem'

    cmd = f'python main.py -model={model} -ds=v3 -restore_model={res} -k={dim_k} -show_detail -{dump} -nb_topk=2000 -nb_rare_k=1000 -msg={name} {add_cmd}'
    print(cmd)

    ret = os.system(cmd)
    if ret != 0:
        input('Error!!!!!!')

all_res = dict(
    id_att_3='id_att_3',
    id_last='id_last',

    c_att_5='c_att_5',
    c_last='c_last',
)


def main():
    run('id_att_3', 1024, dump='dump')
    run('id_last', 1024, dump='dump')
    run('c_att_5', 256, dump='dump', add_cmd='-seq_length=5')
    run('c_last', 256, dump='dump')

    run('id_att_3', 1024, dump='dump_all', add_cmd='-skip_vali')
    run('id_last', 1024, dump='dump_all', add_cmd='-skip_vali')
    run('c_att_5', 256, dump='dump_all', add_cmd='-skip_vali -seq_length=5')
    run('c_last', 256, dump='dump_all', add_cmd='-skip_vali')



if __name__ == '__main__':
    main()
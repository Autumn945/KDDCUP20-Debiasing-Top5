# coding=utf-8
import argparse
import time

import numpy as np
import os
import random
import sys

import Train
import dataset
import models
import utils
from utils import args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    """ yard """
    parser.add_argument('-master_id', type=str, default='cmd')
    parser.add_argument('-run_on_yard', action='store_true')

    """ base """
    parser.add_argument('-show_test', action='store_true')
    parser.add_argument('-run_tb', action='store_true')
    parser.add_argument('-run_test', action='store_true')
    parser.add_argument('-epochs', type=int, default=500)
    parser.add_argument('-min_train_epochs', type=int, default=20)
    parser.add_argument('-es', '--early_stopping', type=int, default=5)
    parser.add_argument('-valistep', '--nb_vali_step', type=int, default=-1)
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('-k', '--dim_k', type=int, default=1024)
    parser.add_argument('-seed', type=int, default=123456)
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-ds', type=str, default='xxx')
    parser.add_argument('-verbose', type=int, default=1)
    parser.add_argument('-msg', type=str, default='')
    parser.add_argument('-model', type=str, default='xxx')

    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-opt', type=str, default='adam')
    parser.add_argument('-dropout', type=float, default=1.0)

    """ L2 """
    parser.add_argument('-l2_all', type=float, default=0)
    parser.add_argument('-l2_emb_item', type=float, default=0)
    parser.add_argument('-l2_emb_user', type=float, default=0)
    parser.add_argument('-l2_emb_ts', type=float, default=0)
    parser.add_argument('-l2_bl', type=float, default=0)
    parser.add_argument('-l2_bias', type=float, default=0)

    """ xxx """
    parser.add_argument('-show_detail', action='store_true')
    parser.add_argument('-skip_test', action='store_true')
    parser.add_argument('-test_result', action='store_true')
    parser.add_argument('-save_ans', action='store_true')
    parser.add_argument('-skip_vali', action='store_true')
    parser.add_argument('-dump', action='store_true')
    parser.add_argument('-dump_all', action='store_true')

    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-nb_rare_k', type=int, default=-1)
    parser.add_argument('-nb_topk', type=int, default=50)

    """ use """
    parser.add_argument('-use_bl', type=int, default=0)
    parser.add_argument('-use_pop_bias_for_pred', type=int, default=0)
    parser.add_argument('-use_cosine_for_pred', type=int, default=0)
    parser.add_argument('-use_item_feat', type=str, default='emb')
    parser.add_argument('-use_item_mlp', type=str, default='')
    parser.add_argument('-use_unused_vali', type=str, default='')

    parser.add_argument('-deg_shift', type=int, default=0)

    """ mode """
    parser.add_argument('-mode', type=str, default='none')
    parser.add_argument('-mode_loss', type=str, default='softmax')
    parser.add_argument('-mode_loss_weight', type=str, default='none')
    parser.add_argument('-mode_neg_sample', type=str, default='all')
    parser.add_argument('-mode_pos_sample', type=str, default='all')
    parser.add_argument('-mode_rare', type=str, default='none')
    parser.add_argument('-mode_pop', type=str, default='none')
    parser.add_argument('-mode_pred', type=str, default='dot')
    parser.add_argument('-mode_pop_bias', type=str, default='none')
    parser.add_argument('-mode_item_emb_init', type=str, default='none')
    parser.add_argument('-mode_pred_phase', type=str, default='789')
    parser.add_argument('-mode_resample', type=str, default='none')

    """ number """
    parser.add_argument('-nb_neg', type=int, default=200)
    parser.add_argument('-seq_length', type=int, default=3)
    parser.add_argument('-seq_max_dt', type=float, default=1e6)
    parser.add_argument('-data_dt_less_than', type=float, default=-1)
    parser.add_argument('-data_dt_greater_than', type=float, default=-1)
    parser.add_argument('-alpha_diff', type=float, default=0.1)
    parser.add_argument('-alpha_score_div', type=float, default=1.0)
    parser.add_argument('-alpha_feat_div', type=float, default=1.0)
    parser.add_argument('-alpha_future_div', type=float, default=1.0)
    parser.add_argument('-alpha_rare', type=float, default=1.0)
    parser.add_argument('-alpha_rare_base', type=float, default=0.0)
    parser.add_argument('-alpha_rare_mul', type=float, default=1.0)
    parser.add_argument('-alpha_pop_base', type=float, default=1.0)
    parser.add_argument('-alpha_neg_sample', type=float, default=1.0)
    parser.add_argument('-alpha_loss_weight', type=float, default=1.0)
    parser.add_argument('-alpha_resample', type=int, default=1)

    """ user """
    parser.add_argument('-user_rate', type=float, default=1.0)
    parser.add_argument('-user_dp', type=float, default=1.0)

    """ GNN  """
    parser.add_argument('-gnn', type=int, default=0)
    parser.add_argument('-gnn_adj_length', type=int, default=3)
    parser.add_argument('-gnn_min_edge_cnt', type=int, default=1)
    parser.add_argument('-gnn_depth', type=int, default=1)
    parser.add_argument('-gnn_use_bn', type=int, default=0)

    parser.add_argument('-ex_other_phase', type=int, default=1)
    parser.add_argument('-ts_unit', type=int, default=10)

    """ restore  """
    parser.add_argument('-restore_model', type = str, default='')
    parser.add_argument('-restore_train', type = int, default=0)
    # parser.add_argument('-restore_msg', type = str, default='')
    # parser.add_argument('-return_model', type = str, default='')

    a = parser.parse_args().__dict__
    return a


def main(**main_args):
    begin_time = time.time()

    # init args
    args.clear()
    command_line_args = parse_args()
    args.update(**command_line_args)
    args.update(**main_args)

    if args.ds == 'test':
        args.update(run_test=True)

    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


    # get Model, set model default args
    Model = vars(models)[args.model]
    args.update(**Model.args)
    args.update(**main_args)

    if args.run_test:
        args.update(epochs=2, nb_vali_step=2, batch_size=4)

    # get data
    data = dataset.Data()
    min_epochs = args.nb_train / (args.batch_size * args.nb_vali_step)

    if min_epochs < 1.0:
        args.update(nb_vali_step=int(np.ceil(args.nb_train / args.batch_size)))
        min_epochs = args.nb_train / (args.batch_size * args.nb_vali_step)
    args.update(min_epochs=int(np.ceil(min_epochs)))

    # run_name: time-x-Model-ds
    model_name = Model.__name__
    time_str = utils.get_time_str()
    run_name = f'{time_str}-{model_name}-{args.ds}'

    if args.msg:
        run_name = f'{run_name}-{args.msg}'
    if args.run_test:
        run_name = f'{run_name}-test'
    if args.restore_model:
        run_name = f'{run_name}-restored'

    args.update(run_name=run_name)

    log_fn = f'{utils.log_dir}/{run_name}.log'
    begin_time_str = utils.get_time_str()
    args.update(pid=os.getpid())
    log = utils.Logger(fn=log_fn, verbose=args.verbose)
    args.update(log=log)
    args.log.log(f'argv: {" ".join(sys.argv)}')
    args.log.log(f'log_fn: {log_fn}')
    args.log.log(f'main_args: {utils.Object(**main_args).json()}')
    args.log.log(f'args: {args.json()}')
    args.log.log(f'Model: {model_name}')
    args.log.log(f'begin time: {begin_time_str}')

    T = Train.Train(Model, data)
    if args.restore_model:
        T.model.restore_from_other(args.restore_model)
    if not args.restore_model or args.restore_train:
        try:
            T.train()
        except KeyboardInterrupt as e:
            pass
        T.model.restore(0)

    if args.skip_vali:
        test_str = 'none'
    else:
        test_str = T.final_test()
        args.log.log(f'vali: {test_str}', red=True)

    if args.dump_all:
        T.dump_features_all_item('vali')
        T.dump_features_all_item('test')
        return

    if args.dump:
        T.dump_features('vali')
        T.dump_features('test')
        return

    args.log.log(run_name, red=True)
    if args.restore_model:
        args.log.log(f'restored from {args.restore_model}', red=True)

    dt = time.time() - begin_time
    end_time_str = utils.get_time_str()
    args.log.log(f'end time: {end_time_str}, dt: {dt / 3600:.2f}h')
    print(end_time_str, log_fn, f'##### over, time: {dt / 3600:.2f}h')

    return test_str, time_str


if __name__ == '__main__':
    print(os.getcwd())
    main()


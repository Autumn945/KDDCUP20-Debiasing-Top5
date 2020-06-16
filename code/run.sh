python raw_data.py

python main.py -ds=v3 -model=ATT_ts -msg=id_att_3
python main.py -ds=v3 -model=ATT_ts -mode_item_emb_init=all -k=256 -lr=1e-3 -seq_length=5 -msg=c_att_5

python main.py -ds=v3 -model=LastItem -msg=id_last
python main.py -ds=v3 -model=LastItem -mode_item_emb_init=all -k=256 -lr=1e-3 -msg=c_last

python run_for_ruse.py

python Fuse.py


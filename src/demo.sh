# Test your own images
#python main.py --data_test Demo --scale 4 --pre_train download --test_only --save_results --dir_demo ../old_test

#python main.py --data_test B100+Pollock+Seismic+Set14+Set5+Urban100 --scale 2 --pre_train download --test_only --save_results --dir_data ../../Data/

#python -m pdb main.py --model EDSR --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test B100 --pre_train ../models/EDSR/EDSR_x2.pt --test_only --dir_data ../../Data/

python main.py --model EDSR --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Set14 --pre_train ../models/EDSR/EDSR_x2.pt --test_only --dir_data ../../Data/

#python main.py --model EDSR --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test B100+Seismic+Set14+Set5+Urban100 --pre_train ../models/EDSR/EDSR_x2.pt --test_only --save_results --dir_data ../../Data/

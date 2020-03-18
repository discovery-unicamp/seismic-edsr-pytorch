# Test your own images

#python main.py --data_test Demo --scale 2 --pre_train download --test_only --save_results
#python main.py --model EDSR --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Demo --pre_train ../models/EDSR/EDSR_x2.pt --test_only --save_results 
python main.py --model EDSR --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --data_test Demo --pre_train ../models/EDSR/EDSR_x4.pt --test_only --save_results 

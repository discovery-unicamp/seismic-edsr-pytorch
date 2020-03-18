# Test your own images
# Para carregar imagens, usar --ext img

python -m pdb main.py \
    \
    --model 'EDSR' \
    --save edsr_test_x2 \
    --act 'relu' \
    --n_resblocks 32 \
    --n_feats 64 \
    --res_scale 1 \
    --shift_mean True \
    \
    --dir_data '../../Data' \
    --data_train 'DIV2K' \
    --data_test 'DIV2K' \
    --data_range '1-20/801-805' \
    --scale '2' \
    --patch_size 48 \
    --rgb_range 255 \
    --n_colors 3 \
    \
    --reset \
    --test_every 100 \
    --epochs 10 \
    --batch_size 16 \

#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_test_x2 --reset --data_train DIV2K --data_test DIV2K --dir_data '../../Dataset' --epochs 1
# python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_test_x2 --reset --dir_data '../../Dataset' --epochs 1
    # --data_range '1-800/801-810' \

# Test your own images
# Para carregar imagens, usar --ext img

# Parameters:
#
# - program execution
# - model
# - data
# - training
# - optimizer

#    --load toy_x2 \
#    --save_results \
#    --resume -1 \
#    --test_only \
#    --reset \

SCALE=2
DATASET=DIV2K_TIFF
PATCH_SIZE=$(( 48*SCALE ))

python main.py \
    --reset \
    --resume -1 \
    --load edsrX${SCALE}_$DATASET \
    --save edsrX${SCALE}_$DATASET \
    --tensorboard \
    --n_threads        6 \
    --seed             42 \
    \
    --template EDSR_paper \
    --scale $SCALE \
    --shift_mean False \
    \
    --dir_data ../../Data \
    --data_train $DATASET \
    --data_test $DATASET \
    --data_range '1-800/1-10' \
    --input_range '-1., 1.' \
    --tensor_range '-1., 1.' \
    --ext img \
    --patch_size $PATCH_SIZE  \
    --n_colors 1 \
    --no_augment \
    \
    --print_every 10 \
    --test_every 50 \
    --epochs 5 \

# Test your own images
# Para carregar imagens, usar --ext img

# Parameters:
#
# - program execution
# - model
# - data
# - training

#    --save_results \
#    --resume -1 \
#    --test_only \
#    --debug \
#    --reset \

SCALE=2
DATASET=DIV2K_TIFF
#DATASET=NAMSS

# Set data_range based on the dataset
if [ "$DATASET" = "DIV2K_TIFF" ]; then
	DATA_RANGE='1-800/1-20'
elif [ "$DATASET" = "NAMSS" ]; then
	DATA_RANGE='1-2000/1-50'
fi


python main.py \
    --load edsrX${SCALE}_$DATASET \
    --save edsrX${SCALE}_$DATASET \
    --n_threads        0 \
    --seed             4226 \
    --tensorboard \
    \
    --template EDSR_paper \
    --scale $SCALE \
    --shift_mean False \
    \
    --cache_data \
    --dir_data ../../Data \
    --data_train $DATASET \
    --data_test $DATASET \
    --data_range $DATA_RANGE \
    --input_range '-1., 1.' \
    --tensor_range '-1., 1.' \
    --ext img \
    --n_colors 1 \
    \
    --print_every 50 \
    --test_every 1000 \
    --epochs 300 \

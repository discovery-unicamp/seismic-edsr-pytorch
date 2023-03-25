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
SEED=4226
TEMPLATE=EDSR_paper
DATA_RANGE='1-800/1-20'
TEST_EVERY=1000


python main.py \
    --load edsrX${SCALE}_$DATASET \
    --save edsrX${SCALE}_$DATASET \
    --n_threads 0 \
    --seed $SEED \
    --tensorboard \
    \
    --template $TEMPLATE \
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
    --test_every $TEST_EVERY \
    --epochs 300 \

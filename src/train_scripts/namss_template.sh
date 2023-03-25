# Parameters:
#
# - program execution
# - model
# - data
# - training

#    --resume -1 \
#    --debug \
#    --reset \

SCALE=
SEED=
TEMPLATE=
EPOCHS=

DATASET=NAMSS
DATA_RANGE='1-2746/1-70'
TEST_EVERY=1030
EXPERIMENT_DIR=${TEMPLATE}-X${SCALE}-$DATASET-Seed${SEED}

python main.py \
    --load $EXPERIMENT_DIR \
    --save $EXPERIMENT_DIR \
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
    --epochs $EPOCHS \

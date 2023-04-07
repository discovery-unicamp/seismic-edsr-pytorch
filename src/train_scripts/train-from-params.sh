
# Run setting the following parameters appropriately:
#
# bash train-from-parameters.sh SCALE= SEED= EPOCHS= TEMPLATE= DATASET= TRAIN_DATA_RANGE= TEST_EVERY=

declare $@

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
    --data_range $TRAIN_DATA_RANGE \
    --input_range '-1., 1.' \
    --tensor_range '-1., 1.' \
    --ext img \
    --n_colors 1 \
    \
    --print_every 50 \
    --test_every $TEST_EVERY \
    --epochs $EPOCHS \


# Run setting the following parameters appropriately:
#
# bash test-from-parameters.sh SCALE= SEED= TEMPLATE= DATASET= TEST_DATA_RANGE=

declare $@

EXPERIMENT_DIR=${TEMPLATE}-X${SCALE}-$DATASET-Seed${SEED}
python main.py \
    --test_only \
    --save_results \
    --load $EXPERIMENT_DIR \
    --save $EXPERIMENT_DIR \
    --resume 0 \
    --pre_train "../experiment/$EXPERIMENT_DIR/model/model_best.pt" \
    --n_threads 0 \
    --n_GPUs 1 \
    --seed $SEED \
    \
    --template $TEMPLATE \
    --scale $SCALE \
    --shift_mean False \
    \
    --dir_data ../../Data \
    --data_train $DATASET \
    --data_test $DATASET \
    --data_range $TEST_DATA_RANGE \
    --input_range '-1., 1.' \
    --tensor_range '-1., 1.' \
    --ext img \
    --n_colors 1 \
    \
    --epochs 0 \

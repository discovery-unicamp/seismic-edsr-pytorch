
# Run setting the following parameters appropriately:
#
# bash test-from-parameters.sh SCALE= SEED= TEMPLATE= DATASET= TEST_DATA_RANGE=

declare $@

EXPERIMENT_DIR=experiment-$EXPERIMENT_NUM
MODEL_DIR=${TEMPLATE}-X${SCALE}-$DATASET-Seed${SEED}
PRE_TRAIN_PATH="../$EXPERIMENT_DIR/$MODEL_DIR/model/model_best.pt"

python main.py \
    --test_only \
    --save_results \
    --resume 0 \
    --experiment_dir $EXPERIMENT_DIR \
    --load $MODEL_DIR \
    --save $MODEL_DIR \
    --pre_train $PRE_TRAIN_PATH \
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
    --no_augment \
    \
    --epochs 0 \

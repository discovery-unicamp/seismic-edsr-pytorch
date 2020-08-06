# Test your own images
# Para carregar imagens, usar --ext img

# Parameters:
#
# - program exectution
# - model
# - data
# - training
# - optimizer

#    --load toy_x2 \
#    --save_results \
#    --resume -1 \
#    --test_only \
    #--reset \
# python -m pdb main.py \

SCALE=4

python main.py \
    --resume -1 \
    --load edsr_synthetic_X$SCALE \
    --tensorboard \
    --template edsr_synthetic \
    --n_threads        12 \
    --seed             42 \
    --save edsr_synthetic_X$SCALE \
    \
    --scale $SCALE \
    \
    --print_every 50 \
    --test_every 1000 \
    --epochs 300 \

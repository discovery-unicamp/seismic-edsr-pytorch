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
#    --reset \
#    --save_results \
#    --resume -1 \
#    --test_only \
# python -m pdb main.py \

python main.py \
    --reset \
    --tensorboard \
    --template TOY \
    --n_threads        12 \
    --seed             42 \
    --save toy_tiff_09ABR_X4 \
    \
    --scale '4' \
    \
    --print_every 100 \
    --test_every 1000 \
    --epochs 10 \

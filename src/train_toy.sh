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
    --save toy_tiff_08ABR \
    \
    --scale '2' \
    \
    --print_every 5 \
    --test_every 5 \
    --epochs 10 \

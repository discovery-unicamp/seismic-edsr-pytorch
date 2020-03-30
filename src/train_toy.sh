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
    --template TOY \
    --n_threads        12 \
    --print_every      100 \
    --seed             42 \
    --save toy \
    \
    --scale '2' \
    \
    --test_every 100 \
    --epochs 10 \

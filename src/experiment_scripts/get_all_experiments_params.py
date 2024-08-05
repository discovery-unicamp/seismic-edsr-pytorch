import glob, pathlib

TEST_RANGE = {
    'DIV2K_TIFF' : '1-80',
    'NAMSS' : '1-284',
    'Urban100' : '1-100',
    }

def get_param_from_path(path):
    path = pathlib.Path(path).parts
    exp_num = path[-2].split('-')[-1]
    model, scale, dataset, seed = path[-1].split('-')
    scale = scale.replace('X', '')
    seed = seed.replace('Seed', '')
    
    params = f"EXPERIMENT_NUM={exp_num} TEMPLATE={model} SCALE={scale} DATASET={dataset} SEED={seed} TEST_DATA_RANGE={TEST_RANGE[dataset]} BENCHMARK=--benchmark CHOP=--chop"
    print(params)
    if 'DIV2K' in dataset:
        dataset = 'Urban100'
        params = f"EXPERIMENT_NUM={exp_num} TEMPLATE={model} SCALE={scale} DATASET={dataset} SEED={seed} TEST_DATA_RANGE={TEST_RANGE[dataset]} BENCHMARK=--benchmark CHOP=--chop"
        print(params)

paths = glob.glob('../experiment-*/*/')
paths = sorted(filter(lambda p: 'BICUBIC' not in p, paths))

for p in paths:
    get_param_from_path(p)

from copy import deepcopy

import model
import template
import torch
import utility

from option import args as _args

models = [
    'EDSR_paper',
    'DDBPN',
    'RCAN',
    'RDN',
    'WDSR',
]

model_info = dict()

for mname in models:
    args = deepcopy(_args)
    args.cpu = True
    args.template = mname
    params_per_scale = dict()
    for scale in [2,4]:
        args.scale[0] = scale

        template.set_template(args)
        checkpoint = utility.checkpoint(args)

        _model = model.Model(args, checkpoint)

        num_trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
        num_non_trainable = sum(p.numel() for p in _model.parameters() if not p.requires_grad)
        total_params = (num_trainable+num_non_trainable)/10**6 

        params_per_scale[scale] = total_params

    model_info[args.model] = params_per_scale

print()
for m in model_info:
    print(m)
    for s in model_info[m]:
        print(f"{s}X - {model_info[m][s]:.1f}M")
    print()

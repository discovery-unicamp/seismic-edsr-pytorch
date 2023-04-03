import model
import template
import torch
import utility

from option import args

models = [
    'EDSR_paper',
    'DDBPN',
    'RCAN',
    'RDN',
    'WDSR',
]

args.cpu = True

for mname in models:
    args.template = mname

    template.set_template(args)
    checkpoint = utility.checkpoint(args)

    _model = model.Model(args, checkpoint)

    num_trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    num_non_trainable = sum(p.numel() for p in _model.parameters() if not p.requires_grad)

    print("Trainable params:", num_trainable)
    print("Non-trainable params:", num_non_trainable)
    print(f"Total params: {(num_trainable+num_non_trainable)/10**6:.1f}M")
    print()

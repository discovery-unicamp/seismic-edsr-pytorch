from option import args, parser
import sys

out_file = "options.txt"
min_space = 3

args = list(vars(args))
args.sort(key=lambda k: k.lower())

max_size = 0
for arg in args:
    if len(arg) > max_size:
        max_size = len(arg)
max_size += min_space

with open(out_file, 'w') as f:
    for arg in args:
        num_spaces = max_size - len(arg)
        a = '--{}{}{}\n'.format(arg, ' '*num_spaces, parser.get_default(arg))
        f.write(a)

print("Opções salvas em", out_file)

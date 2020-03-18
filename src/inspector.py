import inspect
# from option import args
# from utility import timer, checkpoint
# import numpy as np

# instance_sep=

def inspect_class(obj, print_init=True):
	init = '__init__' if print_init else None
	print("Object:", type(obj))

	print("Attributes:")
	for attr_name, _ in inspect.getmembers(obj, predicate=lambda o:
			not inspect.ismethod(o) and not inspect.isbuiltin(o)):
		if not attr_name.startswith('__'):
			print('   ', attr_name)

	print()
	print("Methods:")
	for func_name, func in inspect.getmembers(obj, predicate=lambda o:
			inspect.ismethod(o) or inspect.isbuiltin(o)):
		if not func_name.startswith('__') or func_name == init:
			sig = '()'
			try:
				sig = inspect.signature(func)
			except:
				pass
			print('    ', func_name, str(sig), sep='')

def inspect_module(obj):
	print("Module:", type(obj))

	print("Classes:")
	for class_name, _ in inspect.getmembers(obj, predicate=inspect.isclass):
		if not class_name.startswith('__'):
			print('   ', class_name)

	print()
	print("Functions:")
	for func_name, func in inspect.getmembers(obj, predicate=lambda o:
			inspect.isfunction(o)):
		if not func_name.startswith('__') or func_name == init:
			sig = '()'
			try:
				sig = inspect.signature(func)
			except:
				pass
			print('    ', func_name, str(sig), sep='')

# obj = checkpoint(args)
# obj = np.array(2)
# inspect_class(obj)

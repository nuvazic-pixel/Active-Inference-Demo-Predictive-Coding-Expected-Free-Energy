.PHONY: install linear active mnist

install:
	pip install -e .

linear:
	pc-linear-demo --epochs 200 --infer-steps 30 --hidden 16 --top 4 --plot

active:
	pc-active-demo --plot

mnist:
	mnist-foveated-demo --epochs 5 --plot

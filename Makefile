
PHONY: all

train:
	python train.py --task classification --save output_filename1

eval:
	python eval.py --task fit_centers --weights output_filename1

submit:
	CUDA_VISIBLE_DEVICES=0 python submit.py --task classification --weights classification_resnet34_lr0.001.pth.tar

dev:
	python train.py --task classification --save resnet101_lr0.01 --epochs 200 --bsz 16 --lr_decay 40 --lr 0.01
	python train.py --task classification --save resnet101_lr0.005 --epochs 200 --bsz 16 --lr_decay 40 --lr 0.005
	python train.py --task classification --save resnet101_lr0.003 --epochs 200 --bsz 16 --lr_decay 40 --lr 0.003
	python train.py --task classification --save resnet101_lr0.001 --epochs 200 --bsz 16 --lr_decay 40 --lr 0.001
	python train.py --task classification --save resnet101_lr0.005e300d60 --epochs 300 --bsz 16 --lr_decay 60 --lr 0.005
	python train.py --task classification --save resnet101_lr0.001e500d70 --epochs 500 --bsz 16 --lr_decay 70 --lr 0.001

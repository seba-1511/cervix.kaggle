
PHONY: all

train:
	python train.py --task classification --save output_filename1

eval:
	python eval.py --task fit_centers --weights output_filename1

submit:
	CUDA_VISIBLE_DEVICES=0 python submit.py --task classification --weights classification_dev.pth.tar

dev:
	python train.py --task classification --save resnet34_lr0.005 --epochs 51 --bsz 16 --lr_decay 15 --lr 0.005


PHONY: all

train:
	python train.py --task classification --save output_filename1

eval:
	python eval.py --task fit_centers --weights output_filename1

submit:
	python submit.py --task classification --weights classification_224baseline.pth.tar

dev:
	CUDA_VISIBLE_DEVICES=1,3 python train.py --task classification --save 224baseline_resnet50 --epochs 100 --bsz 16 

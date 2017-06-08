
PHONY: all

train:
	python train.py --task fit_centers --save output_filename1

eval:
	python eval.py --task fit_centers --weights output_filename1

submit:
	python submit --task resnet34_classification --weights resnet34_classification

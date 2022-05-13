#!/bin/bash

pip install gdown

# webemo
gdown --id 1qOY-kAFtPYfUY12qeI-IRq7pjJ1YpG_z

# emotion 6
gdown --id 1I7fWwi8TtjeA6EYleUQOD7jIFOA0Bpx9

# unbiasedEmo
gdown --id 1pqWD6ItRNtQXXPUwV4fqhVewSG8Cfgx7

# unzip
unzip WEBEmo.zip -d ./data && rm WEBEmo.zip

unzip UnBiasedEmo.zip -d ./data && rm UnBiasedEmo.zip

unzip Emotion-6.zip -d ./data && rm Emotion-6.zip

unzip ./data/UnBiasedEmo/images.zip -d ./data/UnBiasedEmo/ && rm ./data/UnBiasedEmo/images.zip

unzip ./data/Emotion-6/images.zip -d ./data/Emotion-6/ && rm ./data/Emotion-6/images.zip && rm ./data/Emotion-6/images/anger/rage/rage/287.jpg

# kaggle
pip install kaggle --upgrade # update kaggle config json
kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge && mkdir ./kaggle && cd ./kaggle && unzip ../challenges-in-representation-learning-facial-expression-recognition-challenge.zip && rm ../challenges-in-representation-learning-facial-expression-recognition-challenge.zip && tar -xf fer2013.tar.gz


# anapy3 --grid_ncpus=8 --grid_submit=batch --grid_mem=20G --grid_gpu -u -m torch.distributed.launch --nproc_per_node=4 main_ddp.py --use_self_imgF --model resnet50 --freeze_modules "~fc,avgpool,layer3,layer4"  --freeze_first_n_epochs 60 --data_dir  ~/VisualEmotion/data/images_bk --lr 0.01 --n_epochs 60 --n_workers 0 --batch_size 64 --milestones 16 32 48 --sampler imbalance --comments "6_classes_exclude_love"

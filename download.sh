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

unzip ./data/Emotion-6/images.zip -d ./data/Emotion-6/ && rm ./data/Emotion-6/images.zip
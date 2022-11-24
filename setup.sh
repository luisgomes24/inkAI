#!bin/bash
python -m virtualenv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
mkdir models
wget https://zenodo.org/record/7348430/files/classifier_model.pt -P models
wget https://zenodo.org/record/7348430/files/dit_fine_tuned.pt -P models

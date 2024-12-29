cd ..
git clone git@github.com:565353780/ma-sh.git
git clone git@github.com:565353780/ulip-manage.git
git clone git@github.com:565353780/base-trainer.git
git clone https://github.com/atong01/conditional-flow-matching.git CFM

cd ma-sh
./dev_setup.sh

cd ../ulip-manage
./dev_setup.sh

cd ../base-trainer
./dev_setup.sh

cd ../CFM
pip install -r requirements.txt
pip install -e .

pip install -U timm einops diffusers flow_matching thop

pip install git+https://github.com/openai/CLIP.git

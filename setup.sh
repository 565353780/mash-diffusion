cd ..
git clone https://github.com/565353780/ma-sh.git
git clone https://github.com/565353780/ulip-manage.git
git clone https://github.com/565353780/base-trainer.git
git clone https://github.com/565353780/dino-v2-detect.git
git clone https://github.com/565353780/mash-occ-decoder.git
git clone https://github.com/565353780/wn-nc.git
git clone https://github.com/atong01/conditional-flow-matching.git CFM

cd ma-sh
./setup.sh

cd ../ulip-manage
./setup.sh

cd ../base-trainer
./setup.sh

cd ../dino-v2-detect
./setup.sh

cd ../mash-occ-decoder
./setup.sh

cd ../wn-nc
./setup.sh

cd ../CFM
pip install -r requirements.txt
pip install -e .

pip install -U timm einops diffusers flow_matching thop

pip install git+https://github.com/openai/CLIP.git

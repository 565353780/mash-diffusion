cd ..
git clone https://github.com/565353780/base-trainer.git
git clone https://github.com/565353780/dino-v2-detect.git
git clone https://github.com/565353780/mash-occ-decoder.git
git clone https://github.com/565353780/wn-nc.git
git clone https://github.com/565353780/blender-manage.git

cd base-trainer
./setup.sh

cd ../dino-v2-detect
./setup.sh

cd ../mash-occ-decoder
./setup.sh

cd ../wn-nc
./setup.sh

cd ../blender-manage
./setup.sh

pip install -U timm einops diffusers flow_matching thop torchcfm \
  tos crcmod

pip install -U cupy-cuda12x

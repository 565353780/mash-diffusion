cd ..
git clone git@github.com:565353780/base-trainer.git
git clone git@github.com:565353780/dino-v2-detect.git
git clone git@github.com:565353780/mash-occ-decoder.git
git clone git@github.com:565353780/wn-nc.git
git clone git@github.com:565353780/blender-manage.git

cd base-trainer
./dev_setup.sh

cd ../dino-v2-detect
./dev_setup.sh

cd ../mash-occ-decoder
./dev_setup.sh

cd ../wn-nc
./dev_setup.sh

cd ../blender-manage
./dev_setup.sh

pip install -U timm einops diffusers flow_matching thop torchcfm

pip install -U cupy-cuda12x

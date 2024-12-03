cd ..
git clone git@github.com:565353780/ma-sh.git
git clone https://github.com/atong01/conditional-flow-matching.git CFM

cd ma-sh
./dev_setup.sh

cd ../CFM
pip install -U torch torchvision torchaudio
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -e .

pip install -U timm einops

pip install git+https://github.com/openai/CLIP.git

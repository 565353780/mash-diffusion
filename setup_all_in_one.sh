cd ..
git clone https://github.com/565353780/ma-sh.git
git clone https://github.com/565353780/base-trainer.git
git clone https://github.com/565353780/dino-v2-detect.git
git clone https://github.com/565353780/mash-occ-decoder.git
git clone https://github.com/565353780/wn-nc.git
git clone https://github.com/565353780/blender-manage.git
git clone https://github.com/kacperkan/light-field-distance.git
git clone https://github.com/thu-ml/SageAttention.git

pip install -U torch torchvision torchaudio

pip install -U opencv-python omegaconf torchmetrics fvcore iopath \
  xformers submitit cuml-cu12 trimesh Cython pykdtree timm einops \
  scikit-image open3d tensorboard prefetch_generator tqdm pyvista \
  PyMCubes diffusers flow_matching thop torchcfm cupy-cuda12x \
  tos crcmod

cd light-field-distance
pip install .

cd ../SageAttention
pip install .

cd ../ma-sh
pip install .

cd ../blender-manage
./setup.sh

cd ../wn-nc/wn_nc/Lib/ANN/

rm -rf build

mkdir build
cd build

cmake .. -DCMAKE_INSTALL_PREFIX=./install
make -j
make install

cd ../../../..
rm -rf bin

./wn_nc/Bash/build_GR_cpu.sh
./wn_nc/Bash/build_GR_cuda.sh

cd wn_nc/Cpp
pip install -e .

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
  OMP_NUM_THREADS=16 \
  torchrun \
  --nproc_per_node=6 \
  --master_port 29500 \
  train_edm.py

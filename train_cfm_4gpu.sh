CUDA_VISIBLE_DEVICES=2,3 \
  OMP_NUM_THREADS=16 \
  torchrun \
  --nproc_per_node=2 \
  --master_port 29500 \
  train_cfm.py

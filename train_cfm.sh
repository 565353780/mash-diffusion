CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=16 \
  torchrun \
  --nproc_per_node=1 \
  --master_port 29600 \
  train_cfm.py

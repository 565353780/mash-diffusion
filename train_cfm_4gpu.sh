CUDA_VISIBLE_DEVICES=1,2,3 \
  OMP_NUM_THREADS=16 \
  torchrun \
  --nproc_per_node=3 \
  --master_port 29500 \
  train_cfm.py

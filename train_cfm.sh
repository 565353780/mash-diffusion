CUDA_VISIBLE_DEVICES=0,1,2,3 \
  OMP_NUM_THREADS=16 \
  torchrun \
  --nproc_per_node=4 \
  --master_port 29600 \
  train_cfm.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  OMP_NUM_THREADS=16 \
  torchrun \
  --nproc_per_node=8 \
  --master_port 29600 \
  train_cfm.py

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
  OMP_NUM_THREADS=16 \
  torchrun \
  --nproc_per_node=7 \
  --master_port 29500 \
  train_cfm.py
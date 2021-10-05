python3 -m torch.distributed.run \
  --nproc_per_node 1 \
  --nnodes 1 \
  --node_rank 0 \
  train.py \
    --distributed \
    --use_amp \
    --epochs 100 \
    --batch_size 16 \
    --project VITON-HD \
    --shuffle \
    --sync_bn \
    --load_height 1024 \
    --load_width 768
    --use_wandb \
    --log_interval 100 \
    --workers 5


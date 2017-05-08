export CUDA_VISIBLE_DEVICES=0
nohup invoke train_direction --config experiments/train_direction_clip_1m.yaml >& train_direction_clip_1m.out & 
export CUDA_VISIBLE_DEVICES=1
nohup invoke train_direction --config experiments/train_direction_clip_50cm.yaml >& train_direction_clip_50cm.out & 
export CUDA_VISIBLE_DEVICES=2
nohup invoke train_direction --config experiments/train_direction_clip_20cm.yaml >& train_direction_clip_20cm.out & 
export CUDA_VISIBLE_DEVICES=3
nohup invoke train_direction --config experiments/train_direction_clip40_1m.yaml >& train_direction_clip40_1m.out & 
export CUDA_VISIBLE_DEVICES=4
nohup invoke train_direction --config experiments/train_direction_clip40_50cm.yaml >& train_direction_clip40_50cm.out & 
export CUDA_VISIBLE_DEVICES=5
nohup invoke train_direction --config experiments/train_direction_clip40_20cm.yaml >& train_direction_clip40_20cm.out & 

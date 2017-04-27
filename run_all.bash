export CUDA_VISIBLE_DEVICES=0
nohup invoke train_distance --config experiments/train_distance_1m.yaml >& train_distance_1m.out & 
export CUDA_VISIBLE_DEVICES=1
nohup invoke train_distance --config experiments/train_distance_50cm.yaml >& train_distance_50cm.out & 
export CUDA_VISIBLE_DEVICES=2
nohup invoke train_distance --config experiments/train_distance_20cm.yaml >& train_distance_20mm.out & 
export CUDA_VISIBLE_DEVICES=3
nohup invoke train_direction --config experiments/train_direction_1m.yaml >& train_direction_1m.out & 
export CUDA_VISIBLE_DEVICES=4
nohup invoke train_direction --config experiments/train_direction_50cm.yaml >& train_direction_50cm.out & 
export CUDA_VISIBLE_DEVICES=5
nohup invoke train_direction --config experiments/train_direction_20cm.yaml >& train_direction_20cm.out & 
export CUDA_VISIBLE_DEVICES=6
nohup invoke train_direction --config experiments/train_direction_clip_1m.yaml >& train_direction_clip_1m.out & 
export CUDA_VISIBLE_DEVICES=7
nohup invoke train_direction --config experiments/train_direction_clip_50cm.yaml >& train_direction_clip_50cm.out & 

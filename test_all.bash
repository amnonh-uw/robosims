export CUDA_VISIBLE_DEVICES=0
nohup invoke test_distance --config experiments/test_distance_1m.yaml >& test_distance_1m.out & 
export CUDA_VISIBLE_DEVICES=1
nohup invoke test_distance --config experiments/test_distance_50cm.yaml >& test_distance_50cm.out & 
export CUDA_VISIBLE_DEVICES=2
nohup invoke test_distance --config experiments/test_distance_20cm.yaml >& test_distance_20cm.out & 
export CUDA_VISIBLE_DEVICES=3
nohup invoke test_direction --config experiments/test_direction_1m.yaml >& test_direction_1m.out & 
export CUDA_VISIBLE_DEVICES=4
nohup invoke test_direction --config experiments/test_direction_50cm.yaml >& test_direction_50cm.out & 
export CUDA_VISIBLE_DEVICES=5
nohup invoke test_direction --config experiments/test_direction_20cm.yaml >& test_direction_20cm.out & 
export CUDA_VISIBLE_DEVICES=6
nohup invoke test_direction --config experiments/test_direction_clip_1m.yaml >& test_direction_clip_1m.out & 
export CUDA_VISIBLE_DEVICES=7
nohup invoke test_direction --config experiments/test_direction_clip_50cm.yaml >& test_direction_clip_50cm.out & 

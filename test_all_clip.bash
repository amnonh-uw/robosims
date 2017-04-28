export CUDA_VISIBLE_DEVICES=0
nohup invoke test_direction --config experiments/test_direction_clip_1m.yaml >& test_direction_clip_1m.out & 
export CUDA_VISIBLE_DEVICES=1
nohup invoke test_direction --config experiments/test_direction_clip_50cm.yaml >& test_direction_clip_50cm.out & 
export CUDA_VISIBLE_DEVICES=2
nohup invoke test_direction --config experiments/test_direction_clip_20cm.yaml >& test_direction_clip_20cm.out & 
export CUDA_VISIBLE_DEVICES=3
nohup invoke test_direction --config experiments/test_direction_clip40_1m.yaml >& test_direction_clip40_1m.out & 
export CUDA_VISIBLE_DEVICES=4
nohup invoke test_direction --config experiments/test_direction_clip40_50cm.yaml >& test_direction_clip40_50cm.out & 
export CUDA_VISIBLE_DEVICES=5
nohup invoke test_direction --config experiments/test_direction_clip40_20cm.yaml >& test_direction_clip40_20cm.out & 

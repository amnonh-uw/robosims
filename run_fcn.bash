export CUDA_VISIBLE_DEVICES=0
nohup invoke train_translation --config experiments/train_direction_FCN.yaml>& train_direction_FCN.out

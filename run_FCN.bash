export CUDA_VISIBLE_DEVICES=1
nohup invoke train_translation --config experiments/train_translation_50cm_FCN.yaml >& train_tanslation_50cm_FCN.out & 
sleep 10
export CUDA_VISIBLE_DEVICES=2
nohup invoke train_translation --config experiments/train_translation_20cm_FCN.yaml >& train_tanslation_20cm_FCN.out & 
sleep 10
export CUDA_VISIBLE_DEVICES=3
nohup invoke train_translation --config experiments/train_translation_1m_FCN.yaml >& train_tanslation_1m_FCN.out & 

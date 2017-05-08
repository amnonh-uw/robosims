export CUDA_VISIBLE_DEVICES=0
nohup invoke gen_train_dataset --config=gen_dataset/1m_large.yaml >& 1m_large.out &
sleep 20
nohup invoke gen_train_dataset --config=gen_dataset/50cm_large.yaml >& 50cmm_large.out &
sleep 20
nohup invoke gen_train_dataset --config=gen_dataset/20cm_large.yaml >& 20cm_large.out &

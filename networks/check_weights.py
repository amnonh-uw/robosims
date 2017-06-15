import numpy as np
# tf_name = 'Convolution6'
# b_name = 'predict_flow_final'
# tf_name = 'Resample4'
# b_name = 'predict_flow_resize'
# tf_name = "Eltwise4"
# b_name = "blob44"
# tf_name = "conv1"
# b_name = "conv1"
tf_name = 'conv1'
b_name = "conv1_weights"
data_dict = np.load('Flownet_Simple.npy', encoding='bytes').item()
t = data_dict[tf_name][b'weights']
c = np.load('blobs/' + b_name + '.npy')
print('tensorflow shape {}'.format(t.shape))
print('caffe shape {}'.format(c.shape))
c = np.transpose(c, [2, 3, 1, 0])
print('tensorflow shape {}'.format(t.shape))
print('caffe shape {}'.format(c.shape))
print(t-c)

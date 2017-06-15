import numpy as np
# tf_name = 'Convolution6'
# b_name = 'predict_flow_final'
# tf_name = 'Resample4'
# b_name = 'predict_flow_resize'
# tf_name = "Eltwise4"
# b_name = "blob44"
# tf_name = "conv1"
# b_name = "conv1"
tf_name = "Concat1"
b_name = "input"
t = np.load('tf_blobs/' + tf_name + '.npy')
print(t.shape)
c = np.load('blobs/' + b_name + '.npy')
print(c.shape)
c = np.transpose(c, [0, 2, 3, 1])
print(t.shape)
print(c.shape)
print(t-c)

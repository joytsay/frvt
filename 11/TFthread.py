import tensorflow as tf
import numpy as np

config = tf.ConfigProto(device_count = {'CPU':1})
# config = tf.ConfigProto()
# config.use_per_session_threads = True
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1
# session_inter_op_thread_pool = 1

# config.use_per_session_threads = 0
serialized = config.SerializeToString()
result = ["0x"+c.encode('hex') for c in serialized]
print(result)
sess = tf.Session(config=config)

x = tf.placeholder(tf.float32, name='X', shape=(4000,9000))
w = tf.placeholder(tf.float32, name='W', shape=(9000,1))
b = tf.fill((4000,1),-1.,name='bias')
y =  tf.matmul(x,w)+b
s = tf.reduce_max(y)

x_data = np.random.randn(4000,9000)
w_data = np.random.randn(9000,1)

with sess:
    while True:
        out_p =  sess.run(s, feed_dict={x:x_data, w:w_data})

print(out_p)


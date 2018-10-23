import tensorflow as tf
from model.input_fn import train_input_fn
'''
x = tf.placeholder(tf.float32, [None, 4], name = 'x')

w_v, w_i = tf.nn.top_k(x, k = tf.shape(x)[0], sorted=False, name = 'w')
#w_i = [[3,2,1,0]]
#ww = tf.expand_dims(w_i, axis=0)
#mul = tf.constant([0,1,2,3], dtype = tf.int32)
mul = tf.range(0, tf.shape(x)[0])
mul = tf.expand_dims(mul, axis= 1)
w_i = tf.shape(x)[0] * mul + w_i
w_i = tf.reshape(w_i, [tf.shape(x)[0]*tf.shape(x)[0], 1])

print(w_i)
xx = tf.reshape(x, [tf.shape(x)[0]*tf.shape(x)[0]])
print(xx)
www = tf.gather_nd(xx, w_i)
www = tf.reshape(www, [tf.shape(x)[0],tf.shape(x)[0]])

with tf.Session() as sess:
    wwww = sess.run(www, feed_dict = {x:[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,16,15]]})

print(wwww)
'''
def dcg_array_tf(values, indices):

    mul = tf.range(0, tf.shape(values)[0])
    mul = tf.expand_dims(mul, axis= 1)
    indices = tf.shape(values)[0] * mul + indices
    indices = tf.reshape(indices, [tf.shape(values)[0]*tf.shape(values)[0], 1])

    reshaped_vals = tf.reshape(values, [tf.shape(values)[0]*tf.shape(values)[0]])
    
    rearr_vals = tf.gather_nd(reshaped_vals, indices)
    rearr_vals = tf.reshape(rearr_vals, [tf.shape(values)[0],tf.shape(values)[0]])
    
    return rearr_vals

def dcg(values, indices = None):
    if indices is not None:
        values = dcg_array_tf(values, indices)
    
    create_range = tf.range(start = 2, limit = tf.shape(values)[0] + 2, delta = 1, name='range')
    dcg_den = tf.zeros_like(create_range, dtype = tf.int32)

    dcg_den = create_range + dcg_den
    dcg_den_float = tf.cast(dcg_den, dtype = tf.float32)
    out = tf.log(dcg_den_float, 'dcg_den')

    dcg_val = values/ out
    dcg_val = tf.reduce_mean(dcg_val, axis = 1, name = 'dcg_matrix')
    
    return dcg_val


x = tf.placeholder(tf.float32, [None, 4], name = 'x')
vals_og, w_i = tf.nn.top_k(x, k = tf.shape(x)[0], sorted=False, name = 'w')

indices_TR = [[1,0,2,3], [3,2,1,0], [0,1,2,3], [3,2,0,1]]

dcg_og = dcg(vals_og)
dcg_tr = dcg(vals_og, indices_TR)

dcg_sm = dcg( tf.contrib.framework.sort(values = vals_og, axis = 1) )

dcg_ac = (dcg_tr - dcg_sm) / (dcg_og - dcg_sm)

dcg_ac= tf.reduce_mean(dcg_ac)

with tf.Session() as sess:
    all = sess.run(dcg_ac, feed_dict = {x:[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,16,15]]})

print(all)
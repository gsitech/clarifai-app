import tensorflow as tf
import os, math

#Prevents the program from printing TF compile warnings to the terminal
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

MODEL_FILE = 'model/model_file.txt'
def hashing_model():
    '''Creates a NN to convert a vector of
        1024 floats to a vector of CODE_SIZE many bits

        :param MODEL_FILE- folder name to store the trained tensorflow graph
    '''
    if not os.path.isfile(MODEL_FILE):
        print('No model data')
        exit(1)
    
    return __initialise_variables(MODEL_FILE)



def __dense_layer(x, W, b):
    '''Creates a fully connected layer

        :param x- input to the layer
        :param W- weights of the current layer
        :param b- biases of the current layer
    '''
    x= tf.add(tf.matmul(x,W),b)

    return tf.nn.relu(x)

def __weights_biases(index, prev_features, features):
    '''Creates weights and biases for a layer

        :param index- (for simplified debugging only)
        :param prev_features- features of the layer connected as input to the
                                current layer
        :param features- features of the current layer

        :return- created weights and biases
    '''
    index_w = 'W'+str(index)
    index_b = 'b'+str(index)

    W = tf.get_variable(index_w, [prev_features, features], dtype=tf.float32 ,initializer= tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(index_b, [features], dtype=tf.float32, initializer= tf.contrib.layers.xavier_initializer())
    return W, b


def __initialise_variables(MODEL_FILE):
    '''Creates a neural network that reduces the dimensionality of
        the representation space.
        HIDDEN_LAYERS specifies the number of hidden layers to be created.
    
        :return- A neural network
    '''

    #placeholder initialiser
    x = tf.placeholder(tf.float32, [None, 1024], name = 'x')
    
    out = x
    count = 1
    with open(MODEL_FILE, 'r') as f:
        ff = f.readlines()
        prev_features = 1024
        for line in ff:
            W, b = __weights_biases(count, prev_features, int(line))
            out = __dense_layer(out, W, b)
            count+=1
            prev_features = int(line)

    return x, out
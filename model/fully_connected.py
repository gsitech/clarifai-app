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
    dense_layers = []
    with open(MODEL_FILE, 'r') as f:
        ff = f.readlines()
        for line in ff:
            out = tf.layers.dense(out, int(line), activation=tf.nn.relu, name='dense'+str(count))
            dense_layers.append(out)
            count+=1
            prev_features = int(line)

    out1 = tf.layers.dense(out, prev_features, activation=tf.nn.sigmoid, name='bitvectors'+str(count))
    dense_layers.append(out1)
    outt = tf.round(out1)
    dense_layers.append(outt)

    count+=1

    ls = tf.layers.dense(outt, prev_features, activation=tf.nn.sigmoid, name='lss'+str(count))

    return x, outt, ls
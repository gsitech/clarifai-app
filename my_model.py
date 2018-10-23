import tensorflow as tf
from model.utils import Params
from model.input_fn import train_input_fn
from model.input_fn import test_input_fn
from model.fully_connected import hashing_model
from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
from model.dcg_function import dcg_accuracy
import argparse, os, time, math
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def sup(params, mode):

    if mode == 'train':
        train(params)
    else:
        return test(params)

def model_fn(params, mode):
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    #with tf.variable_scope('model'):
        # Compute the embeddings with the model
    x, embeddings = hashing_model()

    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    labels = tf.placeholder(tf.int64, [None, 1], 'labels')

    tf.summary.histogram('Embeddings distribution', embeddings)

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin=params.margin,
                                                squared=params.squared)
    elif params.triplet_strategy == "batch_hard":
        loss = batch_hard_triplet_loss(labels, embeddings, margin=params.margin,
                                       squared=params.squared)
    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES

    # Summaries for training
    tf.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

    #Define Accuracy
    accuracy = dcg_accuracy(x, embeddings, params)
    tf.summary.scalar('DCG Accuracy', accuracy)

    # Define training step that minimizes the loss with the Gradient Descent optimizer
    optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
    
    # Define variable that holds the value of the global step
    gst = tf.train.create_global_step()
    train_op = optimizer.minimize(loss, global_step=gst)
    
    if mode == 'train':
        init = tf.global_variables_initializer()
        sess = tf.Session()
        
        merged = tf.summary.merge_all()
        
        train_writer = tf.summary.FileWriter(os.getcwd() + '/train_writer',
                                      sess.graph)
        sess.run(init)

        '''
        vars={}
        vars['x'] = x
        vars['labels'] = labels
        vars['sess'] = sess
        vars['train_op'] = train_op
        vars['accuracy'] = accuracy
        vars['loss'] = loss
        vars['merged'] = merged
        vars['train_writer'] = train_writer
        vars['global_step_tensor'] = gst
        '''
        
        return x, labels, sess, train_op, accuracy, loss, merged, train_writer, gst
    elif mode == 'test':
        return x, embeddings, labels, accuracy
'''
def compute_swaps(arr): 
    n = len(arr) 

    arrpos = [*enumerate(arr)] 
    arrpos.sort(key = lambda it:it[1]) 

    vis = {k:False for k in range(n)} 
       
    ans = 0
    for i in range(n): 
        if vis[i] or arrpos[i][0] == i: 
            continue
              
        cycle_size = 0
        j = i 
        while not vis[j]: 
            vis[j] = True
              
            j = arrpos[j][0] 
            cycle_size += 1
              
        if cycle_size > 0: 
            ans += (cycle_size - 1) 
    return ans

def hamming_dist(og, tr):
    mp = {}
    
    for i,j in enumerate(og): 
    	mp[j] = i

    for i in range (len(tr)): 
    	tr[i] = mp[tr[i]] 
    
    return compute_swaps(tr)

def compute_hamming_acc(og, tr):
    #if len(og.shape[0]) == 1:
    #    return hamming_dist(og,tr)
    res = np.zeros(og.shape[0])
    for i in range (len(og)):
        res[i] = hamming_dist(og[i],tr[i])

    return res

def create_rev_arr(values):
    l = np.zeros(values.shape)
    for i in range (len(values)):
        for j in range (len(values[0])):
            l[i][len(l[0])-j-1] = values[i][j]

    return l
'''
def train(params):
    x, labels, sess, train_op, accuracy, loss, merged, train_writer, gst = model_fn(params, mode='train')

    xdata, labeldata = train_input_fn()
    print('Training begins')
    start_time=time.time()
    
    for j in range (params.num_epochs):
        
        print("EPOCH NUMBER: ", j+1)
        avg_acc=0
        avg_lss=0
        #avg_hamming_acc = 0
        for k in range(0, len(xdata), params.batch_size):
            current_batch_x_train = xdata[k:k+params.batch_size]
            current_batch_label_train = labeldata[k:k+params.batch_size]
            
            _, acc, lss, merg, gg = sess.run([train_op, accuracy, loss, merged, gst],
                            feed_dict = {x: current_batch_x_train, labels: current_batch_label_train})

            avg_acc+=acc
            avg_lss+=lss

            #print('max', dcg_og)
            #print('model val', dcg_tr)
            #print('min val', dcg_ascend)
            
            #hamming_acc = compute_hamming_acc(ind_og, ind_tr)
            #print(hamming_acc)
            #hamming_acc = np.mean(hamming_acc)
            #avg_hamming_acc+= hamming_acc
            #print("Mini-batch training Accuracy", acc)
            #print("Mini-batch training Loss", lss)
            #print("Mini-batch dcg", dcg_acc)

        train_writer.add_summary(merg, global_step=gg)

                            
        print("Average Training DCG Accuracy= ", avg_acc / math.ceil(len(xdata) / params.batch_size))
        print("Average Training Triplet Loss= ", avg_lss / math.ceil(len(xdata) / params.batch_size))

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        
    train_time=time.time() - start_time
    
    saver= tf.train.Saver()
    saver.save(sess , os.path.join(os.getcwd(), params.model_file))
    
    print("Total training time= ", train_time, "seconds")

def test(params):
    x, embeddings, labels, accuracy = model_fn(params, mode = 'test')

    sess = tf.Session()
    try:
        tf.train.Saver().restore(sess , os.path.join(os.getcwd(), params.model_file))
    except:
        print("Please create a model before using it for prediction")
        print("Run the following command-> python my_model.py train")
        exit(1)
    
    xdata, labeldata = test_input_fn()

    print ("Model restored!")

    start_time=time.time()
    out, acc= sess.run([embeddings, accuracy], feed_dict={x:xdata, labels: labeldata})
    test_time=time.time() - start_time
    print("Time taken for prediction= ", test_time, "seconds")

    print("Prediction accuracy", acc)
    
    return out


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('mode', default='test',
                    help="train/test")


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    sup(params, args.mode)
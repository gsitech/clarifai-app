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
    with tf.variable_scope('model'):
        # Create a hashing model
        x, embeddings, ls_embeddings = hashing_model()

    #embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    #tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)
    tf.summary.histogram('Embeddings_distribution', embeddings)

    with tf.variable_scope('labels'):
        labels = tf.placeholder(tf.int64, [None, 1], 'labels')
        
        #TODO
        #Remove the hardcoded 36
        labels_one_hot = tf.one_hot(indices = labels, depth = 36, dtype = tf.float32)
        labels_one_hot = tf.squeeze(labels_one_hot, axis = 1)

    # Define loss
    with tf.variable_scope(params.loss + '_loss'):
        if params.loss == 'triplet':
            if params.triplet_strategy == "batch_all":
                loss, fraction = batch_all_triplet_loss(labels, ls_embeddings, margin=params.margin,
                                                        squared=params.squared)
            elif params.triplet_strategy == "batch_hard":
                loss = batch_hard_triplet_loss(labels, ls_embeddings, margin=params.margin,
                                            squared=params.squared)
            else:
                raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))
            acc_str = "DCG"
        elif params.loss == 'classification':
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                logits = ls_embeddings , labels = labels_one_hot))
            acc_str = "Softmax"
        else:
                raise ValueError("Loss function not recognized: {}".format(params.loss))
    # -----------------------------------------------------------
    # METRICS AND SUMMARIES

    # Summaries for training
    tf.summary.scalar(params.loss + '_loss', loss)
    
    if params.loss == "triplet" and params.triplet_strategy == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

    #Define Accuracy
    with tf.variable_scope(acc_str + "_Accuracy"):
        if acc_str == "DCG":
            accuracy = dcg_accuracy(x, ls_embeddings, params)

        else:
            out_val= tf.nn.softmax(ls_embeddings)
            
            #predicts if the output is equal to its expectation 
            correctness_of_prediction = tf.equal(
                tf.argmax(out_val, 1), tf.argmax(labels_one_hot, 1))

            #accuracy of the NN
            accuracy = tf.reduce_mean(
                tf.cast(correctness_of_prediction, tf.float32))

    tf.summary.scalar(acc_str + "_Accuracy", accuracy)

    # Define training step that minimizes the loss with the Gradient Descent optimizer
    with tf.variable_scope('Optimiser'):
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
        
        return x, labels, sess, train_op, accuracy, loss, merged, train_writer, gst
    elif mode == 'test':
        return x, embeddings, labels, accuracy

def train(params):
    x, labels, sess, train_op, accuracy, loss, merged, train_writer, gst = model_fn(params, mode='train')

    xdata, labeldata = train_input_fn()
    print('Training begins')
    start_time=time.time()
    
    for j in range (params.num_epochs):
        
        print("EPOCH NUMBER: ", j+1)
        avg_acc=0
        avg_lss=0
        
        for k in range(0, len(xdata), params.batch_size):
            current_batch_x_train = xdata[k:k+params.batch_size]
            current_batch_label_train = labeldata[k:k+params.batch_size]
            
            _, acc, lss, merg, gg = sess.run([train_op, accuracy, loss, merged, gst],
                            feed_dict = {x: current_batch_x_train, labels: current_batch_label_train})

            avg_acc+=acc
            avg_lss+=lss

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
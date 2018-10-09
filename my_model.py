import tensorflow as tf
from model.utils import Params
from model.input_fn import train_input_fn
from model.input_fn import test_input_fn
from model.fully_connected import hashing_model
from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
from model.triplet_loss import _pairwise_distances as dist
import argparse, os, time

def sup(params, mode):

    if mode == 'train':
        train(params)
    else:
        return test(params)

def model_fn(params, mode):
    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        # Compute the embeddings with the model
        x, embeddings = hashing_model()

    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    labels = tf.placeholder(tf.int64, [None, 1], 'labels')

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
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    # TODO: some other metrics like rank-1 accuracy?
    '''with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}

        if params.triplet_strategy == "batch_all":
            eval_metric_ops['fraction_positive_triplets'] = tf.metrics.mean(fraction)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
    '''

    #Compute training accuracy

    og_emb_dist = tf.add((10*tf.eye(tf.shape(x)[0], dtype = tf.float32)), dist(x, squared=True))
    tr_emb_dist = tf.add((10*tf.eye(tf.shape(x)[0], dtype = tf.float32)), dist(embeddings, squared=True))
    
    og_emb_nn = tf.argmin(og_emb_dist, axis = 0)
    tr_emb_nn = tf.argmin(tr_emb_dist, axis = 0)
    
    accuracy = tf.equal(og_emb_nn, tr_emb_nn)
    accuracy = tf.cast(accuracy, dtype = tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    tf.summary.scalar('accuracy', accuracy)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    if params.triplet_strategy == "batch_all":
        tf.summary.scalar('fraction_positive_triplets', fraction)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)
    
    global_step = tf.train.get_global_step()
    
    train_op = optimizer.minimize(loss, global_step=global_step)

    if mode == 'train':
        init = tf.global_variables_initializer()
        sess = tf.Session()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.getcwd() + '/train_writer',
                                      sess.graph)
        sess.run(init)

        return x, labels, sess, train_op, accuracy, loss, merged, train_writer
    elif mode == 'test':
        return x, embeddings, labels#, accuracy


def train(params):
    x, labels, sess, train_op, accuracy, loss, merged, train_writer = model_fn(params, mode='train')

    xdata, labeldata = train_input_fn()

    print('Training begins')
    start_time=time.time()
    jj=1

    for j in range (params.num_epochs):
        
        print("EPOCH NUMBER: ", j+1)
        avg_acc = 0
        for k in range(0, len(xdata), params.batch_size):
            current_batch_x_train = xdata[k:k+params.batch_size]
            current_batch_label_train = labeldata[k:k+params.batch_size]
            
            _, acc, lss, merg= sess.run([train_op, accuracy, loss, merged],
                            feed_dict = {x: current_batch_x_train, labels: current_batch_label_train})

            avg_acc+=acc
            #for i in range (len(o)):
            #    print('o'+ str(i),o[i])
            #    print('t'+ str(i),t[i])
            #break
            print("Batch Accuracy", acc)
        train_writer.add_summary(merg, j+1)
        print("Average accuracy= ", avg_acc * params.batch_size/len(xdata))
        print("Training loss= ", lss)
        break
    train_time=time.time() - start_time
    
    saver= tf.train.Saver()
    saver.save(sess , os.path.join(os.getcwd(), params.model_file))
    
    print("Total training time= ", train_time, "seconds")

def test(params):
    x, embeddings, labels = model_fn(params, mode = 'test')

    xdata, labeldata = test_input_fn()
    
    sess = tf.Session()
    tf.train.Saver().restore(sess , os.path.join(os.getcwd(), params.model_file))
    print ("Model restored!")
    #start_time=time.time()
    out= sess.run(embeddings, feed_dict={x:xdata, labels: labeldata})

    return out, sess


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('mode', default='test',
                    help="train/test")


if __name__ == '__main__':
    #tf.reset_default_graph()
    #tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    sup(params, args.mode)
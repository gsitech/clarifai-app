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
    og_emb = dist(x)
    tr_emb = dist(embeddings)
    difference = tf.subtract(og_emb,tr_emb)
    difference = tf.maximum(difference, -1.0 * difference)
    accuracy = tf.subtract(1.0, difference)
    accuracy = tf.reduce_mean(accuracy)

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
        sess.run(init)
        return x, labels, sess, train_op, accuracy, loss
    elif mode == 'test':
        return x, embeddings, labels#, accuracy


def train(params):
    x, labels, sess, train_op, accuracy, loss = model_fn(params, mode='train')

    xdata, labeldata = train_input_fn()

    print('Training begins')
    start_time=time.time()

    for j in range (params.num_epochs):
        
        print("EPOCH NUMBER: ", j+1)
        for k in range(0, len(xdata), params.batch_size):
            current_batch_x_train = xdata[k:k+params.batch_size]
            current_batch_label_train = labeldata[k:k+params.batch_size]

            _, acc, lss= sess.run([train_op, accuracy, loss],
                            feed_dict = {x: current_batch_x_train, labels: current_batch_label_train})

        print("Training accuracy= ", acc)
        print("Training loss= ", lss)
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
parser.add_argument('--model_dir', default='experiments\\base_model',
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
    '''
    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(lambda: train_input_fn(args.data_dir, params))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
    for key in res:
        print("{}: {}".format(key, res[key]))
    '''
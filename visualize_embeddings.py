import argparse
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from model.utils import Params
from model.input_fn import test_input_fn, get_labels_as_strings
from my_model import sup

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--sprite_filename', default='experiments/clarifai_sprite.png',
                    help="Sprite image for the projector")


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    
    # Compute embeddings on the test set
    tf.logging.info("Predicting")

    #Comment the next line to visualise original embeddings
    embeddings = sup(params, 'test')

    #Comment the next line to visualise quantised embeddings
    #embeddings, _ = test_input_fn()

    tf.logging.info("Embeddings shape: {}".format(embeddings.shape))
    labels_string = get_labels_as_strings()

    # Visualize test embeddings
    embedding_var = tf.Variable(embeddings, name = 'embedding_var')

    eval_dir = os.path.join(args.model_dir, 'eval')
    
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'embedding_var'
    
    # Specify where you find the sprite (use Create_sprite.py in the root dir to create your own sprite)
    # Copy the embedding sprite image to the eval directory
    shutil.copy2(args.sprite_filename, eval_dir)
    embedding.sprite.image_path = pathlib.Path(args.sprite_filename).name
    embedding.sprite.single_image_dim.extend([74, 74])

    # Specify where you find the metadata
    # Save the metadata file needed for Tensorboard projector
    metadata_filename = "clarifai_metadata.tsv"
    with open(os.path.join(eval_dir, metadata_filename), 'w') as f:
        f.write('index\tlabel\n')
        for i in range (len(labels_string)):
            f.write('{}\t{}\n'.format(i,labels_string[i]))
    embedding.metadata_path = metadata_filename
    
    # Visualise the embeddings
    projector.visualize_embeddings(tf.summary.FileWriter(eval_dir), config)
    
    with tf.Session() as ses:
        sav = tf.train.Saver([embedding_var])
        ses.run(embedding_var.initializer)
        sav.save(ses, os.path.join(eval_dir, "embeddings.ckpt"), 1)
import tensorflow as tf


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)
    
    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16
    
        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

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


def dcg_accuracy(og_embed, tr_embed, params):

    #Compute pairwise distances
    og_embed_pd = _pairwise_distances(og_embed, squared=True)
    tr_embed_pd = _pairwise_distances(tr_embed, squared=True)
    
    #Acquire top k values
    og_embed_pd_top_k, _ = tf.nn.top_k(og_embed_pd, k = tf.shape(og_embed)[0], sorted=False, name = 'DCG_og_emb')
    _,        indices_tr = tf.nn.top_k(tr_embed_pd, k = tf.shape(og_embed)[0], sorted=False, name = 'DCG_tr_emb')
    
    dcg_og = dcg(og_embed_pd_top_k)
    dcg_tr = dcg(og_embed_pd, indices_tr)

    #Smallest possible value of DCG
    dcg_sm = dcg( tf.contrib.framework.sort(values = og_embed_pd, axis = 1) )

    #DCG Accuracy
    dcg_ac = (dcg_tr - dcg_sm) / (dcg_og - dcg_sm)

    return tf.reduce_mean(dcg_ac)


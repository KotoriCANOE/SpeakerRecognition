import tensorflow as tf

def _pairwise_distances(embeddings, squared=False):
    # get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    # get squared L2 norm for each embedding.
    # we can just take the diagonal of dot_product
    # this also provides more numerical stability (the diagonal of the result will be exactly 0)
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)
    # compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2 - a <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    # compensate for computation errors to avoid negative values
    distances = tf.maximum(distances, 0.0)
    if not squared:
        # added a small epsilon to avoid infinite gradient when distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances += mask * 1e-16
        distances = tf.sqrt(distances)
        # correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances *= 1.0 - mask
    # return
    return distances

def _get_anchor_positive_triplet_mask(labels):
    # check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    # check if labels{i] == labels[j]
    # broadcasting (1, batch_size) and (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    # combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)
    return mask

def _get_anchor_negative_triplet_mask(labels):
    # check if labels[i] != labels[k]
    # broadcasting (1, batch_size) and (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask = tf.logical_not(labels_equal)
    return mask

def batch_hard(labels, embeddings, margin, squared=False):
    # pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared)
    # for each anchor, get the hardest positive
    # first, we need to get a mask for every valid positive (they should have the same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)
    # put elements where (a, p) are not valid to 0
    # (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    # for each anchor, get the hardest negative
    # first, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)
    # we add the maximum value in each row to the invalid negatives
    # (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    # combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)
    # get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)
    # return
    return triplet_loss

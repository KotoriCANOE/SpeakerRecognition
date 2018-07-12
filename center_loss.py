import tensorflow as tf

def get_center_loss(embeddings, labels, num_classes, alpha=0.5):
    length = embeddings.shape[-1]
    centers = tf.get_variable('centers', [num_classes, length], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    # get center of each label
    centers_batch = tf.gather(centers, labels)
    loss = tf.nn.l2_loss(embeddings - centers_batch)
    diff = centers_batch - embeddings
    # number of the samples of each label
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff /= tf.cast((1 + appear_times), tf.float32)
    diff *= alpha
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    # return
    return loss, centers, centers_update_op

import tensorflow as tf

def get_center_loss(embeddings, labels, num_classes, alpha=0.5):
    dtype = tf.float32
    length = embeddings.shape[-1]
    centers = tf.get_variable('centers', [num_classes, length], dtype,
        tf.initializers.zeros(dtype), trainable=False)
    labels = tf.reshape(labels, [-1])
    # get center of each label
    centers_batch = tf.gather(centers, labels)
    # loss
    diff = centers_batch - embeddings
    loss = tf.nn.l2_loss(diff)
    # number of the samples of each label
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff /= tf.cast((1 + appear_times), tf.float32)
    diff *= alpha
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    # return
    return loss, centers, centers_update_op

def get_center_loss_unbias(embeddings, labels, num_classes, decay=0.9):
    dtype = tf.float32
    length = embeddings.shape[-1]
    centers = tf.get_variable('centers', [num_classes, length], dtype,
        tf.initializers.zeros(dtype), trainable=False)
    steps = tf.get_variable('steps', [num_classes], tf.int32,
        tf.initializers.zeros(tf.int32), trainable=False)
    labels = tf.reshape(labels, [-1])
    # get center of each label
    centers_batch = tf.gather(centers, labels)
    bias_fix = 1 - decay ** tf.cast(tf.gather(steps, labels), dtype) + 1e-8
    centers_batch_unbias = centers_batch / tf.reshape(bias_fix, [-1, 1])
    # loss
    loss = tf.nn.l2_loss(centers_batch_unbias - embeddings)
    # number of the samples of each label
    unique_labels, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff = (centers_batch - embeddings) / tf.cast(appear_times, dtype)
    # update
    update_ops = []
    with tf.control_dependencies([centers_batch_unbias]):
        update_ops.append(tf.scatter_sub(centers, labels, (1 - decay) * diff))
        update_ops.append(tf.scatter_add(steps, unique_labels, 1))
    # return
    return loss, centers, update_ops


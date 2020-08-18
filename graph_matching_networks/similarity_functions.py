import tensorflow as tf

def pairwise_euclidean_similarity(x, y):
  """Compute the pairwise Euclidean similarity between x and y.

  This function computes the following similarity value between each pair of x_i
  and y_j: s(x_i, y_j) = -|x_i - y_j|^2.

  Args:
    x: NxD float tensor.
    y: MxD float tensor.

  Returns:
    s: NxM float tensor, the pairwise euclidean similarity.
  """
  s = 2 * tf.matmul(x, y, transpose_b=True)
  diag_x = tf.reduce_sum(x * x, axis=-1, keepdims=True)
  diag_y = tf.reshape(tf.reduce_sum(y * y, axis=-1), (1, -1))
  return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
  """Compute the dot product similarity between x and y.

  This function computes the following similarity value between each pair of x_i
  and y_j: s(x_i, y_j) = x_i^T y_j.

  Args:
    x: NxD float tensor.
    y: MxD float tensor.

  Returns:
    s: NxM float tensor, the pairwise dot product similarity.
  """
  return tf.matmul(x, y, transpose_b=True)


def pairwise_cosine_similarity(x, y):
  """Compute the cosine similarity between x and y.

  This function computes the following similarity value between each pair of x_i
  and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

  Args:
    x: NxD float tensor.
    y: MxD float tensor.

  Returns:
    s: NxM float tensor, the pairwise cosine similarity.
  """
  x = tf.nn.l2_normalize(x, axis=-1)
  y = tf.nn.l2_normalize(y, axis=-1)
  return tf.matmul(x, y, transpose_b=True)


PAIRWISE_SIMILARITY_FUNCTION = {
    'euclidean': pairwise_euclidean_similarity,
    'dotproduct': pairwise_dot_product_similarity,
    'cosine': pairwise_cosine_similarity,
}


def get_pairwise_similarity(name):
  """Get pairwise similarity metric by name.

  Args:
    name: string, name of the similarity metric, one of {dot-product, cosine,
      euclidean}.

  Returns:
    similarity: a (x, y) -> sim function.

  Raises:
    ValueError: if name is not supported.
  """
  if name not in PAIRWISE_SIMILARITY_FUNCTION:
    raise ValueError('Similarity metric name "%s" not supported.' % name)
  else:
    return PAIRWISE_SIMILARITY_FUNCTION[name]


def exact_hamming_similarity(x, y):
  """Compute the binary Hamming similarity."""
  match = tf.cast(tf.equal(x > 0, y > 0), dtype=tf.float32)
  return tf.reduce_mean(match, axis=1)


def compute_similarity(config, x, y):
  """Compute the distance between x and y vectors.

  The distance will be computed based on the training loss type.

  Args:
    config: a config dict.
    x: [n_examples, feature_dim] float tensor.
    y: [n_examples, feature_dim] float tensor.

  Returns:
    dist: [n_examples] float tensor.

  Raises:
    ValueError: if loss type is not supported.
  """
  if config['training']['loss'] == 'margin':
    # similarity is negative distance
    return -euclidean_distance(x, y)
  elif config['training']['loss'] == 'hamming':
    return exact_hamming_similarity(x, y)
  else:
    raise ValueError('Unknown loss type %s' % config['training']['loss'])


def auc(scores, labels, **auc_args):
  """Compute the AUC for pair classification.

  See `tf.metrics.auc` for more details about this metric.

  Args:
    scores: [n_examples] float.  Higher scores mean higher preference of being
      assigned the label of +1.
    labels: [n_examples] int.  Labels are either +1 or -1.
    **auc_args: other arguments that can be used by `tf.metrics.auc`.

  Returns:
    auc: the area under the ROC curve.
  """
  scores_max = tf.reduce_max(scores)
  scores_min = tf.reduce_min(scores)
  # normalize scores to [0, 1] and add a small epislon for safety
  scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

  labels = (labels + 1) / 2
  # The following code should be used according to the tensorflow official
  # documentation:
  # value, _ = tf.metrics.auc(labels, scores, **auc_args)

  # However `tf.metrics.auc` is currently (as of July 23, 2019) buggy so we have
  # to use the following:
  _, value = tf.metrics.auc(labels, scores, **auc_args)
  return value


def euclidean_distance(x, y):
  """This is the squared Euclidean distance."""
  return tf.reduce_sum((x - y)**2, axis=-1)


def approximate_hamming_similarity(x, y):
  """Approximate Hamming similarity."""
  return tf.reduce_mean(tf.tanh(x) * tf.tanh(y), axis=1)


def pairwise_loss(x, y, labels, loss_type='margin', margin=1.0):
  """Compute pairwise loss.

  Args:
    x: [N, D] float tensor, representations for N examples.
    y: [N, D] float tensor, representations for another N examples.
    labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
      and y[i] are similar, and -1 otherwise.
    loss_type: margin or hamming.
    margin: float scalar, margin for the margin loss.

  Returns:
    loss: [N] float tensor.  Loss for each pair of representations.
  """
  labels = tf.cast(labels, x.dtype)
  if loss_type == 'margin':
    return tf.nn.relu(margin - labels * (1 - euclidean_distance(x, y)))
  elif loss_type == 'hamming':
    return 0.25 * (labels - approximate_hamming_similarity(x, y))**2
  else:
    raise ValueError('Unknown loss_type %s' % loss_type)
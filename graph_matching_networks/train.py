import numpy as np
import random, copy, math
import time
import collections

import tensorflow as tf

from geo_data_loader import read_data, GraphGeoDataset, GraphData
from model import *


def fill_feed_dict(placeholders, batch):
  """Create a feed dict for the given batch of data.

  Args:
    placeholders: a dict of placeholders.
    batch: a batch of data, should be either a single `GraphData` instance for
      triplet training, or a tuple of (graphs, labels) for pairwise training.

  Returns:
    feed_dict: a feed_dict that can be used in a session run call.
  """
  if isinstance(batch, GraphData):
    graphs = batch
    labels = None
  else:
    graphs, labels = batch

  feed_dict = {
      placeholders['node_features']: graphs.node_features,
      placeholders['edge_features']: graphs.edge_features,
      placeholders['from_idx']: graphs.from_idx,
      placeholders['to_idx']: graphs.to_idx,
      placeholders['graph_idx']: graphs.graph_idx,
  }
  if labels is not None:
    feed_dict[placeholders['labels']] = labels
  return feed_dict


def evaluate(sess, eval_metrics, placeholders, validation_set, batch_size):
  """Evaluate model performance on the given validation set.

  Args:
    sess: a `tf.Session` instance used to run the computation.
    eval_metrics: a dict containing two tensors 'pair_auc' and 'triplet_acc'.
    placeholders: a placeholder dict.
    validation_set: a `GraphSimilarityDataset` instance, calling `pairs` and
      `triplets` functions with `batch_size` creates iterators over a finite
      sequence of batches to evaluate on.
    batch_size: number of batches to use for each session run call.

  Returns:
    metrics: a dict of metric name => value mapping.
  """
  accumulated_pair_auc = []
  for batch in validation_set.pairs(batch_size):
    feed_dict = fill_feed_dict(placeholders, batch)
    pair_auc = sess.run(eval_metrics['pair_auc'], feed_dict=feed_dict)
    accumulated_pair_auc.append(pair_auc)

  accumulated_triplet_acc = []
  for batch in validation_set.triplets(batch_size):
    feed_dict = fill_feed_dict(placeholders, batch)
    triplet_acc = sess.run(eval_metrics['triplet_acc'], feed_dict=feed_dict)
    accumulated_triplet_acc.append(triplet_acc)

  return {
      'pair_auc': np.mean(accumulated_pair_auc),
      'triplet_acc': np.mean(accumulated_triplet_acc),
  }

def get_default_config():
  """The default configs."""
  node_state_dim = 32
  graph_rep_dim = 128
  graph_embedding_net_config = dict(
      node_state_dim=node_state_dim,
      edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
      node_hidden_sizes=[node_state_dim * 2],
      n_prop_layers=5,
      # set to False to not share parameters across message passing layers
      share_prop_params=True,
      # initialize message MLP with small parameter weights to prevent
      # aggregated message vectors blowing up, alternatively we could also use
      # e.g. layer normalization to keep the scale of these under control.
      edge_net_init_scale=0.1,
      # other types of update like `mlp` and `residual` can also be used here.
      node_update_type='gru',
      # set to False if your graph already contains edges in both directions.
      use_reverse_direction=True,
      # set to True if your graph is directed
      reverse_dir_param_different=False,
      # we didn't use layer norm in our experiments but sometimes this can help.
      layer_norm=False)
  graph_matching_net_config = graph_embedding_net_config.copy()
  graph_matching_net_config['similarity'] = 'dotproduct'

  return dict(
      encoder=dict(
          node_hidden_sizes=[node_state_dim],
          edge_hidden_sizes=None),
      aggregator=dict(
          node_hidden_sizes=[graph_rep_dim],
          graph_transform_sizes=[graph_rep_dim],
          gated=True,
          aggregation_type='sum'),
      graph_embedding_net=graph_embedding_net_config,
      graph_matching_net=graph_matching_net_config,
      # Set to `embedding` to use the graph embedding net.
      model_type='matching',
      data=dict(
          problem='graph_edit_distance',
          dataset_params=dict(
              # always generate graphs with 20 nodes and p_edge=0.2.
              n_nodes_range=[20, 20],
              p_edge_range=[0.2, 0.2],
              n_changes_positive=1,
              n_changes_negative=2,
              validation_dataset_size=1000)),
      training=dict(
          batch_size=20,
          learning_rate=1e-3,
          mode='pair',
          loss='margin',
          margin=1.0,
          # A small regularizer on the graph vector scales to avoid the graph
          # vectors blowing up.  If numerical issues is particularly bad in the
          # model we can add `snt.LayerNorm` to the outputs of each layer, the
          # aggregated messages and aggregated node representations to
          # keep the network activation scale in a reasonable range.
          graph_vec_regularizer_weight=1e-6,
          # Add gradient clipping to avoid large gradients.
          clip_value=10.0,
          # Increase this to train longer.
          n_training_steps=10000,
          # Print training information every this many training steps.
          print_after=100,
          # Evaluate on validation set every `eval_after * print_after` steps.
          eval_after=10),
      evaluation=dict(
          batch_size=20),
      seed=8,
      )


if __name__ == '__main__':
    config = get_default_config()

    # Let's just run for a small number of training steps.
    config['training']['n_training_steps'] = 10000
    # Set random seeds
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)

    IGN04,_ = read_data('IGN10',  # TODO fix this to make automatic
                      with_classes=True,
                      prefer_attr_nodes=True,
                      prefer_attr_edges=False,
                      produce_labels_nodes=False,
                      as_graphs=True,
                      is_symmetric=1,
                      path='/home/margokat/projects/graph_nn/data/IGN_all_clean/%s/' % 'IGN_2010', with_node_posistions=False)

    IGN19, _ = read_data('IGN19',  # TODO fix this to make automatic
                      with_classes=True,
                      prefer_attr_nodes=True,
                      prefer_attr_edges=False,
                      produce_labels_nodes=False,
                      as_graphs=True,
                      is_symmetric=1,
                      path='/home/margokat/projects/graph_nn/data/IGN_all_clean/%s/' % 'IGN_2019',
                      with_node_posistions=False)


    training_set = GraphGeoDataset(data_first=IGN04,  data_second=IGN19, permute=True)

    if config['training']['mode'] == 'pair':
        training_data_iter = training_set.pairs(config['training']['batch_size'])
        first_batch_graphs, _ = next(training_data_iter)
    else:
        training_data_iter = training_set.triplets(config['training']['batch_size'])
        first_batch_graphs = next(training_data_iter)

    node_feature_dim = first_batch_graphs.node_features.shape[-1]
    edge_feature_dim = first_batch_graphs.edge_features.shape[-1]

    tensors, placeholders, model = build_model(
        config, node_feature_dim, edge_feature_dim)

    accumulated_metrics = collections.defaultdict(list)

    t_start = time.time()

    init_ops = (tf.global_variables_initializer(),
                tf.local_variables_initializer())

    # If we already have a session instance, close it and start a new one
    if 'sess' in globals():
        sess.close()

    # We will need to keep this session instance around for e.g. visualization.
    # But you should probably wrap it in a `with tf.Session() sess:` context if you
    # want to use the code elsewhere.
    sess = tf.Session()
    sess.run(init_ops)

    # use xrange here if you are still on python 2
    for i_iter in range(config['training']['n_training_steps']):
        batch = next(training_data_iter)
        _, train_metrics = sess.run(
            [tensors['train_step'], tensors['metrics']['training']],
            feed_dict=fill_feed_dict(placeholders, batch))

        # accumulate over minibatches to reduce variance in the training metrics
        for k, v in train_metrics.items():
            accumulated_metrics[k].append(v)

        if (i_iter + 1) % config['training']['print_after'] == 0:
            metrics_to_print = {
                k: np.mean(v) for k, v in accumulated_metrics.items()}
            info_str = ', '.join(
                ['%s %.4f' % (k, v) for k, v in metrics_to_print.items()])
            # reset the metrics
            accumulated_metrics = collections.defaultdict(list)

            # if ((i_iter + 1) // config['training']['print_after'] %
            #         config['training']['eval_after'] == 0):
            #     eval_metrics = evaluate(
            #         sess, tensors['metrics']['validation'], placeholders,
            #         validation_set, config['evaluation']['batch_size'])
            #     info_str += ', ' + ', '.join(
            #         ['%s %.4f' % ('val/' + k, v) for k, v in eval_metrics.items()])

            print('iter %d, %s, time %.2fs' % (
                i_iter + 1, info_str, time.time() - t_start))
            t_start = time.time()
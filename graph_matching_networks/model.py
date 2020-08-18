import sonnet as snt
import tensorflow as tf
from similarity_functions import *
class GraphEncoder(snt.AbstractModule):
  """Encoder module that projects node and edge features to some embeddings."""

  def __init__(self,
               node_hidden_sizes=None,
               edge_hidden_sizes=None,
               name='graph-encoder'):
    """Constructor.

    Args:
      node_hidden_sizes: if provided should be a list of ints, hidden sizes of
        node encoder network, the last element is the size of the node outputs.
        If not provided, node features will pass through as is.
      edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
        edge encoder network, the last element is the size of the edge outptus.
        If not provided, edge features will pass through as is.
      name: name of this module.
    """
    super(GraphEncoder, self).__init__(name=name)

    # this also handles the case of an empty list
    self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
    self._edge_hidden_sizes = edge_hidden_sizes

  def _build(self, node_features, edge_features=None):
    """Encode node and edge features.

    Args:
      node_features: [n_nodes, node_feat_dim] float tensor.
      edge_features: if provided, should be [n_edges, edge_feat_dim] float
        tensor.

    Returns:
      node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
      edge_outputs: if edge_features is not None and edge_hidden_sizes is not
        None, this is [n_edges, edge_embedding_dim] float tensor, edge
        embeddings; otherwise just the input edge_features.
    """
    if self._node_hidden_sizes is None:
      node_outputs = node_features
    else:
      node_outputs = snt.nets.MLP(
          self._node_hidden_sizes, name='node-feature-mlp')(node_features)

    if edge_features is None or self._edge_hidden_sizes is None:
      edge_outputs = edge_features
    else:
      edge_outputs = snt.nets.MLP(
          self._edge_hidden_sizes, name='edge-feature-mlp')(edge_features)

    return node_outputs, edge_outputs


def graph_prop_once(node_states,
                    from_idx,
                    to_idx,
                    message_net,
                    aggregation_module=tf.math.unsorted_segment_sum,
                    edge_features=None):
  """One round of propagation (message passing) in a graph.

  Args:
    node_states: [n_nodes, node_state_dim] float tensor, node state vectors, one
      row for each node.
    from_idx: [n_edges] int tensor, index of the from nodes.
    to_idx: [n_edges] int tensor, index of the to nodes.
    message_net: a network that maps concatenated edge inputs to message
      vectors.
    aggregation_module: a module that aggregates messages on edges to aggregated
      messages for each node.  Should be a callable and can be called like the
      following,
      `aggregated_messages = aggregation_module(messages, to_idx, n_nodes)`,
      where messages is [n_edges, edge_message_dim] tensor, to_idx is the index
      of the to nodes, i.e. where each message should go to, and n_nodes is an
      int which is the number of nodes to aggregate into.
    edge_features: if provided, should be a [n_edges, edge_feature_dim] float
      tensor, extra features for each edge.

  Returns:
    aggregated_messages: an [n_nodes, edge_message_dim] float tensor, the
      aggregated messages, one row for each node.
  """
  from_states = tf.gather(node_states, from_idx)
  to_states = tf.gather(node_states, to_idx)

  edge_inputs = [from_states, to_states]
  if edge_features is not None:
    edge_inputs.append(edge_features)

  edge_inputs = tf.concat(edge_inputs, axis=-1)
  messages = message_net(edge_inputs)

  return aggregation_module(messages, to_idx, tf.shape(node_states)[0])


class GraphPropLayer(snt.AbstractModule):
  """Implementation of a graph propagation (message passing) layer."""

  def __init__(self,
               node_state_dim,
               edge_hidden_sizes,
               node_hidden_sizes,
               edge_net_init_scale=0.1,
               node_update_type='residual',
               use_reverse_direction=True,
               reverse_dir_param_different=True,
               layer_norm=False,
               name='graph-net'):
    """Constructor.

    Args:
      node_state_dim: int, dimensionality of node states.
      edge_hidden_sizes: list of ints, hidden sizes for the edge message
        net, the last element in the list is the size of the message vectors.
      node_hidden_sizes: list of ints, hidden sizes for the node update
        net.
      edge_net_init_scale: initialization scale for the edge networks.  This
        is typically set to a small value such that the gradient does not blow
        up.
      node_update_type: type of node updates, one of {mlp, gru, residual}.
      use_reverse_direction: set to True to also propagate messages in the
        reverse direction.
      reverse_dir_param_different: set to True to have the messages computed
        using a different set of parameters than for the forward direction.
      layer_norm: set to True to use layer normalization in a few places.
      name: name of this module.
    """
    super(GraphPropLayer, self).__init__(name=name)

    self._node_state_dim = node_state_dim
    self._edge_hidden_sizes = edge_hidden_sizes[:]

    # output size is node_state_dim
    self._node_hidden_sizes = node_hidden_sizes[:] + [node_state_dim]
    self._edge_net_init_scale = edge_net_init_scale
    self._node_update_type = node_update_type

    self._use_reverse_direction = use_reverse_direction
    self._reverse_dir_param_different = reverse_dir_param_different

    self._layer_norm = layer_norm

  def _compute_aggregated_messages(
      self, node_states, from_idx, to_idx, edge_features=None):
    """Compute aggregated messages for each node.

    Args:
      node_states: [n_nodes, input_node_state_dim] float tensor, node states.
      from_idx: [n_edges] int tensor, from node indices for each edge.
      to_idx: [n_edges] int tensor, to node indices for each edge.
      edge_features: if not None, should be [n_edges, edge_embedding_dim]
        tensor, edge features.

    Returns:
      aggregated_messages: [n_nodes, aggregated_message_dim] float tensor, the
        aggregated messages for each node.
    """
    self._message_net = snt.nets.MLP(
        self._edge_hidden_sizes,
        initializers={
            'w': tf.variance_scaling_initializer(
                scale=self._edge_net_init_scale),
            'b': tf.zeros_initializer()},
        name='message-mlp')

    aggregated_messages = graph_prop_once(
        node_states,
        from_idx,
        to_idx,
        self._message_net,
        aggregation_module=tf.math.unsorted_segment_sum,
        edge_features=edge_features)

    # optionally compute message vectors in the reverse direction
    if self._use_reverse_direction:
      if self._reverse_dir_param_different:
        self._reverse_message_net = snt.nets.MLP(
            self._edge_hidden_sizes,
            initializers={
                'w': tf.variance_scaling_initializer(
                    scale=self._edge_net_init_scale),
                'b': tf.zeros_initializer()},
            name='reverse-message-mlp')
      else:
        self._reverse_message_net = self._message_net

      reverse_aggregated_messages = graph_prop_once(
          node_states,
          to_idx,
          from_idx,
          self._reverse_message_net,
          aggregation_module=tf.math.unsorted_segment_sum,
          edge_features=edge_features)

      aggregated_messages += reverse_aggregated_messages

    if self._layer_norm:
      aggregated_messages = snt.LayerNorm()(aggregated_messages)

    return aggregated_messages

  def _compute_node_update(self,
                           node_states,
                           node_state_inputs,
                           node_features=None):
    """Compute node updates.

    Args:
      node_states: [n_nodes, node_state_dim] float tensor, the input node
        states.
      node_state_inputs: a list of tensors used to compute node updates.  Each
        element tensor should have shape [n_nodes, feat_dim], where feat_dim can
        be different.  These tensors will be concatenated along the feature
        dimension.
      node_features: extra node features if provided, should be of size
        [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
        different types of skip connections.

    Returns:
      new_node_states: [n_nodes, node_state_dim] float tensor, the new node
        state tensor.

    Raises:
      ValueError: if node update type is not supported.
    """
    if self._node_update_type in ('mlp', 'residual'):
      node_state_inputs.append(node_states)
    if node_features is not None:
      node_state_inputs.append(node_features)

    if len(node_state_inputs) == 1:
      node_state_inputs = node_state_inputs[0]
    else:
      node_state_inputs = tf.concat(node_state_inputs, axis=-1)

    if self._node_update_type == 'gru':
      _, new_node_states = snt.GRU(self._node_state_dim)(
          node_state_inputs, node_states)
      return new_node_states
    else:
      mlp_output = snt.nets.MLP(
          self._node_hidden_sizes, name='node-mlp')(node_state_inputs)
      if self._layer_norm:
        mlp_output = snt.LayerNorm()(mlp_output)
      if self._node_update_type == 'mlp':
        return mlp_output
      elif self._node_update_type == 'residual':
        return node_states + mlp_output
      else:
        raise ValueError('Unknown node update type %s' % self._node_update_type)

  def _build(self,
             node_states,
             from_idx,
             to_idx,
             edge_features=None,
             node_features=None):
    """Run one propagation step.

    Args:
      node_states: [n_nodes, input_node_state_dim] float tensor, node states.
      from_idx: [n_edges] int tensor, from node indices for each edge.
      to_idx: [n_edges] int tensor, to node indices for each edge.
      edge_features: if not None, should be [n_edges, edge_embedding_dim]
        tensor, edge features.
      node_features: extra node features if provided, should be of size
        [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
        different types of skip connections.

    Returns:
      node_states: [n_nodes, node_state_dim] float tensor, new node states.
    """
    aggregated_messages = self._compute_aggregated_messages(
        node_states, from_idx, to_idx, edge_features=edge_features)

    return self._compute_node_update(node_states,
                                     [aggregated_messages],
                                     node_features=node_features)

AGGREGATION_TYPE = {
    'sum': tf.unsorted_segment_sum,
    'mean': tf.unsorted_segment_mean,
    'sqrt_n': tf.unsorted_segment_sqrt_n,
    'max': tf.unsorted_segment_max,
}


class GraphAggregator(snt.AbstractModule):
  """This module computes graph representations by aggregating from parts."""

  def __init__(self,
               node_hidden_sizes,
               graph_transform_sizes=None,
               gated=True,
               aggregation_type='sum',
               name='graph-aggregator'):
    """Constructor.

    Args:
      node_hidden_sizes: the hidden layer sizes of the node transformation nets.
        The last element is the size of the aggregated graph representation.
      graph_transform_sizes: sizes of the transformation layers on top of the
        graph representations.  The last element of this list is the final
        dimensionality of the output graph representations.
      gated: set to True to do gated aggregation, False not to.
      aggregation_type: one of {sum, max, mean, sqrt_n}.
      name: name of this module.
    """
    super(GraphAggregator, self).__init__(name=name)

    self._node_hidden_sizes = node_hidden_sizes
    self._graph_transform_sizes = graph_transform_sizes
    self._graph_state_dim = node_hidden_sizes[-1]
    self._gated = gated
    self._aggregation_type = aggregation_type
    self._aggregation_op = AGGREGATION_TYPE[aggregation_type]

  def _build(self, node_states, graph_idx, n_graphs):
    """Compute aggregated graph representations.

    Args:
      node_states: [n_nodes, node_state_dim] float tensor, node states of a
        batch of graphs concatenated together along the first dimension.
      graph_idx: [n_nodes] int tensor, graph ID for each node.
      n_graphs: integer, number of graphs in this batch.

    Returns:
      graph_states: [n_graphs, graph_state_dim] float tensor, graph
        representations, one row for each graph.
    """
    node_hidden_sizes = self._node_hidden_sizes
    if self._gated:
      node_hidden_sizes[-1] = self._graph_state_dim * 2

    node_states_g = snt.nets.MLP(
        node_hidden_sizes, name='node-state-g-mlp')(node_states)

    if self._gated:
      gates = tf.nn.sigmoid(node_states_g[:, :self._graph_state_dim])
      node_states_g = node_states_g[:, self._graph_state_dim:] * gates

    graph_states = self._aggregation_op(node_states_g, graph_idx, n_graphs)

    # unsorted_segment_max does not handle empty graphs in the way we want
    # it assigns the lowest possible float to empty segments, we want to reset
    # them to zero.
    if self._aggregation_type == 'max':
      # reset everything that's smaller than -1e5 to 0.
      graph_states *= tf.cast(graph_states > -1e5, tf.float32)

    # transform the reduced graph states further

    # pylint: disable=g-explicit-length-test
    if (self._graph_transform_sizes is not None and
        len(self._graph_transform_sizes) > 0):
      graph_states = snt.nets.MLP(
          self._graph_transform_sizes, name='graph-transform-mlp')(graph_states)

    return graph_states


class GraphEmbeddingNet(snt.AbstractModule):
  """A graph to embedding mapping network."""

  def __init__(self,
               encoder,
               aggregator,
               node_state_dim,
               edge_hidden_sizes,
               node_hidden_sizes,
               n_prop_layers,
               share_prop_params=False,
               edge_net_init_scale=0.1,
               node_update_type='residual',
               use_reverse_direction=True,
               reverse_dir_param_different=True,
               layer_norm=False,
               name='graph-embedding-net'):
    """Constructor.

    Args:
      encoder: GraphEncoder, encoder that maps features to embeddings.
      aggregator: GraphAggregator, aggregator that produces graph
        representations.
      node_state_dim: dimensionality of node states.
      edge_hidden_sizes: sizes of the hidden layers of the edge message nets.
      node_hidden_sizes: sizes of the hidden layers of the node update nets.
      n_prop_layers: number of graph propagation layers.
      share_prop_params: set to True to share propagation parameters across all
        graph propagation layers, False not to.
      edge_net_init_scale: scale of initialization for the edge message nets.
      node_update_type: type of node updates, one of {mlp, gru, residual}.
      use_reverse_direction: set to True to also propagate messages in the
        reverse direction.
      reverse_dir_param_different: set to True to have the messages computed
        using a different set of parameters than for the forward direction.
      layer_norm: set to True to use layer normalization in a few places.
      name: name of this module.
    """
    super(GraphEmbeddingNet, self).__init__(name=name)

    self._encoder = encoder
    self._aggregator = aggregator
    self._node_state_dim = node_state_dim
    self._edge_hidden_sizes = edge_hidden_sizes
    self._node_hidden_sizes = node_hidden_sizes
    self._n_prop_layers = n_prop_layers
    self._share_prop_params = share_prop_params
    self._edge_net_init_scale = edge_net_init_scale
    self._node_update_type = node_update_type
    self._use_reverse_direction = use_reverse_direction
    self._reverse_dir_param_different = reverse_dir_param_different
    self._layer_norm = layer_norm

    self._prop_layers = []
    self._layer_class = GraphPropLayer

  def _build_layer(self, layer_id):
    """Build one layer in the network."""
    return self._layer_class(
        self._node_state_dim,
        self._edge_hidden_sizes,
        self._node_hidden_sizes,
        edge_net_init_scale=self._edge_net_init_scale,
        node_update_type=self._node_update_type,
        use_reverse_direction=self._use_reverse_direction,
        reverse_dir_param_different=self._reverse_dir_param_different,
        layer_norm=self._layer_norm,
        name='graph-prop-%d' % layer_id)

  def _apply_layer(self,
                   layer,
                   node_states,
                   from_idx,
                   to_idx,
                   graph_idx,
                   n_graphs,
                   edge_features):
    """Apply one layer on the given inputs."""
    del graph_idx, n_graphs
    return layer(node_states, from_idx, to_idx, edge_features=edge_features)

  def _build(self,
             node_features,
             edge_features,
             from_idx,
             to_idx,
             graph_idx,
             n_graphs):
    """Compute graph representations.

    Args:
      node_features: [n_nodes, node_feat_dim] float tensor.
      edge_features: [n_edges, edge_feat_dim] float tensor.
      from_idx: [n_edges] int tensor, index of the from node for each edge.
      to_idx: [n_edges] int tensor, index of the to node for each edge.
      graph_idx: [n_nodes] int tensor, graph id for each node.
      n_graphs: int, number of graphs in the batch.

    Returns:
      graph_representations: [n_graphs, graph_representation_dim] float tensor,
        graph representations.
    """
    if len(self._prop_layers) < self._n_prop_layers:
      # build the layers
      for i in range(self._n_prop_layers):
        if i == 0 or not self._share_prop_params:
          layer = self._build_layer(i)
        else:
          layer = self._prop_layers[0]
        self._prop_layers.append(layer)

    node_features, edge_features = self._encoder(node_features, edge_features)
    node_states = node_features

    layer_outputs = [node_states]

    for layer in self._prop_layers:
      # node_features could be wired in here as well, leaving it out for now as
      # it is already in the inputs
      node_states = self._apply_layer(
          layer,
          node_states,
          from_idx,
          to_idx,
          graph_idx,
          n_graphs,
          edge_features)
      layer_outputs.append(node_states)

    # these tensors may be used e.g. for visualization
    self._layer_outputs = layer_outputs
    return self._aggregator(node_states, graph_idx, n_graphs)

  def reset_n_prop_layers(self, n_prop_layers):
    """Set n_prop_layers to the provided new value.

    This allows us to train with certain number of propagation layers and
    evaluate with a different number of propagation layers.

    This only works if n_prop_layers is smaller than the number used for
    training, or when share_prop_params is set to True, in which case this can
    be arbitrarily large.

    Args:
      n_prop_layers: the new number of propagation layers to set.
    """
    self._n_prop_layers = n_prop_layers

  @property
  def n_prop_layers(self):
    return self._n_prop_layers

  def get_layer_outputs(self):
    """Get the outputs at each layer."""
    if hasattr(self, '_layer_outputs'):
      return self._layer_outputs
    else:
      raise ValueError('No layer outputs available.')

def compute_cross_attention(x, y, sim):
  """Compute cross attention.

  x_i attend to y_j:
  a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
  y_j attend to x_i:
  a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))
  attention_x = sum_j a_{i->j} y_j
  attention_y = sum_i a_{j->i} x_i

  Args:
    x: NxD float tensor.
    y: MxD float tensor.
    sim: a (x, y) -> similarity function.

  Returns:
    attention_x: NxD float tensor.
    attention_y: NxD float tensor.
  """
  a = sim(x, y)
  a_x = tf.nn.softmax(a, axis=1)  # i->j
  a_y = tf.nn.softmax(a, axis=0)  # j->i
  attention_x = tf.matmul(a_x, y)
  attention_y = tf.matmul(a_y, x, transpose_a=True)
  return attention_x, attention_y


def batch_block_pair_attention(data,
                               block_idx,
                               n_blocks,
                               similarity='dotproduct'):
  """Compute batched attention between pairs of blocks.

  This function partitions the batch data into blocks according to block_idx.
  For each pair of blocks, x = data[block_idx == 2i], and
  y = data[block_idx == 2i+1], we compute

  x_i attend to y_j:
  a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
  y_j attend to x_i:
  a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

  and

  attention_x = sum_j a_{i->j} y_j
  attention_y = sum_i a_{j->i} x_i.

  Args:
    data: NxD float tensor.
    block_idx: N-dim int tensor.
    n_blocks: integer.
    similarity: a string, the similarity metric.

  Returns:
    attention_output: NxD float tensor, each x_i replaced by attention_x_i.

  Raises:
    ValueError: if n_blocks is not an integer or not a multiple of 2.
  """
  if not isinstance(n_blocks, int):
    raise ValueError('n_blocks (%s) has to be an integer.' % str(n_blocks))

  if n_blocks % 2 != 0:
    raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_blocks)

  sim = get_pairwise_similarity(similarity)

  results = []

  # This is probably better than doing boolean_mask for each i
  partitions = tf.dynamic_partition(data, block_idx, n_blocks)

  # It is rather complicated to allow n_blocks be a tf tensor and do this in a
  # dynamic loop, and probably unnecessary to do so.  Therefore we are
  # restricting n_blocks to be a integer constant here and using the plain for
  # loop.
  for i in range(0, n_blocks, 2):
    x = partitions[i]
    y = partitions[i + 1]
    attention_x, attention_y = compute_cross_attention(x, y, sim)
    results.append(attention_x)
    results.append(attention_y)

  results = tf.concat(results, axis=0)
  # the shape of the first dimension is lost after concat, reset it back
  results.set_shape(data.shape)
  return results

class GraphPropMatchingLayer(GraphPropLayer):
  """A graph propagation layer that also does cross graph matching.

  It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
  forms the first pair and graph 2 and 3 are the second pair etc., and computes
  cross-graph attention-based matching for each pair.
  """

  def _build(self,
             node_states,
             from_idx,
             to_idx,
             graph_idx,
             n_graphs,
             similarity='dotproduct',
             edge_features=None,
             node_features=None):
    """Run one propagation step with cross-graph matching.

    Args:
      node_states: [n_nodes, node_state_dim] float tensor, node states.
      from_idx: [n_edges] int tensor, from node indices for each edge.
      to_idx: [n_edges] int tensor, to node indices for each edge.
      graph_idx: [n_onodes] int tensor, graph id for each node.
      n_graphs: integer, number of graphs in the batch.
      similarity: type of similarity to use for the cross graph attention.
      edge_features: if not None, should be [n_edges, edge_feat_dim] tensor,
        extra edge features.
      node_features: if not None, should be [n_nodes, node_feat_dim] tensor,
        extra node features.

    Returns:
      node_states: [n_nodes, node_state_dim] float tensor, new node states.

    Raises:
      ValueError: if some options are not provided correctly.
    """
    aggregated_messages = self._compute_aggregated_messages(
        node_states, from_idx, to_idx, edge_features=edge_features)

    # new stuff here
    cross_graph_attention = batch_block_pair_attention(
        node_states, graph_idx, n_graphs, similarity=similarity)
    attention_input = node_states - cross_graph_attention

    return self._compute_node_update(node_states,
                                     [aggregated_messages, attention_input],
                                     node_features=node_features)


class GraphMatchingNet(GraphEmbeddingNet):
  """Graph matching net.

  This class uses graph matching layers instead of the simple graph prop layers.

  It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
  forms the first pair and graph 2 and 3 are the second pair etc., and computes
  cross-graph attention-based matching for each pair.
  """

  def __init__(self,
               encoder,
               aggregator,
               node_state_dim,
               edge_hidden_sizes,
               node_hidden_sizes,
               n_prop_layers,
               share_prop_params=False,
               edge_net_init_scale=0.1,
               node_update_type='residual',
               use_reverse_direction=True,
               reverse_dir_param_different=True,
               layer_norm=False,
               similarity='dotproduct',
               name='graph-matching-net'):
    super(GraphMatchingNet, self).__init__(
        encoder,
        aggregator,
        node_state_dim,
        edge_hidden_sizes,
        node_hidden_sizes,
        n_prop_layers,
        share_prop_params=share_prop_params,
        edge_net_init_scale=edge_net_init_scale,
        node_update_type=node_update_type,
        use_reverse_direction=use_reverse_direction,
        reverse_dir_param_different=reverse_dir_param_different,
        layer_norm=layer_norm,
        name=name)
    self._similarity = similarity
    self._layer_class = GraphPropMatchingLayer

  def _apply_layer(self,
                   layer,
                   node_states,
                   from_idx,
                   to_idx,
                   graph_idx,
                   n_graphs,
                   edge_features):
    """Apply one layer on the given inputs."""
    return layer(node_states, from_idx, to_idx, graph_idx, n_graphs,
                 similarity=self._similarity, edge_features=edge_features)


def reshape_and_split_tensor(tensor, n_splits):
  """Reshape and split a 2D tensor along the last dimension.

  Args:
    tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
      multiple of `n_splits`.
    n_splits: int, number of splits to split the tensor into.

  Returns:
    splits: a list of `n_splits` tensors.  The first split is [tensor[0],
      tensor[n_splits], tensor[n_splits * 2], ...], the second split is
      [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
  """
  feature_dim = tensor.shape.as_list()[-1]
  # feature dim must be known, otherwise you can provide that as an input
  assert isinstance(feature_dim, int)
  tensor = tf.reshape(tensor, [-1, feature_dim * n_splits])
  return tf.split(tensor, n_splits, axis=-1)


def build_placeholders(node_feature_dim, edge_feature_dim):
  """Build the placeholders needed for the model.

  Args:
    node_feature_dim: int.
    edge_feature_dim: int.

  Returns:
    placeholders: a placeholder name -> placeholder tensor dict.
  """
  # `n_graphs` must be specified as an integer, as `tf.dynamic_partition`
  # requires so.
  return {
      'node_features': tf.placeholder(tf.float32, [None, node_feature_dim]),
      'edge_features': tf.placeholder(tf.float32, [None, edge_feature_dim]),
      'from_idx': tf.placeholder(tf.int32, [None]),
      'to_idx': tf.placeholder(tf.int32, [None]),
      'graph_idx': tf.placeholder(tf.int32, [None]),
      # only used for pairwise training and evaluation
      'labels': tf.placeholder(tf.int32, [None]),
  }


def build_model(config, node_feature_dim, edge_feature_dim):
  """Create model for training and evaluation.

  Args:
    config: a dictionary of configs, like the one created by the
      `get_default_config` function.
    node_feature_dim: int, dimensionality of node features.
    edge_feature_dim: int, dimensionality of edge features.

  Returns:
    tensors: a (potentially nested) name => tensor dict.
    placeholders: a (potentially nested) name => tensor dict.
    model: a GraphEmbeddingNet or GraphMatchingNet instance.

  Raises:
    ValueError: if the specified model or training settings are not supported.
  """
  encoder = GraphEncoder(**config['encoder'])
  aggregator = GraphAggregator(**config['aggregator'])
  if config['model_type'] == 'embedding':
    model = GraphEmbeddingNet(
        encoder, aggregator, **config['graph_embedding_net'])
  elif config['model_type'] == 'matching':
    model = GraphMatchingNet(
        encoder, aggregator, **config['graph_matching_net'])
  else:
    raise ValueError('Unknown model type: %s' % config['model_type'])

  training_n_graphs_in_batch = config['training']['batch_size']
  if config['training']['mode'] == 'pair':
    training_n_graphs_in_batch *= 2
  elif config['training']['mode'] == 'triplet':
    training_n_graphs_in_batch *= 4
  else:
    raise ValueError('Unknown training mode: %s' % config['training']['mode'])

  placeholders = build_placeholders(node_feature_dim, edge_feature_dim)

  # training
  model_inputs = placeholders.copy()
  del model_inputs['labels']
  model_inputs['n_graphs'] = training_n_graphs_in_batch
  graph_vectors = model(**model_inputs)

  if config['training']['mode'] == 'pair':
    x, y = reshape_and_split_tensor(graph_vectors, 2)
    loss = pairwise_loss(x, y, placeholders['labels'],
                         loss_type=config['training']['loss'],
                         margin=config['training']['margin'])

    # optionally monitor the similarity between positive and negative pairs
    is_pos = tf.cast(tf.equal(placeholders['labels'], 1), tf.float32)
    is_neg = 1 - is_pos
    n_pos = tf.reduce_sum(is_pos)
    n_neg = tf.reduce_sum(is_neg)
    sim = compute_similarity(config, x, y)
    sim_pos = tf.reduce_sum(sim * is_pos) / (n_pos + 1e-8)
    sim_neg = tf.reduce_sum(sim * is_neg) / (n_neg + 1e-8)
  else:
    x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
    loss = triplet_loss(x_1, y, x_2, z,
                        loss_type=config['training']['loss'],
                        margin=config['training']['margin'])

    sim_pos = tf.reduce_mean(compute_similarity(config, x_1, y))
    sim_neg = tf.reduce_mean(compute_similarity(config, x_2, z))

  graph_vec_scale = tf.reduce_mean(graph_vectors**2)
  if config['training']['graph_vec_regularizer_weight'] > 0:
    loss += (config['training']['graph_vec_regularizer_weight'] *
             0.5 * graph_vec_scale)

  # monitor scale of the parameters and gradients, these are typically helpful
  optimizer = tf.train.AdamOptimizer(
      learning_rate=config['training']['learning_rate'])
  grads_and_params = optimizer.compute_gradients(loss)
  grads, params = zip(*grads_and_params)
  grads, _ = tf.clip_by_global_norm(grads, config['training']['clip_value'])
  train_step = optimizer.apply_gradients(zip(grads, params))

  grad_scale = tf.global_norm(grads)
  param_scale = tf.global_norm(params)

  # evaluation
  model_inputs['n_graphs'] = config['evaluation']['batch_size'] * 2
  eval_pairs = model(**model_inputs)
  x, y = reshape_and_split_tensor(eval_pairs, 2)
  similarity = compute_similarity(config, x, y)
  pair_auc = auc(similarity, placeholders['labels'])

  model_inputs['n_graphs'] = config['evaluation']['batch_size'] * 4
  eval_triplets = model(**model_inputs)
  x_1, y, x_2, z = reshape_and_split_tensor(eval_triplets, 4)
  sim_1 = compute_similarity(config, x_1, y)
  sim_2 = compute_similarity(config, x_2, z)
  triplet_acc = tf.reduce_mean(tf.cast(sim_1 > sim_2, dtype=tf.float32))

  return {
      'train_step': train_step,
      'metrics': {
          'training': {
              'loss': loss,
              'grad_scale': grad_scale,
              'param_scale': param_scale,
              'graph_vec_scale': graph_vec_scale,
              'sim_pos': sim_pos,
              'sim_neg': sim_neg,
              'sim_diff': sim_pos - sim_neg,
          },
          'validation': {
              'pair_auc': pair_auc,
              'triplet_acc': triplet_acc,
          },
      },
  }, placeholders, model
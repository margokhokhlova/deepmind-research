import os
import random, copy, math
import abc
import collections
import six
import numpy as np
import networkx as nx
import tensorflow as tf

import contextlib

# data manimulations
def permute_graph_nodes(g):
  """Permute node ordering of a graph, returns a new graph."""
  g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='default')
  # create a random mapping old label -> new label
  node_mapping = dict(zip(g.nodes(), sorted(g.nodes(), key=lambda k: random.random())))
  # build a new graph
  new_g = nx.relabel_nodes(g, node_mapping)
  return new_g


def substitute_random_edges(g, n):
  """Substitutes n edges from graph g with another n randomly picked edges."""
  g = copy.deepcopy(g)
  n_nodes = g.number_of_nodes()
  edges = list(g.edges())
  # sample n edges without replacement
  e_remove = [edges[i] for i in np.random.choice(
      np.arange(len(edges)), n, replace=False)]
  edge_set = set(edges)
  e_add = set()
  while len(e_add) < n:
    e = np.random.choice(n_nodes, 2, replace=False)
    # make sure e does not exist and is not already chosen to be added
    if ((e[0], e[1]) not in edge_set and (e[1], e[0]) not in edge_set and
        (e[0], e[1]) not in e_add and (e[1], e[0]) not in e_add):
      e_add.add((e[0], e[1]))

  for i, j in e_remove:
    g.remove_edge(i, j)
  for i, j in e_add:
    g.add_edge(i, j)
  return g

GraphData = collections.namedtuple('GraphData', [
    'from_idx',
    'to_idx',
    'node_features',
    'edge_features',
    'graph_idx',
    'n_graphs'])


@six.add_metaclass(abc.ABCMeta)
class GraphSimilarityDataset(object):
  """Base class for all the graph similarity learning datasets.

  This class defines some common interfaces a graph similarity dataset can have,
  in particular the functions that creates iterators over pairs and triplets.
  """

  @abc.abstractmethod
  def triplets(self, batch_size):
    """Create an iterator over triplets.

    Args:
      batch_size: int, number of triplets in a batch.

    Yields:
      graphs: a `GraphData` instance.  The batch of triplets put together.  Each
        triplet has 3 graphs (x, y, z).  Here the first graph is duplicated once
        so the graphs for each triplet are ordered as (x, y, x, z) in the batch.
        The batch contains `batch_size` number of triplets, hence `4*batch_size`
        many graphs.
    """
    pass

  @abc.abstractmethod
  def pairs(self, batch_size):
    """Create an iterator over pairs.

    Args:
      batch_size: int, number of pairs in a batch.

    Yields:
      graphs: a `GraphData` instance.  The batch of pairs put together.  Each
        pair has 2 graphs (x, y).  The batch contains `batch_size` number of
        pairs, hence `2*batch_size` many graphs.
      labels: [batch_size] int labels for each pair, +1 for similar, -1 for not.
    """
    pass


def read_data(
        name,
        with_classes=True,
        prefer_attr_nodes=False,
        prefer_attr_edges=False,
        produce_labels_nodes=False,
        as_graphs=False,
        is_symmetric=True, path = None, with_node_posistions=False):
    """Create a dataset iterable for GraphKernel.

    Parameters
    ----------
    name : str
        The dataset name.

    with_classes : bool, default=False
        Return an iterable of class labels based on the enumeration.

    produce_labels_nodes : bool, default=False
        Produce labels for nodes if not found.
        Currently this means labeling its node by its degree inside the Graph.
        This operation is applied only if node labels are non existent.

    prefer_attr_nodes : bool, default=False
        If a dataset has both *node* labels and *node* attributes
        set as labels for the graph object for *nodes* the attributes.

    prefer_attr_edges : bool, default=False
        If a dataset has both *edge* labels and *edge* attributes
        set as labels for the graph object for *edge* the attributes.

    as_graphs : bool, default=False
        Return data as a list of Graph Objects.

    is_symmetric : bool, default=False
        Defines if the graph data describe a symmetric graph.

    Returns
    -------
    Gs : iterable
        An iterable of graphs consisting of a dictionary, node
        labels and edge labels for each graph.

    classes : np.array, case_of_appearance=with_classes==True
        An one dimensional array of graph classes aligned with the lines
        of the `Gs` iterable. Useful for classification.

    """
    indicator_path = path+str(name)+"_graph_indicator.txt"
    edges_path =  path + "/" + str(name) + "_A.txt"
    node_labels_path = path + "/" + str(name) + "_node_labels.txt"
    node_attributes_path = path +"/"+str(name)+"_node_attributes.txt"
    edge_labels_path = path + "/" + str(name) + "_edge_labels.txt"
    edge_attributes_path = \
        path + "/" + str(name) + "_edge_attributes.txt"
    graph_classes_path = \
        path + "/" + str(name) + "_graph_labels.txt"
    node_positions_path = \
        path + "/" + str(name) + "_node_positions.txt"
    # node graph correspondence
    ngc = dict()
    # edge line correspondence
    elc = dict()
    # dictionary that keeps sets of edges
    Graphs = dict()
    # dictionary of labels for nodes
    node_labels = dict()
    # dictionary of labels for edges
    edge_labels = dict()
    #dictionary for positions
    node_positions = dict()

    # Associate graphs nodes with indexes
    with open(indicator_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            ngc[i] = int(line[:-1])
            if int(line[:-1]) not in Graphs:
                Graphs[int(line[:-1])] = set()
            if int(line[:-1]) not in node_labels:
                node_labels[int(line[:-1])] = dict()
            if int(line[:-1]) not in edge_labels:
                edge_labels[int(line[:-1])] = dict()

    # Create backwards configuration
    graph_node_correspondence = collections.defaultdict(list)
    for node in range(len(ngc)):
        graph_node_correspondence[ngc[node+1]].append(node+1)


    # Extract graph edges
    with open(edges_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            edge = line[:-1].replace(' ', '').split(",")
            elc[i] = (int(edge[0]), int(edge[1]))
            Graphs[ngc[int(edge[0])]].add((int(edge[0]), int(edge[1])))
            if is_symmetric:
                Graphs[ngc[int(edge[1])]].add((int(edge[1]), int(edge[0])))

    # Extract node attributes
    if prefer_attr_nodes:
        with open(node_attributes_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                node_labels[ngc[i]][i] = \
                    [float(num) for num in
                     line[:-1].replace(' ', '').split(",")]
                #if np.isnan(node_labels[ngc[i]][i]).any():  # then there are None values
                node_labels[ngc[i]][i] = [0.00 if math.isnan(x) else x for x in node_labels[ngc[i]][i]][:]  # remove NaNs and take only 2 last

                #node_labels[ngc[i]][i] = [x for x in node_labels[ngc[i]][i][1:2]]  # remove NaNs
    # Extract node labels
    elif not produce_labels_nodes:
        with open(node_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                node_labels[ngc[i]][i] = int(line[:-1])
    elif produce_labels_nodes:
        for i in range(1, len(Graphs)+1):
            node_labels[i] = dict(collections.Counter(s for (s, d) in Graphs[i] if s != d))
            if not bool(node_labels[i]): #if labels are empty
                node_labels[i] = {s:0 for s in graph_node_correspondence[i]}

    # Extract edge attributes
    if prefer_attr_edges:
        with open(edge_attributes_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                attrs = [float(num)
                         for num in line[:-1].replace(' ', '').split(",")]
                edge_labels[ngc[elc[i][0]]][elc[i]] = attrs
                if is_symmetric:
                    edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = attrs

    # Extract edge labels
    elif not prefer_attr_edges and  os.path.exists(edge_labels_path):
        with open(edge_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                edge_labels[ngc[elc[i][0]]][elc[i]] = float(line[:-1])
                if is_symmetric:
                    edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = \
                        float(line[:-1])
    elif not prefer_attr_edges and  not os.path.exists(edge_labels_path):
        with open(edges_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                edge_labels[ngc[elc[i][0]]][elc[i]] = 1
                if is_symmetric:
                    edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = 1

    Gs = list()
    if as_graphs:
        for i in range(1, len(Graphs)+1):
            nx_graph = nx.Graph()
            #nx_graph.add_nodes_from(Graphs[i])
            nx_graph.add_edges_from(edge_labels[i])
            nx.set_node_attributes(nx_graph, node_labels[i], 'labels')
            Gs.append(nx_graph)
    else:
        for i in range(1, len(Graphs)+1):
            Gs.append([Graphs[i], node_labels[i], edge_labels[i]])

    if with_node_posistions:
        with open(node_positions_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                positions = [float(num) for num in
                     line.replace(' ', '').split(",")]
                if ngc[i] not in node_positions:
                    node_positions[ngc[i]]= {i:positions}
                else:
                    node_positions[ngc[i]].update({i:positions})#
        classes = []
        with open(graph_classes_path, "r") as f:
            for line in f:
                classes.append(int(line[:-1]) - 1)

        classes = np.array(classes, dtype=np.int)
        return Gs, classes, node_positions

    if with_classes:
        classes = []
        with open(graph_classes_path, "r") as f:
            for line in f:
                classes.append(int(line[:-1])-1)

        classes = np.array(classes, dtype=np.int)
        return Gs, classes
    else:
        return Gs




# def unison_shuffled_inplace(a, b):
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     a[:] = a[p]
#     b[:] = b[p]

def shuffle_arrays(arrays, set_seed=-1):
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)











class GraphGeoDataset(GraphSimilarityDataset):
  """Graph edit distance dataset."""

  def __init__(self,
                data_first, data_second,
               permute=True):
    """Constructor.

    Args:
     data_first - the first year dataset
     data_second - the second year dataset
      permute: if True (default), permute node orderings in addition to
        changing edges; if False, the node orderings across a pair or triplet of
        graphs will be the same, useful for visualization.
    """
    self._data1 = data_first
    self._data2 = data_second
    self._permute = permute
    self._dataset_size = len(data_first) # length of the dataset
    self._seed = 111

  def _get_pair(self, idx, positive):
    """Get one pair of matched graphs."""
    g1 = self._data1[idx]
    if positive:
        g2 = self._data2[idx]
    else:
        g2 = self._data2[random.randint(0,self._dataset_size-1)] #any random non-corresponding graph
    if self._permute:
        g1, g2 = permute_graph_nodes(g1), permute_graph_nodes(g2)
    return g1, g2



  def _get_triplet(self):
    """Generate one triplet of graphs."""
    g = self._get_graph()
    if self._permute:
      permuted_g = permute_graph_nodes(g)
    else:
      permuted_g = g
    pos_g = substitute_random_edges(g, self._k_pos)
    neg_g = substitute_random_edges(g, self._k_neg)
    return permuted_g, pos_g, neg_g

  def triplets(self, batch_size):
    """Yields batches of triplet data."""
    while True:
      batch_graphs = []
      for _ in range(batch_size):
        g1, g2, g3 = self._get_triplet()
        batch_graphs.append((g1, g2, g1, g3))
      yield self._pack_batch(batch_graphs)




  def pairs(self, batch_size):
    """Yield pairs and labels."""

    if hasattr(self, '_pairs') and hasattr(self, '_labels'):
      pairs = self._pairs
      labels = self._labels
    else:
      # get a fixed set of pairs first
      with reset_random_state(self._seed):
        shuffle_arrays([self._data1, self._data2], set_seed = self._seed)
        pairs = []
        labels = []
        positive = True
        for idx in range(self._dataset_size):
          pairs.append(self._get_pair(idx, positive))
          labels.append(1 if positive else -1)
          positive = not positive
      labels = np.array(labels, dtype=np.int32)

      self._pairs = pairs
      self._labels = labels

    ptr = 0
    while ptr + batch_size <= len(pairs):
      batch_graphs = pairs[ptr:ptr + batch_size]
      packed_batch = self._pack_batch(batch_graphs)
      yield packed_batch, labels[ptr:ptr + batch_size]
      ptr += batch_size
  # def pairs(self, idx, batch_size):
  #   """Yields batches of pair data."""
  #   while True:
  #     batch_graphs = []
  #     batch_labels = []
  #     positive = True
  #     for _ in range(batch_size):
  #       g1, g2 = self._get_pair(idx,positive)
  #       batch_graphs.append((g1, g2))
  #       batch_labels.append(1 if positive else -1)
  #       positive = not positive
  #       idx +=1
  #
  #     packed_graphs = self._pack_batch(batch_graphs)
  #     labels = np.array(batch_labels, dtype=np.int32)
  #     yield packed_graphs, labels

  def _pack_batch(self, graphs, padded_to = 150):
    """Pack a batch of graphs into a single `GraphData` instance.

    Args:
      graphs: a list of generated networkx graphs.
      padded_to: the padding size for the nodes, adding dummy nodes without edges.
    Returns:
      graph_data: a `GraphData` instance, with node and edge indices properly
        shifted.
    """
    graphs = tf.nest.flatten(graphs)
    from_idx = []
    to_idx = []
    graph_idx = []
    graphs_features = []
    n_total_nodes = 0
    n_total_edges = 0
    for i, g in enumerate(graphs):
      #add dummy nodes
      for j in range(len(g.nodes),padded_to):
        g.add_node(j, labels = [0,0])
      n_nodes = g.number_of_nodes()
      n_edges = g.number_of_edges()
      edges = np.array(g.edges(), dtype=np.int32)
      # shift the node indices for the edges
      from_idx.append(edges[:, 0] + n_total_nodes)
      to_idx.append(edges[:, 1] + n_total_nodes)
      graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)
      graphs_features.extend(list(nx.get_node_attributes(g,'labels').values()))
      n_total_nodes += n_nodes
      n_total_edges += n_edges

    return GraphData(
        from_idx=np.concatenate(from_idx, axis=0),
        to_idx=np.concatenate(to_idx, axis=0),
        # node labels are taken into account, but the edges all have weight 1
        node_features=np.array(graphs_features),
        edge_features=np.ones((n_total_edges, 1), dtype=np.float32),
        graph_idx=np.concatenate(graph_idx, axis=0),
        n_graphs=len(graphs))


@contextlib.contextmanager
def reset_random_state(seed):
  """This function creates a context that uses the given seed."""
  np_rnd_state = np.random.get_state()
  rnd_state = random.getstate()
  np.random.seed(seed)
  random.seed(seed + 1)
  try:
    yield
  finally:
    random.setstate(rnd_state)
    np.random.set_state(np_rnd_state)


class FixedGraphGeoDataset(GraphGeoDataset):
  """A fixed dataset of pairs or triplets for the graph edit distance task.

  This dataset can be used for evaluation.
  """

  def __init__(self,
               n_nodes_range,
               p_edge_range,
               n_changes_positive,
               n_changes_negative,
               dataset_size,
               permute=True,
               seed=1234):
    super(FixedGraphGeoDataset, self).__init__(
        n_nodes_range, p_edge_range, n_changes_positive, n_changes_negative,
        permute=permute)
    self._dataset_size = dataset_size
    self._seed = seed

  def triplets(self, batch_size):
    """Yield triplets."""

    if hasattr(self, '_triplets'):
      triplets = self._triplets
    else:
      # get a fixed set of triplets
      with reset_random_state(self._seed):
        triplets = []
        for _ in range(self._dataset_size):
          g1, g2, g3 = self._get_triplet()
          triplets.append((g1, g2, g1, g3))
      self._triplets = triplets

    ptr = 0
    while ptr + batch_size <= len(triplets):
      batch_graphs = triplets[ptr:ptr + batch_size]
      yield self._pack_batch(batch_graphs)
      ptr += batch_size


  def reset_pairs(self):
      ''' small workaround to calculate new pairs for training '''
      with reset_random_state(self._seed):
        pairs = []
        labels = []
        positive = True
        for _ in range(self._dataset_size):
          pairs.append(self._get_pair(positive))
          labels.append(1 if positive else -1)
          positive = not positive
      labels = np.array(labels, dtype=np.int32)

      self._pairs = pairs
      self._labels = labels

  def pairs(self, batch_size):
    """Yield pairs and labels."""

    if hasattr(self, '_pairs') and hasattr(self, '_labels'):
      pairs = self._pairs
      labels = self._labels
    else:
      # get a fixed set of pairs first
      self.reset_pairs()

    ptr = 0
    while ptr + batch_size <= len(pairs):
      batch_graphs = pairs[ptr:ptr + batch_size]
      packed_batch = self._pack_batch(batch_graphs)
      yield packed_batch, labels[ptr:ptr + batch_size]
      ptr += batch_size
      if ptr + batch_size <= len(pairs):
          ptr = 0
          # workaround to reset pairs
          self.reset_pairs()


###### DATA #########




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

    # Let's just run for a small number of training steps.  This may take you a few
    # minutes.
    config['training']['n_training_steps'] = 100
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

    for i_iter in range(3):
        batch = next(training_data_iter)

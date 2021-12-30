from typing import Optional, Callable

import numpy as np
import os

import torch
from torch.nn.functional import one_hot
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.io.tu import split
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce


NUM_LABELS = {'ENZYMES': 3, 'COLLAB': 0, 'IMDBBINARY': 0, 'IMDBMULTI': 0, 'MUTAG': 7, 'NCI1': 37, 'NCI109': 38, 'PROTEINS': 3, 'PTC': 22, 'DD': 89}
NUM_CLASSES = {'COLLAB': 3, 'IMDBBINARY': 2, 'IMDBMULTI': 3, 'MUTAG': 2, 'NCI1': 2, 'NCI109': 2, 'PROTEINS': 2, 'PTC': 2}

LEARNING_RATES = {'COLLAB': 0.00256, 'IMDBBINARY': 0.00064, 'IMDBMULTI': 0.00064, 'MUTAG': 0.00512, 'NCI1': 0.00256, 'NCI109': 0.00256, 'PROTEINS': 0.00064, 'PTC': 0.00128}
DECAY_RATES = {'COLLAB': 0.7, 'IMDBBINARY': 0.4, 'IMDBMULTI': 0.7, 'MUTAG': 0.7, 'NCI1': 0.7, 'NCI109': 0.7, 'PROTEINS': 0.7, 'PTC': 0.6}
CHOSEN_EPOCH = {'COLLAB': 100, 'IMDBBINARY': 40, 'IMDBMULTI': 150, 'MUTAG': 130, 'NCI1': 99, 'NCI109': 99, 'PROTEINS': 20, 'PTC': 9}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))


def load_dataset(ds_name):
    """
    construct graphs and labels from dataset text in data folder
    :param ds_name: name of data set you want to load
    :return: two numpy arrays of shape (num_of_graphs).
            the graphs array contains in each entry a ndarray represent adjacency matrix of a graph of shape (num_vertex, num_vertex, num_vertex_labels)
            the labels array in index i represent the class of graphs[i]
    """
    directory = BASE_DIR + "/../data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
    graphs = []
    labels = []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name] + 1), dtype=np.float32)
            labels.append(int(graph_meta[1]))
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                if NUM_LABELS[ds_name] != 0:
                    curr_graph[j, j, int(vertex[0]) + 1] = 1.
                for k in range(2, len(vertex)):
                    curr_graph[j, int(vertex[k]), 0] = 1.
            curr_graph = noramlize_graph(curr_graph)
            graphs.append(curr_graph)
    graphs = np.array(graphs, dtype=object)
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2, 0, 1])
    return graphs, np.array(labels)


def get_train_val_indexes(num_val, ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param num_val: number of the split
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/../data/benchmark_graphs/{0}/10fold_idx".format(ds_name)
    train_file = "train_idx-{0}.txt".format(num_val)
    train_idx = []
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "test_idx-{0}.txt".format(num_val)
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def get_parameter_split(ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/../data/benchmark_graphs/{0}/".format(ds_name)
    train_file = "tests_train_split.txt"
    train_idx = []
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "tests_val_split.txt"
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def group_same_size(graphs, labels):
    """
    group graphs of same size to same array
    :param graphs: numpy array of shape (num_of_graphs) of numpy arrays of graphs adjacency matrix
    :param labels: numpy array of labels
    :return: two numpy arrays. graphs arrays in the shape (num of different size graphs) where each entry is a numpy array
            in the shape (number of graphs with this size, num vertex, num. vertex, num vertex labels)
            the second arrayy is labels with correspons shape
    """
    sizes = list(map(lambda t: t.shape[1], graphs))
    indexes = np.argsort(sizes)
    graphs = graphs[indexes]
    labels = labels[indexes]
    r_graphs = []
    r_labels = []
    one_size = []
    start = 0
    size = graphs[0].shape[1]
    for i in range(len(graphs)):
        if graphs[i].shape[1] == size:
            one_size.append(np.expand_dims(graphs[i], axis=0))
        else:
            r_graphs.append(np.concatenate(one_size, axis=0))
            r_labels.append(np.array(labels[start:i]))
            start = i
            one_size = []
            size = graphs[i].shape[1]
            one_size.append(np.expand_dims(graphs[i], axis=0))
    r_graphs.append(np.concatenate(one_size, axis=0))
    r_labels.append(np.array(labels[start:]))
    return r_graphs, r_labels


# helper method to shuffle each same size graphs array
def shuffle_same_size(graphs, labels):
    r_graphs, r_labels = [], []
    for i in range(len(labels)):
        curr_graph, curr_labels = shuffle(graphs[i], labels[i])
        r_graphs.append(curr_graph)
        r_labels.append(curr_labels)
    return r_graphs, r_labels


def split_to_batches(graphs, labels, size):
    """
    split the same size graphs array to batches of specified size
    last batch is in size num_of_graphs_this_size % size
    :param graphs: array of arrays of same size graphs
    :param labels: the corresponding labels of the graphs
    :param size: batch size
    :return: two arrays. graphs array of arrays in size (batch, num vertex, num vertex. num vertex labels)
                corresponds labels
    """
    r_graphs: np.Array = []
    r_labels: np.Array = []
    for k in range(len(graphs)):
        r_graphs = r_graphs + np.split(graphs[k], [j for j in range(size, graphs[k].shape[0], size)])
        r_labels = r_labels + np.split(labels[k], [j for j in range(size, labels[k].shape[0], size)])
    return np.array(r_graphs, dtype=object), np.array(r_labels, dtype=object)


# helper method to shuffle the same way graphs and labels arrays
def shuffle(graphs, labels):
    shf = np.arange(labels.shape[0], dtype=np.int32)
    np.random.shuffle(shf)
    return np.array(graphs)[shf], labels[shf]


def noramlize_graph(curr_graph):

    split = np.split(curr_graph, [1], axis=2)

    adj = np.squeeze(split[0], axis=2)
    deg = np.sqrt(np.sum(adj, 0))
    deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg != 0)
    normal = np.diag(deg)
    norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
    ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
    spred_adj = np.multiply(ones, norm_adj)
    labels = np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1)), split[1], axis=2)
    return np.add(spred_adj, labels)


class GraphDataset(IterableDataset):
    __repr_indent = 4

    def __init__(self, name: str, *, train: bool = True, batch_size: int = 1, num_fold: int = 0):

        assert name in NUM_LABELS, "{} is not contained.".format(name)
        self.name = name
        self.batch_size = batch_size
        self.num_fold = num_fold
        self.train = train

        graphs, labels = load_dataset(name)

        if num_fold == 0:
            train_idx, test_idx = get_parameter_split(name)
        else:
            train_idx, test_idx = get_train_val_indexes(num_fold, name)

        if train:
            self.data, self.labels = graphs[train_idx], labels[train_idx]
        else:
            self.data, self.labels = graphs[test_idx], labels[test_idx]
            self.data = [np.expand_dims(g, 0) for g in self.data]

        self.max_set_size = max(map(lambda t: t.shape[1], graphs))

    @property
    def in_feature(self):
        return NUM_LABELS[self.name] + 1

    @property
    def num_classes(self):
        return NUM_CLASSES[self.name]

    def __iter__(self):
        if self.train:
            graphs, labels = self.reshuffle_data()
        else:
            graphs, labels = self.data, self.labels
        return iter(zip(graphs, labels))

    def reshuffle_data(self):
        graphs, labels = group_same_size(self.data, self.labels)
        graphs, labels = shuffle_same_size(graphs, labels)
        graphs, labels = split_to_batches(graphs, labels, self.batch_size)
        self.num_iterations_train = len(graphs)
        graphs, labels = shuffle(graphs, labels)
        return graphs, labels

    def __repr__(self) -> str:
        head = "Dataset {} ({})".format(self.name, "Train" if self.train else "Test")
        body = [
            f"In features: {self.in_feature}",
            f"Number of Classes: {self.num_classes}",
            f"Number of Data: {len(self.data)}",
            f"Batch size: {self.batch_size}",
            f"No. Fold: {self.num_fold}",
        ]
        lines = [head] + [" " * self.__repr_indent + line for line in body]
        return '\n'.join(lines)


def read_tudataset(ds_name):
    """
    construct graphs and labels from dataset text in data folder
    :param ds_name: name of data set you want to load
    :return: two numpy arrays of shape (num_of_graphs).
            the graphs array contains in each entry a ndarray represent adjacency matrix of a graph of shape (num_vertex, num_vertex, num_vertex_labels)
            the labels array in index i represent the class of graphs[i]
    """
    directory = BASE_DIR + "/../data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
    y = []
    batch = []
    edge_index = []
    node_labels = []
    current_num_nodes = 0
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            y.append(int(graph_meta[1]))
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                batch.append(i)
                if NUM_LABELS[ds_name] != 0:
                    node_labels.append(int(vertex[0]))
                for k in range(2, len(vertex)):
                    edge_index.append((j + current_num_nodes, int(vertex[k]) + current_num_nodes))
            current_num_nodes += num_vertex

    x = one_hot(torch.tensor(node_labels).long(), num_classes=-1) if NUM_LABELS[ds_name] != 0 else None

    batch = torch.tensor(batch)
    edge_index = torch.tensor(edge_index).transpose(-1, -2)
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    edge_index, _ = remove_self_loops(edge_index, None)
    edge_index, _ = coalesce(edge_index, _, num_nodes, num_nodes)
    y = torch.tensor(y)

    data = Data(x=x, edge_index=edge_index, y=y)
    data, slices = split(data, batch)

    return data, slices


class TUDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        self.data, self.slices = read_tudataset(self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


def get_tudataset(name: str,
                  num_fold: int = 0,
                  transform: Optional[Callable] = None,
                  pre_transform: Optional[Callable] = None,
                  pre_filter: Optional[Callable] = None):

    directory = BASE_DIR + "/../data/test/{0}".format(name)
    dataset = TUDataset(directory,
                        name,
                        transform=transform,
                        pre_transform=pre_transform,
                        pre_filter=pre_filter)

    if num_fold == 0:
        train_idx, test_idx = get_parameter_split(name)
    else:
        train_idx, test_idx = get_train_val_indexes(num_fold, name)

    return dataset[train_idx], dataset[test_idx]

""" Utils for io, language, connectivity graphs etc """

import sys
import json
import numpy as np
import networkx as nx
import torch

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')


def load_nav_graphs(scans):
    """ Load connectivity graph for each scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            g = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            g.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(g, values=positions, name='position')
            graphs[scan] = g
    return graphs


def load_datasets(splits):
    data = []
    for split in splits:
        assert split in ['train', 'val_seen', 'val_unseen', 'test']
        with open('tasks/R2R/data/R2R_%s.json' % split) as f:
            data += json.load(f)
    return data


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def check_config_trainer(config):
    field_list = ['action_space', 'features', 'img_spec', 'splits', 'batch_size', 'seed', 'results_path']
    assert isinstance(config, dict), 'expected type dict for argument config, found %s' % type(config)
    for field in field_list:
        assert field in config, 'missing required field in config: %s' % field
    assert config['action_space'] in ['low', 'high'], 'action space should be either "low" or "high", found %s' % config['action_space']
    return config


def check_config_judge(config):
    field_list = ['action_space', 'features', 'img_spec', 'splits', 'batch_size', 'seed', 'results_path']
    assert isinstance(config, dict), 'expected type dict for argument config, found %s' % type(config)
    for field in field_list:
        assert field in config, 'missing required field in config: %s' % field
    assert config['action_space'] in ['low', 'high'], 'action space should be either "low" or "high", found %s' % config['action_space']

    if isinstance(config['splits'], str):
        config['splits'] = [config['splits']]
    assert isinstance(config['splits'], list), 'expected type list or str type for argument "splits", found %s' % type(config['splits'])

    return config


def my_split_func(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    include_words = ['up', 'down', 'out', 'into', 'through', 'above', 'before', 'below', 'to', 'over', 'under']

    for word in include_words:
        stop_words.remove(word)

    word_tokens = tokenizer.tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    filtered_sentence_reversed = [w for w in reversed(word_tokens) if w not in stop_words]

    return filtered_sentence, filtered_sentence_reversed


def batched_sentence_embedding(batch, word_encoder, device=torch.device('cpu')):
    """
    :param batch: batch of instructions of variable lengths --> suppose range [min_l, max_l]
    :param word_encoder: provides single-word embeddings --- must support __getitem__ method
    :param device: may be cpu or cuda -- default is cpu
    :return: tensor of shape [batch_len, max_l, embedding_size] where sentences are zero-padded to have same size
    """
    split_batch = []

    for sentence in batch:
        spl, spl_rev = my_split_func(sentence)
        split_batch.append(spl)

    lengths = [len(spl) for spl in split_batch]
    max_l = max(lengths)

    t = torch.zeros(len(batch), max_l, 300)

    for i, spl in enumerate(split_batch):
        e = torch.stack([word_encoder[word] for word in spl])
        t[i, :e.shape[0], :] = e.squeeze(dim=1)

    t = t.transpose(1, 2)

    embeddings = t.to(device=device)
    return embeddings, lengths


def append_coordinates(features, agent_heading, agent_elevation):
    """ Appends elevation and headings coordinates to attention heatmap """

    """
    Assume features is 36 x num_features: appends 36-dimensional maps with elevation and headings.
    Indexing is the following:
            _________________________________________________
            |                                                |
    up      | 24  25  26  27  28  29  30  31  32  33  34  35 |
            |                                                |
    center  | 12  13  14  15  16  17  18  19  20  21  22  23 |
            |                                                |
    down    |  0   1   2   3   4   5   6   7   8   9  10  11 |
            |________________________________________________|

            left                 center                  right
    """

    abs_elevations = torch.tensor([-0.5, 0, 0.5], dtype=torch.float)
    elevations = abs_elevations - agent_elevation
    elevations_map = elevations.repeat(12, 1).transpose(0, 1).contiguous().view(36, 1)

    abs_headings = torch.tensor(np.linspace(0, (11./6.)*np.pi, 12), dtype=torch.float)
    headings = abs_headings - agent_heading

    headings_cos_map = torch.cos(headings).repeat(3).view(36, 1)
    headings_sin_map = torch.sin(headings).repeat(3).view(36, 1)

    feature_map = torch.cat((features, elevations_map, headings_cos_map, headings_sin_map), dim=-1)

    return feature_map


def to_one_hot(indexes, output_dim):
    """
    :param indexes: list of numbers in the range [0, output_dim)
    :param output_dim: size of a single one-hot tensor
    :return: tensor containing one_hot representation of indexes
    """
    assert output_dim >= 2
    assert output_dim > max(indexes)
    assert min(indexes) >= 0

    return torch.eye(output_dim)[indexes]



from tasks.R2R.utils import load_datasets, load_nav_graphs
from tasks.R2R.env import env_list
from tasks.R2R.utils import check_config_judge
from collections import defaultdict

import json
import os
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)


class CLS(object):
    """ Coverage weighted by length score (CLS).
        Link to the original paper:
        https://arxiv.org/abs/1905.12255
    """
    def __init__(self, graph, weight='weight', threshold=3.0):
        """Initializes a CLS object.

        Args:
          graph: networkx graph for the environment.
          weight: networkx edge weight key (str).
          threshold: distance threshold $d_{th}$ (float).
        """
        self.graph = graph
        self.weight = weight
        self.threshold = threshold
        self.distance = dict(
            nx.all_pairs_dijkstra_path_length(
                self.graph, weight=self.weight))

    def __call__(self, prediction, reference):
        """Computes the CLS metric.

        Args:
          prediction: list of nodes (str), path predicted by agent.
          reference: list of nodes (str), the ground truth path.

        Returns:
          the CLS between the prediction and reference path (float).
        """

        def length(nodes):
            lens = []
            for edge in zip(nodes[:-1], nodes[1:]):
                try:
                    lens.append(self.graph.edges[edge].get(self.weight, 1.0))
                except KeyError:
                    pass
            return np.sum(lens)

        coverage = np.mean([
            np.exp(-np.min([  # pylint: disable=g-complex-comprehension
                self.distance[u][v] for v in prediction
            ]) / self.threshold) for u in reference
        ])
        expected = coverage * length(reference)
        score = expected / (expected + np.abs(expected - length(prediction)))
        return coverage * score


class DTW(object):
    """ Dynamic Time Warping (DTW) evaluation metrics. """

    def __init__(self, graph, weight='weight', threshold=3.0):
        """Initializes a DTW object.

        Args:
          graph: networkx graph for the environment.
          weight: networkx edge weight key (str).
          threshold: distance threshold $d_{th}$ (float).
        """
        self.graph = graph
        self.weight = weight
        self.threshold = threshold
        self.distance = dict(
            nx.all_pairs_dijkstra_path_length(self.graph, weight=self.weight))

    def __call__(self, prediction, reference, metric='sdtw'):
        """Computes DTW metrics.

        Args:
          prediction: list of nodes (str), path predicted by agent.
          reference: list of nodes (str), the ground truth path.
          metric: one of ['ndtw', 'sdtw', 'dtw'].

        Returns:
          the DTW between the prediction and reference path (float).
        """
        assert metric in ['ndtw', 'sdtw', 'dtw']

        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction)+1):
            for j in range(1, len(reference)+1):
                best_previous_cost = min(
                    dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
                cost = self.distance[prediction[i-1]][reference[j-1]]
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]

        if metric == 'dtw':
            return dtw

        ndtw = np.exp(-dtw/(self.threshold * len(reference)))
        if metric == 'ndtw':
            return ndtw

        success = self.distance[prediction[-1]][reference[-1]] <= self.threshold
        return success * ndtw


class Evaluation(object):
    """ Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] """

    def __init__(self, splits):
        self.error_margin = 3.0
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        for item in load_datasets(splits):
            self.gt[item['path_id']] = item
            self.scans.append(item['scan'])
            self.instr_ids += ['%d_%d' % (item['path_id'], i) for i in range(3)]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        self.scores = None
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        self.cls = {}
        self.dtw = {}
        for scan, G in self.graphs.items():
            self.cls[scan] = CLS(G)
            self.dtw[scan] = DTW(G)

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        """ Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule). """
        gt = self.gt[int(instr_id.split('_')[0])]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['trajectory_steps'].append(len(path) - 1)
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            if prev[0] != curr[0]:
                try:
                    self.graphs[gt['scan']][prev[0]][curr[0]]
                except KeyError:
                    print('Error: The provided trajectory moves from %s to %s but the navigation graph contains no ' 
                          'edge between these viewpoints. Please ensure the provided navigation trajectories ' 
                          'are valid, so that trajectory length can be accurately calculated.' % (prev[0], curr[0]))
                    raise
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_path_lengths'].append(self.distances[gt['scan']][start][goal])
        # new metrics
        pred = [it[0] for it in path]
        self.scores['cls'].append(self.cls[gt['scan']](pred, gt['path']))
        self.scores['sdtw'].append(self.dtw[gt['scan']](pred, gt['path'], metric='sdtw'))
        self.scores['ndtw'].append(self.dtw[gt['scan']](pred, gt['path'], metric='ndtw'))
        self.scores['dtw'].append(self.dtw[gt['scan']](pred, gt['path'], metric='dtw'))

    def score(self, output_file):
        """ Evaluate each agent trajectory based on how close it got to the goal location """
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    self._score_item(item['instr_id'], item['trajectory'])
        assert len(instr_ids) == 0, 'Trajectories not provided for %d instruction ids: %s' % (len(instr_ids), instr_ids)
        assert len(self.scores['nav_errors']) == len(self.instr_ids)
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])

        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])

        spls = []
        for err, length, sp in zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_path_lengths']):
            if err < self.error_margin:
                spls.append(sp / max(length, sp))
            else:
                spls.append(0)

        score_summary = {
            'length': np.average(self.scores['trajectory_lengths']),
            'steps': np.average(self.scores['trajectory_steps']),
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle success_rate': float(oracle_successes) / float(len(self.scores['oracle_errors'])),
            'success_rate': float(num_successes) / float(len(self.scores['nav_errors'])),
            'spl': np.average(spls),
            'cls': np.average(self.scores['cls']),
            'dtw': np.average(self.scores['dtw']),
            'ndtw': np.average(self.scores['ndtw']),
            'sdtw': np.average(self.scores['sdtw']),
        }

        assert score_summary['spl'] <= score_summary['success_rate']
        return score_summary, self.scores


class Judge:
    def __init__(self, config):
        self.results = dict()
        self.config = check_config_judge(config)
        self.env = env_list[config['action_space']](features=config['features'],
                                                    img_spec=config['img_spec'],
                                                    batch_size=config['batch_size'],
                                                    seed=config['seed'],
                                                    splits=config['splits']
                                                    )

        self.results_path = os.path.join(self.config['results_path'], 'results.json')
        self.evaluations = [Evaluation([split]) for split in config['splits']]

        self.main_split = 'val_unseen'
        self.main_metric = 'spl'

    def test(self, agent):
        agent.eval()
        self.env.reset_epoch()

        # We rely on env showing the entire batch before repeating anything
        self.results = {}
        looped = False
        while True:
            if agent.get_name() == 'Dynamic':
                trajectories, _ = agent.rollout(self.env)
            else:
                trajectories = agent.rollout(self.env)

            for traj in trajectories:
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj['path']

            if looped:
                break

        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]

        with open(self.results_path, 'w') as f:
            json.dump(output, f)

        main_metric = None

        for split, evaluation in zip(self.config['splits'], self.evaluations):
            score_summary, scores = evaluation.score(self.results_path)
            print("Agent: %s -- Split: %s" % (agent.get_name(), ",".join(evaluation.splits)))
            pp.pprint(score_summary)
            if split == self.main_split:
                assert self.main_metric in score_summary, 'Field %s not found in score_summary' % self.main_metric
                main_metric = score_summary[self.main_metric]

        return main_metric

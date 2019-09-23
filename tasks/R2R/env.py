
import sys
import csv
import numpy as np
import math
import base64
import random
import networkx as nx

from tasks.R2R.utils import load_datasets, load_nav_graphs, print_progress

sys.path.append('build')
import MatterSim


csv.field_size_limit(sys.maxsize)


def _make_id(scan_id, viewpoint_id):
    return scan_id + '_' + viewpoint_id


def load_features(feature_store):
    image_w, image_h, vfov = 640, 480, 60

    # if the tsv file for image features is provided
    if feature_store:
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
        features = {}
        with open(feature_store, "r") as tsv_in_file:
            print('Reading image features file %s' % feature_store)
            reader = list(csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames))
            total_length = len(reader)

            print('Loading image features...')
            for i, item in enumerate(reader):
                image_h = int(item['image_h'])
                image_w = int(item['image_w'])
                vfov = int(item['vfov'])
                long_id = _make_id(item['scanId'], item['viewpointId'])
                features[long_id] = np.frombuffer(base64.b64decode(item['features']),
                                                  dtype=np.float32).reshape((36, 2048))
                print_progress(i + 1, total_length, prefix='Progress:',
                               suffix='Complete', bar_length=50)
    else:
        print('Image features not provided')
        features = None

    return features, (image_w, image_h, vfov)


class EnvBatch:
    """ A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features """

    def __init__(self, features, img_spec, batch_size=100):
        self.features = features
        self.image_w, self.image_h, self.vfov = img_spec

        self.batch_size = batch_size
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setBatchSize(self.batch_size)
        self.sim.setCameraResolution(self.image_w, self.image_h)
        self.sim.setCameraVFOV(math.radians(self.vfov))
        self.sim.initialize()

    def new_episode(self, scan_ids, viewpoint_ids, headings):
        self.sim.newEpisode(scan_ids, viewpoint_ids, headings, [0] * self.batch_size)

    def get_states(self):
        """ Get list of states augmented with precomputed image features. rgb field will be empty. """
        feature_states = []
        for state in self.sim.getState():
            long_id = _make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def make_actions(self, actions):
        """ Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. """
        ix = []
        heading = []
        elevation = []
        for i, h, e in actions:
            ix.append(int(i))
            heading.append(float(h))
            elevation.append(float(e))
        self.sim.makeAction(ix, heading, elevation)

    def make_simple_actions(self, simple_indices):
        """ Take an action using a simple interface: 0-forward, 1-turn left, 2-turn right, 3-look up, 4-look down.
            All viewpoint changes are 30 degrees. Forward, look up and look down may not succeed - check state.
            WARNING - Very likely this simple interface restricts some edges in the graph. Parts of the
            environment may not longer be navigable. """
        actions = []
        for i, index in enumerate(simple_indices):
            if index == 0:
                actions.append((1, 0, 0))
            elif index == 1:
                actions.append((0, -1, 0))
            elif index == 2:
                actions.append((0, 1, 0))
            elif index == 3:
                actions.append((0, 0, 1))
            elif index == 4:
                actions.append((0, 0, -1))
            else:
                sys.exit("Invalid simple action")
        self.make_actions(actions)


class R2RBatch:
    """ Implements the Room to Room navigation task, using discretized viewpoints and pretrained features """

    def __init__(self, features, img_spec, batch_size=100, seed=10, splits='train', tokenizer=None):
        self.env = EnvBatch(features, img_spec, batch_size=batch_size)
        self.data = []
        self.scans = []

        if isinstance(splits, str):
            splits = [splits]

        assert isinstance(splits, list), 'expected type list or str type for argument "splits", found %s' % type(splits)

        print('Loading {} dataset'.format(",".join(splits)))

        json_data = load_datasets(splits)
        total_length = len(json_data)

        for i, item in enumerate(json_data):
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                self.scans.append(item['scan'])
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instructions'] = instr
                if tokenizer:
                    new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                self.data.append(new_item)
            print_progress(i + 1, total_length, prefix='Progress:', suffix='Complete', bar_length=50)
        self.scans = set(self.scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def _load_nav_graphs(self):
        """ Load connectivity graph for each scan, useful for reasoning about shortest paths """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix + self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self):
        """ Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. """
        self.ix = 0

    def _get_obs(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError


class LowLevelR2RBatch(R2RBatch):
    def __init__(self, features, img_spec, batch_size=100, seed=10, splits='train', tokenizer=None):
        super(LowLevelR2RBatch, self).__init__(features, img_spec, batch_size, seed, splits, tokenizer)

    def _shortest_path_action(self, state, goalviewpoint_id):
        """ Determine next action on the shortest path to goal, for supervised training. """
        if state.location.viewpointId == goalviewpoint_id:
            return 0, 0, 0  # do nothing
        path = self.paths[state.scanId][state.location.viewpointId][goalviewpoint_id]
        nextviewpoint_id = path[1]
        # Can we see the next viewpoint?
        for i, loc in enumerate(state.navigableLocations):
            if loc.viewpointId == nextviewpoint_id:
                # Look directly at the viewpoint before moving
                if loc.rel_heading > math.pi / 6.0:
                    return 0, 1, 0  # Turn right
                elif loc.rel_heading < -math.pi / 6.0:
                    return 0, -1, 0  # Turn left
                elif loc.rel_elevation > math.pi / 6.0 and state.viewIndex // 12 < 2:
                    return 0, 0, 1  # Look up
                elif loc.rel_elevation < -math.pi / 6.0 and state.viewIndex // 12 > 0:
                    return 0, 0, -1  # Look down
                else:
                    return i, 0, 0  # Move
        # Can't see it - first neutralize camera elevation
        if state.viewIndex // 12 == 0:
            return 0, 0, 1  # Look up
        elif state.viewIndex // 12 == 2:
            return 0, 0, -1  # Look down
        # Otherwise decide which way to turn
        pos = [state.location.x, state.location.y, state.location.z]
        target_rel = self.graphs[state.scanId].node[nextviewpoint_id]['position'] - pos
        target_heading = math.pi / 2.0 - math.atan2(target_rel[1], target_rel[0])  # convert to rel to y axis
        if target_heading < 0:
            target_heading += 2.0 * math.pi
        if state.heading > target_heading and state.heading - target_heading < math.pi:
            return 0, -1, 0  # Turn left
        if target_heading > state.heading and target_heading - state.heading > math.pi:
            return 0, -1, 0  # Turn left
        return 0, 1, 0  # Turn right

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.get_states()):
            item = self.batch[i]
            obs.append({
                'instr_id': item['instr_id'],
                'scan': state.scanId,
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'step': state.step,
                'navigableLocations': state.navigableLocations,
                'instructions': item['instructions'],
                'teacher': self._shortest_path_action(state, item['path'][-1]),
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
        return obs

    def reset(self):
        """ Load a new minibatch / episodes. """
        self._next_minibatch()
        scan_ids = [item['scan'] for item in self.batch]
        viewpoint_ids = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.new_episode(scan_ids, viewpoint_ids, headings)
        return self._get_obs()

    def step(self, actions):
        """ Take action (same interface as make_actions) """
        self.env.make_actions(actions)
        return self._get_obs()


class HighLevelR2RBatch(R2RBatch):
    def __init__(self, features, img_spec, batch_size=100, seed=10, splits='train', tokenizer=None):
        super(HighLevelR2RBatch, self).__init__(features, img_spec, batch_size, seed, splits, tokenizer)

    def _pano_navigable(self, state, goalviewpoint_id):
        """ Get the navigable viewpoints and their relative heading and elevation,
            as well as the index for 36 image features. """
        navigable_graph = self.graphs[state.scanId].adj[state.location.viewpointId]
        teacher_path = self.paths[state.scanId][state.location.viewpointId][goalviewpoint_id]

        gt_viewpoint_idx = None

        if len(teacher_path) > 1:
            next_gt_viewpoint = teacher_path[1]
        else:
            # the current viewpoint is our ground-truth
            next_gt_viewpoint = state.location.viewpointId
            gt_viewpoint_idx = (state.location.viewpointId, state.viewIndex)

        # initialize a dict to save info for all navigable points
        navigable = dict()

        # add the current viewpoint into navigable, so the agent can stay
        navigable[state.location.viewpointId] = {
            'position': [state.location.x, state.location.y, state.location.z],
            'heading': state.heading,
            'rel_heading': state.location.rel_heading,
            'rel_elevation': state.location.rel_elevation,
            'index': state.viewIndex
        }

        for viewpoint_id, weight in navigable_graph.items():
            dict_tmp = {}

            node = self.graphs[state.scanId].nodes[viewpoint_id]
            target_rel = node['position'] - [state.location.x, state.location.y, state.location.z]
            dict_tmp['position'] = list(node['position'])

            # note that this "heading" is computed regarding the global coordinate
            # the actual "heading" between the current viewpoint to next viewpoint
            # needs to take into account the current heading
            target_heading = math.pi / 2.0 - math.atan2(target_rel[1], target_rel[0])  # convert to rel to y axis
            if target_heading < 0:
                target_heading += 2.0*math.pi

            assert state.heading >= 0

            dict_tmp['rel_heading'] = target_heading - state.heading
            dict_tmp['heading'] = target_heading

            # compute the relative elevation
            dist = math.sqrt(sum(target_rel ** 2))  # compute the relative distance
            rel_elevation = np.arcsin(target_rel[2] / dist)
            dict_tmp['rel_elevation'] = rel_elevation

            # elevation level -> 0 (bottom), 1 (middle), 2 (top)
            elevation_level = round(rel_elevation / (30 * math.pi / 180)) + 1
            # To prevent if elevation degree > 45 or < -45
            elevation_level = max(min(2, elevation_level), 0)

            # viewpoint index depends on the elevation as well
            horizontal_idx = int(round(target_heading / (math.pi / 6.0)))
            horizontal_idx = 0 if horizontal_idx == 12 else horizontal_idx
            viewpoint_idx = int(horizontal_idx + 12 * elevation_level)

            dict_tmp['index'] = viewpoint_idx

            # let us get the ground-truth viewpoint index for seq2seq training
            if viewpoint_id == next_gt_viewpoint:
                gt_viewpoint_idx = (viewpoint_id, viewpoint_idx)

            # save into dict
            navigable[viewpoint_id] = dict_tmp

        return navigable, gt_viewpoint_idx

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.get_states()):
            item = self.batch[i]

            goal_viewpoint = item['path'][-1]

            # compute the navigable viewpoints and next ground-truth viewpoint
            navigable, gt_viewpoint_idx = self._pano_navigable(state, goal_viewpoint)

            # in synthetic data, path_id is unique since we only has 1 instruction per path, we will then use it as 'instr_id'
            if 'synthetic' in self.splits:
                item['instr_id'] = str(item['path_id'])

            obs.append({
                'instr_id': item['instr_id'],
                'scan': state.scanId,
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'step': state.step,
                'navigableLocations': navigable,
                'instructions': item['instructions'],
                'teacher': item['path'],
                'new_teacher': self.paths[state.scanId][state.location.viewpointId][item['path'][-1]],
                'gt_viewpoint_idx': gt_viewpoint_idx
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
                obs[-1]['instructions'] = item['instructions']
        return obs

    def reset(self):
        """ Load a new mini-batch / episodes. """
        self._next_minibatch()
        scan_ids, viewpoint_ids, headings = [], [], []
        for item in self.batch:
            scan_ids.append(item['scan'])
            viewpoint_ids.append(item['path'][0])
            headings.append(item['heading'])
        self.env.new_episode(scan_ids, viewpoint_ids, headings)
        return self._get_obs()

    def step(self, actions):
        scan_ids, viewpoint_ids, headings = actions
        self.env.new_episode(scan_ids, viewpoint_ids, headings)
        return self._get_obs()


env_list = {
    "low": LowLevelR2RBatch,
    "high": HighLevelR2RBatch,
}

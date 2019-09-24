import torch
import numpy as np
import sys

from tasks.R2R.Models import DynamicDecoder, InstructionEncoder
from tasks.R2R.utils import append_coordinates, batched_sentence_embedding, to_one_hot

sys.path.append('speaksee')
import speaksee.vocab as ssvoc


class R2RAgent:

    low_level_actions = [
      (0, -1, 0),  # left
      (0, 1, 0),   # right
      (0, 0, 1),   # up
      (0, 0, -1),  # down
      (1, 0, 0),   # forward
      (0, 0, 0),   # <end>
    ]

    def __init__(self, config):
        self.config = config
        self.name = 'Base'

    def get_name(self):
        return self.name

    def get_config(self):
        return self.config

    def rollout(self, env):
        raise NotImplementedError

    def train(self):
        """ Should call Module.train() on each torch.nn.Module, if present """
        pass

    def eval(self):
        """ Should call Module.eval() on each torch.nn.Module, if present """
        pass


class Oracle(R2RAgent):
    def __init__(self, config):
        super(Oracle, self).__init__(config)
        self.name = 'Oracle'

    def rollout(self, env):
        obs = env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))

        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = env.step(actions)
            for i, a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break

        return traj


class Stop(R2RAgent):
    def __init__(self, config):
        super(Stop, self).__init__(config)
        self.name = 'Stop'

    def rollout(self, env):
        obs = env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        return traj


class Random(R2RAgent):
    def __init__(self, config):
        super(Random, self).__init__(config)
        self.name = 'Random'

    def rollout(self, env):
        obs = env.reset()
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))

        for t in range(20):
            actions_idx = np.random.randint(0, len(R2RAgent.low_level_actions), len(obs))
            actions = [(0, 1, 0) if len(obs[i]['navigableLocations']) <= 1 and idx == R2RAgent.low_level_actions.index((1, 0, 0))
                       else R2RAgent.low_level_actions[idx] for i, idx in enumerate(actions_idx)]
            obs = env.step(actions)
            for i, a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i, ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break

        return traj


class Dynamic(R2RAgent):

    env_actions = [
        (0, -1, 0),  # left
        (0, 1, 0),   # right

        (0, 0, 1),   # up
        (0, 0, -1),  # down

        (1, 0, 0),   # forward

        (0, 0, 0),   # <end>
        (0, 0, 0),   # <start>
    ]

    def __init__(self, config):
        super(Dynamic, self).__init__(config)
        self.name = 'Dynamic'
        self.mode = None

        self.device = config['device']
        self.max_episode_len = config['max_episode_len']
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_heads = config['num_heads']
        self.glove = ssvoc.GloVe()
        self.lstm_input_size = 36 * self.num_heads + Dynamic.n_inputs()

        self.encoder = InstructionEncoder(input_size=300,
                                          hidden_size=512,
                                          use_bias=True).to(device=self.device)

        self.policy = DynamicDecoder(input_size=self.lstm_input_size,
                                     hidden_size=512, output_size=6,
                                     key_size=128, query_size=128, value_size=512,
                                     image_size=2051, filter_size=512,
                                     num_heads=self.num_heads,
                                     drop_prob=0.5,
                                     use_bias=True,
                                     filter_activation=torch.nn.Tanh(),
                                     policy_activation=torch.nn.Softmax(dim=-1)).to(device=self.device)

    @staticmethod
    def n_inputs():
        return len(Dynamic.env_actions)

    def train(self):
        self.mode = 'train'
        self.encoder.train()
        self.policy.train()

    def eval(self):
        self.mode = 'eval'
        self.encoder.eval()
        self.policy.eval()

    def save(self, encoder_path, policy_path):
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.policy.state_dict(), policy_path)

    def load(self, encoder_path, policy_path):
        pretrained_dict_encoder = torch.load(encoder_path)
        pretrained_dict_decoder = torch.load(policy_path)

        encoder_dict = self.encoder.state_dict()
        decoder_dict = self.policy.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict_encoder = {k: v for k, v in pretrained_dict_encoder.items() if k in encoder_dict}
        pretrained_dict_decoder = {k: v for k, v in pretrained_dict_decoder.items() if k in decoder_dict}

        # 2. overwrite entries in the existing state dict
        encoder_dict.update(pretrained_dict_encoder)
        decoder_dict.update(pretrained_dict_decoder)

        # 3. load the new state dict
        self.encoder.load_state_dict(pretrained_dict_encoder)
        self.policy.load_state_dict(pretrained_dict_decoder)

    def _get_targets_and_features(self, obs):
        target_actions = []
        target_idx = []
        features = []

        for i, ob in enumerate(obs):
            target_actions.append(
                ob['teacher'] if ob['teacher'] in self.env_actions else (1, 0, 0)
            )
            target_idx.append(self.env_actions.index(
                ob['teacher'] if ob['teacher'] in self.env_actions else (1, 0, 0)
            ))
            features.append(torch.from_numpy(ob['feature']))

        return target_actions, torch.tensor(target_idx), features

    def _encode_instruction(self, instructions):
        instr_embedding, instr_len = batched_sentence_embedding(instructions, self.glove, device=self.device)
        value = self.encoder(instr_embedding)
        return value

    def get_trainable_params(self):
        return list(self.encoder.parameters()) + list(self.policy.parameters())

    def rollout(self, env):

        assert self.mode is not None, "This agent contains trainable modules! Please call either agent.train() or agent.eval() before rollout"
        assert self.mode in ['train', 'eval'], "Agent.mode expected to be in ['train', 'eval'], found %s" % self.mode

        obs = env.reset()
        ended = np.array([False] * len(obs))
        losses = []

        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        instr = [ob['instructions'] for ob in obs]
        value = self._encode_instruction(instr)

        target_actions, target_idx, features = self._get_targets_and_features(obs)
        previous_action = to_one_hot([Dynamic.n_inputs() - 1] * len(obs), Dynamic.n_inputs())  # Action at t=0 is <start> for every agent

        for t in range(self.max_episode_len):

            image_features = torch.stack(
                [append_coordinates(features[i], ob['heading'], ob['elevation']) for i, ob in enumerate(obs)]
            ).to(device=self.device)

            pred, logits, response_map = self.policy(image_features, value, previous_action, init_lstm_state=t == 0)

            """ Losses """
            step_loss = self.criterion(pred, target_idx.to(device=self.device))
            losses.append(step_loss)

            """ Performs steps """
            # Mask outputs where agent can't move forward
            probs = logits.clone().detach().to(device=torch.device('cpu'))
            for i, ob in enumerate(obs):
                if len(ob['navigableLocations']) <= 1:
                    probs[i, self.env_actions.index((1, 0, 0))] = 0.

            if self.mode == 'eval':
                _, a_t = probs.max(1)  # argmax
                actions = [self.env_actions[idx] for idx in a_t]
            else:
                m = torch.distributions.Categorical(probs)  # sampling from distribution
                a_t = m.sample()
                actions = [self.env_actions[idx] if target_actions[i] != (0, 0, 0) else (0, 0, 0) for i, idx in enumerate(a_t)]

            """ Next step """
            obs = env.step(actions)

            for i, ob in enumerate(obs):
                if not ended[i]:
                    if actions[i] == (0, 0, 0):
                        ended[i] = True
                    else:
                        traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            if ended.all():
                break

            target_actions, target_idx, features = self._get_targets_and_features(obs)
            previous_action = to_one_hot(a_t, self.n_inputs())

        """ Compute the loss for the whole rollout """
        losses = torch.stack(losses).to(device=self.device)
        rollout_loss = torch.mean(losses)

        return traj, rollout_loss

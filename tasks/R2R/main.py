
import argparse

import sys
import os
import torch
import torch.optim as optim
import numpy as np

sys.path.append('/homes/flandi/cvpr2020')

from tasks.R2R.Agents import get_agent
from tasks.R2R.env import load_features
from tasks.R2R.trainer import Trainer
from tasks.R2R.eval import Judge


parser = argparse.ArgumentParser(description='PyTorch for Matterport3D Agent with Dynamic Convolutional Filters')

# General options
parser.add_argument('--name', type=str, default='custom_experiment',
                    help='name for the experiment')
parser.add_argument('--results_dir', type=str, default='tasks/R2R/results',
                    help='home directory for results')
parser.add_argument('--feature_store', type=str, default='img_features/ResNet-152-imagenet.tsv',
                    help='feature store file')
parser.add_argument('--eval_only', action="store_true",
                    help='if true, does not train the model before evaluating')
parser.add_argument('--seed', type=int, default=42,
                    help='initial random seed')
# Training options
parser.add_argument('--num_epoch', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--eval_every', type=int, default=5,
                    help='number of training epochs between evaluations')
parser.add_argument('--patience', type=int, default=30,
                    help='number of epochs to wait before early stopping')
parser.add_argument('--lr', type=float, default=0.001,
                    help='base learning rate')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
# Agent options
parser.add_argument('--num_heads', type=int, default=1,
                    help='number of heads for multi-headed dynamic convolution')
parser.add_argument('--max_episode_len', type=int, default=20,
                    help='agent max number of steps before stopping')


""" Device info """
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Running on device: {}'.format(device))


def main(opts):

    space = 'low'
    splits = 'train'
    results_path = os.path.join(opts.results_dir, opts.name)
    features, img_spec = load_features(opts.feature_store)

    agent_config = {'action_space': space,
                    'max_episode_len': opts.max_episode_len,
                    'num_heads': opts.num_heads,
                    'device': device,
                    }

    trainer_config = {'action_space': space,
                      'features': features,
                      'img_spec': img_spec,
                      'splits': splits,
                      'batch_size': opts.batch_size,
                      'seed': opts.seed,
                      'results_path': results_path,
                      }

    judge_config = {'action_space': space,
                    'features': features,
                    'img_spec': img_spec,
                    'splits': ['val_seen', 'val_unseen'],
                    'batch_size': opts.batch_size,
                    'seed': opts.seed,
                    'results_path': results_path,
                    }

    agent = get_agent('Dynamic', agent_config)
    judge = Judge(judge_config)

    if opts.eval_only:
        agent.load(os.path.join(results_path, 'encoder_weights_best'),
                   os.path.join(results_path, 'decoder_weights_best'))
        metric = judge.test(agent)
        print('Main metric result for this test: {:.4f}'.format(metric))
    else:
        trainer = Trainer(trainer_config)
        optimizer = optim.Adam(agent.get_trainable_params(), lr=opts.lr)
        best = trainer.train(agent, optimizer, opts.num_epoch, patience=opts.patience, eval_every=opts.eval_every, judge=judge)
        print('Best metric result for this test: {:.4f}'.format(best))

    print('----- End -----')


if __name__ == '__main__':
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.results_dir, args.name)):
        print('WARNING: Experiment with this name already exists! - {}'.format(args.name))
    else:
        os.makedirs(os.path.join(args.results_dir, args.name))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)

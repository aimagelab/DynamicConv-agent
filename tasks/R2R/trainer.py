
import os

from tasks.R2R.env import LowLevelR2RBatch
from tasks.R2R.utils import check_config_trainer, print_progress


class Trainer:
    def __init__(self, config):
        self.results = dict()
        self.config = check_config_trainer(config)
        self.env = LowLevelR2RBatch(features=config['features'],
                                    img_spec=config['img_spec'],
                                    batch_size=config['batch_size'],
                                    seed=config['seed'],
                                    splits=config['splits']
                                    )
        print('Success!')

    def _train_epoch(self, agent, optimizer, num_iter):
        epoch_loss = 0.
        self.env.reset_epoch()

        for it in range(num_iter):
            optimizer.zero_grad()
            _, loss = agent.rollout(self.env)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            suffix_msg = 'Running Loss: {:.4f}'.format(epoch_loss / (it+1))
            print_progress(it, num_iter, suffix=suffix_msg)
        else:
            suffix_msg = 'Running Loss: {:.4f}'.format(epoch_loss / num_iter)
            print_progress(num_iter, num_iter, suffix=suffix_msg)

        return epoch_loss / num_iter

    def train(self, agent, optimizer, num_epoch, num_iter_epoch=None, patience=None, eval_every=None, judge=None):
        best_metric = 0.
        agent.train()

        if num_iter_epoch is None:
            num_iter_epoch = len(self.env.data) // self.env.batch_size + 1
        if eval_every is None:
            if judge is None:
                eval_every = num_epoch + 1  # Never tested
            else:
                eval_every = num_epoch  # Test only on the last epoch
        if patience is None:
            patience = num_epoch
        reset_patience = patience

        for epoch in range(num_epoch):
            mean_loss = self._train_epoch(agent, optimizer, num_iter_epoch)
            print("Epoch {}/{} terminated: Epoch Loss = {:.4f}".format(epoch+1, num_epoch, mean_loss))
            agent.save(os.path.join(self.config['results_path'], 'encoder_weights_last'),
                       os.path.join(self.config['results_path'], 'decoder_weights_last'))

            if (epoch+1) % eval_every == 0:
                metric = judge.test(agent)
                if metric is not None:
                    print('Main metric results for this test: {:.4f}'.format(metric))
                    if metric > best_metric:
                        best_metric = metric
                        patience = reset_patience
                        print('New best! Saving weights...')
                        agent.save(os.path.join(self.config['results_path'], 'encoder_weights_best'),
                                   os.path.join(self.config['results_path'], 'decoder_weights_best'))
                    else:
                        patience -= 1
                        if patience == 0:
                            print('{} epochs without improvement in main metric ({}) - patience is over!'.format(reset_patience, judge.main_metric))
                            break

        print("Finishing training")
        return best_metric

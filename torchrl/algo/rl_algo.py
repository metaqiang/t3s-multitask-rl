import copy
import pickle
import time
from collections import deque
import numpy as np

import torch

import torchrl.algo.utils as atu

import gym

import os
import os.path as osp
import wandb

from torchrl.task_scheduler import TaskScheduler
import torchrl.plot_history as plot_history
from tqdm import tqdm


TASK_SAMPLE_NUM = int(os.getenv('TASK_SAMPLE_NUM', '10'))
RESTORE = int(os.getenv('RESTORE', '0'))
ID = os.getenv('ID', 'null')
DIR = os.getenv('DIR', 'null')
EPOCH = int(os.getenv('EPOCH', '0'))


class RLAlgo():
    """
    Base RL Algorithm Framework
    """

    def __init__(self,
                 env=None,
                 replay_buffer=None,
                 collector=None,
                 logger=None,
                 continuous=None,
                 discount=0.99,
                 num_epochs=3000,
                 epoch_frames=1000,
                 max_episode_frames=999,
                 batch_size=128,
                 device='cpu',
                 train_render=False,
                 eval_episodes=1,
                 eval_render=False,
                 save_interval=100,
                 save_dir=None
                 ):

        self.env = env

        self.continuous = isinstance(self.env.action_space, gym.spaces.Box)

        self.replay_buffer = replay_buffer
        self.collector = collector
        # device specification
        self.device = device

        # environment relevant information
        self.discount = discount
        self.num_epochs = num_epochs
        self.epoch_frames = epoch_frames
        self.max_episode_frames = max_episode_frames

        self.train_render = train_render
        self.eval_render = eval_render

        # training information
        self.batch_size = batch_size
        self.training_update_num = 0
        self.sample_key = None

        # Logger & relevant setting
        self.logger = logger

        self.episode_rewards = deque(maxlen=30)
        self.training_episode_rewards = deque(maxlen=30)
        self.eval_episodes = eval_episodes

        self.save_interval = save_interval
        self.save_dir = save_dir
        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.best_eval = None

    def start_epoch(self):
        pass

    def finish_epoch(self):
        return {}

    def pretrain(self):
        pass

    def update_per_epoch(self):
        pass

    def snapshot(self, prefix, epoch):
        print(f'snapshot at {epoch}')

        for name, network in self.snapshot_networks:
            model_file_name = "model_{}_{}.pth".format(name, epoch)
            model_path = osp.join(prefix, model_file_name)
            torch.save(network.state_dict(), model_path)
        
        torch.save(self.log_alpha, osp.join(prefix, "log_alpha_{}.pth".format(epoch)))

        with open(osp.join(prefix, "replay_buffer.pkl"), 'wb') as f:
            pickle.dump(self.replay_buffer, f)

    def train(self):
        global EPOCH

        if RESTORE:
            for name, network in self.snapshot_networks:
                model_file_name = "model_{}_{}.pth".format(name, EPOCH)
                model_path = osp.join(self.save_dir, model_file_name)
                network.load_state_dict(torch.load(model_path))

            self.log_alpha = torch.load(osp.join(self.save_dir, "log_alpha_{}.pth".format(EPOCH)))

            EPOCH += 1

            wandb.init(
                name=os.environ['NAME'],
                project='multitask-yyq',
                group=os.environ['GROUP'],
                reinit=True,
                id=ID,
                dir='./log',
                resume="allow" if RESTORE else None,
            )
        else:
            wandb.init(
                name=os.environ['NAME'],
                project='multitask-yyq',
                group=os.environ['GROUP'],
                reinit=True,
                dir='./log',
            )

        self.pretrain()

        total_frames = 0
        if hasattr(self, "pretrain_frames"):
            total_frames = self.pretrain_frames

        #*
        self.start_epoch()
        task_scheduler = TaskScheduler(num_tasks=10, task_sample_num=TASK_SAMPLE_NUM)

        for epoch in tqdm(range(EPOCH, self.num_epochs)):
            log_dict = {}

            self.current_epoch = epoch
            start = time.time()

            for _ in range(task_scheduler.num_tasks // TASK_SAMPLE_NUM):
                task_sample_index = task_scheduler.sample()

                self.start_epoch()

                explore_start_time = time.time()

                training_epoch_info = self.collector.train_one_epoch(
                    task_sample_index)

                for reward in training_epoch_info["train_rewards"]:
                    self.training_episode_rewards.append(reward)
                explore_time = time.time() - explore_start_time

                train_start_time = time.time()
            self.update_per_epoch(task_sample_index, task_scheduler)

            train_time = time.time() - train_start_time

            finish_epoch_info = self.finish_epoch()

            eval_start_time = time.time()
            eval_infos = self.collector.eval_one_epoch()

            task_scheduler.update_success_rate_array(eval_infos)
            task_scheduler.update_return_array(eval_infos)
            task_scheduler.update_p()

            eval_time = time.time() - eval_start_time

            total_frames += self.collector.active_worker_nums * self.epoch_frames

            infos = {}

            for reward in eval_infos["eval_rewards"]:
                self.episode_rewards.append(reward)
            # del eval_infos["eval_rewards"]

            if self.best_eval is None or \
                    np.mean(eval_infos["eval_rewards"]) > self.best_eval:
                self.best_eval = np.mean(eval_infos["eval_rewards"])
                self.snapshot(self.save_dir, 'best')
            del eval_infos["eval_rewards"]

            infos["Running_Average_Rewards"] = np.mean(self.episode_rewards)
            infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
            infos["Running_Training_Average_Rewards"] = np.mean(
                self.training_episode_rewards)
            infos["Explore_Time"] = explore_time
            infos["Train___Time"] = train_time
            infos["Eval____Time"] = eval_time
            infos.update(eval_infos)
            infos.update(finish_epoch_info)

            log_dict['mean_success_rate'] = infos['mean_success_rate']

            self.logger.add_epoch_info(epoch, total_frames,
                                       time.time() - start, infos)

            if epoch % self.save_interval == 0:
                self.snapshot(self.save_dir, epoch)

                task_scheduler.save(self.save_dir)
                # filename = plot_history.plot(root_dir=self.save_dir, num_tasks=task_scheduler.num_tasks, sample_gap=1, perfermance_gap=10, delta_gap=10)
                # log_dict['history'] = wandb.Image(filename)

            wandb.log(log_dict)

        self.snapshot(self.save_dir, "finish")
        self.collector.terminate()

    def update(self, batch):
        raise NotImplementedError

    def _update_target_networks(self):
        if self.use_soft_update:
            for net, target_net in self.target_networks:
                atu.soft_update_from_to(net, target_net, self.tau)
        else:
            if self.training_update_num % self.target_hard_update_period == 0:
                for net, target_net in self.target_networks:
                    atu.copy_model_params_from_to(net, target_net)

    @property
    def networks(self):
        return [
        ]

    @property
    def snapshot_networks(self):
        return [
        ]

    @property
    def target_networks(self):
        return [
        ]

    def to(self, device):
        for net in self.networks:
            net.to(device)

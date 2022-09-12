from collections import defaultdict
import torch
import numpy as np
import os

###
task_name_list_10 = ['reach-v1', 'push-v1', 'pick-place-v1', 'door-v1', 'drawer-open-v1', 'drawer-close-v1',
                  'button-press-topdown-v1', 'ped-insert-side-v1', 'window-open-v1', 'window-close-v1']

task_name_list_50 = ['reach-v1', 'push-v1', 'pick-place-v1', 'reach-wall-v1', 'pick-place-wall-v1', 'push-wall-v1', 'door-open-v1', 'door-close-v1', 'drawer-open-v1', 'drawer-close-v1', 'button-press_topdown-v1', 'button-press-v1', 'button-press-topdown-wall-v1', 'button-press-wall-v1', 'peg-insert-side-v1', 'peg-unplug-side-v1', 'window-open-v1', 'window-close-v1', 'dissassemble-v1', 'hammer-v1', 'plate-slide-v1', 'plate-slide-side-v1', 'plate-slide-back-v1', 'plate-slide-back-side-v1', 'handle-press-v1', 'handle-pull-v1', 'handle-press-side-v1', 'handle-pull-side-v1', 'stick-push-v1', 'stick-pull-v1', 'basket-ball-v1', 'soccer-v1', 'faucet-open-v1', 'faucet-close-v1', 'coffee-push-v1', 'coffee-pull-v1', 'coffee-button-v1', 'sweep-v1', 'sweep-into-v1', 'pick-out-of-hole-v1', 'assembly-v1', 'shelf-place-v1', 'push-back-v1', 'lever-pull-v1', 'dial-turn-v1', 'bin-picking-v1', 'box-close-v1', 'hand-insert-v1', 'door-lock-v1', 'door-unlock-v1']
###

###
SCHEDULER_MODE = int(os.getenv('SCHEDULER_MODE', '0'))
print(f'SCHEDULER_MODE: {SCHEDULER_MODE}')
###

class TaskScheduler():
    def __init__(self, num_tasks=10, task_sample_num=5, k=0.1, alpha=0.5, n=3, sample_gap=1):
        if num_tasks == 10:
            self.task_name_list = task_name_list_10
        elif num_tasks == 50:
            self.task_name_list = task_name_list_50
        else:
            raise ValueError

        self.num_tasks = num_tasks
        self.task_sample_num = task_sample_num
        self.k = k
        self.alpha = alpha
        self.n = n
        self.sample_gap = sample_gap

        self.success_rate_array = np.zeros(self.num_tasks)
        self.success_rate_array_smooth = np.zeros(self.num_tasks)
        self.return_array = np.zeros(self.num_tasks)
        self.return_array_smooth = np.zeros(self.num_tasks)
        
        self.p = torch.softmax(torch.ones(num_tasks), dim=0).numpy()

        self.sample_history = []

        self.return_array_history = []
        self.return_array_smooth_history = []
        self.return_array_dict = defaultdict(list)

        self.success_rate_array_history = []
        self.success_rate_array_smooth_history = []
        self.success_rate_array_dict = defaultdict(list)

        self.delta_success_rate_history = []
        self.delta_return_history = []

        self.sample_count = 0

    def update_p(self):
        p1 = torch.softmax(torch.from_numpy(1 - self.success_rate_array_smooth) / self.k, dim=0).numpy()
        # p1 = torch.softmax(torch.from_numpy(
        #     1 - self.return_array_smooth / 5000) / self.k, dim=0).numpy()
        # p2 = torch.softmax(torch.from_numpy(-(self.return_array - self.last_return_array) / 5000) / self.k, dim=0).numpy()

        if SCHEDULER_MODE == 0:
            self.p = p1
        elif SCHEDULER_MODE == 1:
            self.p = 0.8 * p1 + (1 - 0.8) * p2
        elif SCHEDULER_MODE == 2:
            self.p = self.alpha * p1 + (1 - self.alpha) * p2
        elif SCHEDULER_MODE == 3:
            self.p = p2
        else:
            raise ValueError

        self.print_info('p: {}'.format([round(i, 2) for i in self.p]))

    def update_success_rate_array(self, eval_log_dict):
        self.last_success_rate_array = self.success_rate_array.copy()
        self.success_rate_array = np.zeros(self.num_tasks)

        ###
        for i, task_name in enumerate(self.task_name_list):
            self.success_rate_array[i] = eval_log_dict[f'{task_name}_success_rate']
            self.success_rate_array_dict[i].append(eval_log_dict[f'{task_name}_success_rate'])
        ###

        self.success_rate_array_smooth = np.array(
            [np.mean(self.success_rate_array_dict[i][-self.n:]) for i in range(self.num_tasks)])

        self.delta_success_rate_history.append(self.success_rate_array - self.last_success_rate_array)

        self.print_info('success_rate_array: {}\nsuccess_rate_array_smooth:{}'.format([round(
            i, 2) for i in self.success_rate_array], [round(i, 2) for i in self.success_rate_array_smooth]))

        self.success_rate_array_history.append(self.success_rate_array)
        self.success_rate_array_smooth_history.append(
            self.success_rate_array_smooth)

    def update_return_array(self, eval_log_dict):
        self.last_return_array = self.return_array.copy()
        self.return_array = np.zeros(self.num_tasks)

        ###
        for i in range(self.num_tasks):
            cur_return = eval_log_dict[self.task_name_list[i] + '_eval_rewards']

            self.return_array[i] = cur_return
            self.return_array_dict[i].append(cur_return)
        ###
        
        self.return_array_smooth = np.array(
            [np.mean(self.return_array_dict[i][-self.n:]) for i in range(self.num_tasks)])

        self.delta_return_history.append(self.return_array - self.last_return_array)

        self.print_info('return_array: {}\nreturn_array_smooth: {}'.format([round(
            i, 2) for i in self.return_array], [round(i, 2) for i in self.return_array_smooth]))

        self.return_array_history.append(self.return_array)
        self.return_array_smooth_history.append(self.return_array_smooth)

    def sample(self):
        if self.sample_count % self.sample_gap == 0:
            self.ids = np.random.choice(list(range(
                self.num_tasks)), replace=False, p=self.p, size=self.task_sample_num).tolist()

        # self.ids = np.random.choice(list(range(
        #     self.num_tasks)), replace=False, p=torch.softmax(torch.ones(self.num_tasks), dim=0).numpy(), size=self.task_sample_num).tolist()

        # self.ids = [0]

        print('sample: {}'.format(self.ids))
        self.sample_history.append(self.ids)

        self.sample_count += 1
        return self.ids

    def save(self, root_dir):
        os.makedirs(root_dir, exist_ok=True)
        
        np.save(os.path.join(root_dir, 'sample_history.npy'),
                np.array(self.sample_history))
        np.save(os.path.join(root_dir, 'return_array_history.npy'),
                np.array(self.return_array_history))
        np.save(os.path.join(root_dir, 'return_array_smooth_history.npy'),
                np.array(self.return_array_smooth_history))
        np.save(os.path.join(root_dir, 'success_rate_array_history.npy'),
                np.array(self.success_rate_array_history))
        np.save(os.path.join(root_dir, 'success_rate_array_smooth_history.npy'), np.array(
            self.success_rate_array_smooth_history))
        np.save(os.path.join(root_dir, 'delta_success_rate_history.npy'), np.array(
            self.delta_success_rate_history))
        np.save(os.path.join(root_dir, 'delta_return_history.npy'), np.array(
            self.delta_return_history))

        print(f'history save at {root_dir}')

    @staticmethod
    def print_info(info):
        print(f'\033[1;36;40m{info}\033[0m')

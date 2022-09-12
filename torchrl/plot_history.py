import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import math


def get_performence_sum(matric, task_idx, s_idx, e_idx):
    return np.sum(matric[s_idx:e_idx, task_idx])


def get_performence_mean(matric, task_idx, s_idx, e_idx):
    return np.mean(matric[s_idx:e_idx, task_idx])


def plot(root_dir, num_tasks, sample_gap=1, perfermance_gap=1, delta_gap=1, show=False, percent=1):
    def count_sample(task_idx, s_idx, e_idx):
        total_count = 0
        task_count = 0

        if e_idx > sample_history.shape[0]:
            e_idx = sample_history.shape[0]

        for i in range(s_idx, e_idx):
            for idx in sample_history[i, :]:
                total_count += 1
                if idx == task_idx:
                    task_count += 1

        if total_count == 0:
            return 0

        return task_count / total_count

    sample_history = np.load(os.path.join(root_dir, 'sample_history.npy'))
    sample_history = sample_history[:int(sample_history.shape[0] * percent), :]
    print(f'shape of sample_history: {sample_history.shape}')

    return_array_history = np.load(os.path.join(root_dir, 'return_array_history.npy'))
    return_array_history = return_array_history[:int(return_array_history.shape[0] * percent), :]
    print(f'shape of return_array_history: {return_array_history.shape}')    

    success_rate_array_history = np.load(os.path.join(root_dir, 'success_rate_array_history.npy'))
    success_rate_array_history = success_rate_array_history[:int(success_rate_array_history.shape[0] * percent), :]
    print(f'shape of success_rate_array_history: {success_rate_array_history.shape}')    

    delta_success_rate_history = np.load(os.path.join(root_dir, 'delta_success_rate_history.npy'))
    delta_success_rate_history = delta_success_rate_history[:int(delta_success_rate_history.shape[0] * percent), :]
    print(f'shape of delta_success_rate_history: {delta_success_rate_history.shape}')    

    delta_return_history = np.load(os.path.join(root_dir, 'delta_return_history.npy'))
    delta_return_history = delta_return_history[:int(delta_return_history.shape[0] * percent), :]
    print(f'shape of delta_return_history: {delta_return_history.shape}') 

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(figsize = (6, 9), nrows=5)

    # get sample_history
    data_sample_to_show = None
    for i in range(math.ceil(sample_history.shape[0] / sample_gap)):
        new_column = np.zeros(num_tasks).reshape(-1, 1)

        for idx in range(num_tasks):
            new_column[idx, 0] = count_sample(idx, i*sample_gap, (i+1)*sample_gap)

        if data_sample_to_show is None:
            data_sample_to_show = new_column
        else:
            data_sample_to_show = np.hstack((data_sample_to_show, new_column))

    sns.heatmap(data_sample_to_show, cmap = 'RdBu', center=0, ax=ax1, xticklabels=False)
    ax1.set_title('Task Sample History')

    # plot success_rate_array_history
    data_return_to_show = None
    for i in range(math.ceil(success_rate_array_history.shape[0] / perfermance_gap)):
        new_column = np.zeros(num_tasks).reshape(-1, 1)

        for idx in range(num_tasks):
            new_column[idx, 0] = get_performence_mean(success_rate_array_history, idx, i*perfermance_gap, (i+1)*perfermance_gap)

        if data_return_to_show is None:
            data_return_to_show = new_column
        else:
            data_return_to_show = np.hstack((data_return_to_show, new_column))

    sns.heatmap(data_return_to_show, cmap = 'RdBu_r', center=0, ax=ax2, xticklabels=False)
    ax2.set_title('Success Rate History')

    # plot delta_success_rate_history
    data_delta_success_rate_history_to_show = None
    for i in range(math.ceil(delta_success_rate_history.shape[0] / delta_gap)):
        new_column = np.zeros(num_tasks).reshape(-1, 1)

        for idx in range(num_tasks):
            new_column[idx, 0] = get_performence_sum(delta_success_rate_history, idx, i*delta_gap, (i+1)*delta_gap)

        if data_delta_success_rate_history_to_show is None:
            data_delta_success_rate_history_to_show = new_column
        else:
            data_delta_success_rate_history_to_show = np.hstack((data_delta_success_rate_history_to_show, new_column))

    sns.heatmap(data_delta_success_rate_history_to_show, cmap="PiYG", center=0, ax=ax3, xticklabels=False)
    ax3.set_title('Delta Success Rate History')

    # plot return_array_history
    data_return_to_show = None
    for i in range(math.ceil(return_array_history.shape[0] / perfermance_gap)):
        new_column = np.zeros(num_tasks).reshape(-1, 1)

        for idx in range(num_tasks):
            new_column[idx, 0] = get_performence_mean(return_array_history, idx, i*perfermance_gap, (i+1)*perfermance_gap)

        if data_return_to_show is None:
            data_return_to_show = new_column
        else:
            data_return_to_show = np.hstack((data_return_to_show, new_column))

    sns.heatmap(data_return_to_show, cmap = 'RdBu_r', center=0, ax=ax4, xticklabels=False)
    ax4.set_title('Return History')

    # plot delta_return_history
    delta_return_history_to_show = None
    for i in range(math.ceil(delta_return_history.shape[0] / delta_gap)):
        new_column = np.zeros(num_tasks).reshape(-1, 1)

        for idx in range(num_tasks):
            new_column[idx, 0] = get_performence_sum(delta_return_history, idx, i*delta_gap, (i+1)*delta_gap)

        if delta_return_history_to_show is None:
            delta_return_history_to_show = new_column
        else:
            delta_return_history_to_show = np.hstack((delta_return_history_to_show, new_column))

    sns.heatmap(delta_return_history_to_show, cmap="PiYG", center=0, ax=ax5, xticklabels=False)
    ax5.set_title('Delta Return History')


    plt.tight_layout()
    
    if show:
        plt.show()

    filename = os.path.join(root_dir, f'history.png')

    plt.savefig(filename)
    print(f'history pic save at {filename}')

    return filename


if __name__ == '__main__':
    # plot(root_dir='./model/embodiedai/multitask/mtan_run_rand_as_sr_k5_seed1-jdxg226', sample_gap=1, perfermance_gap=1, delta_gap=1, show=False)
    # plot(root_dir='./model/embodiedai/multitask/custom_v2_seed1-wmlf285', sample_gap=10 * 20, perfermance_gap=10, delta_gap=10, show=False, percent=1)
    # plot(root_dir='./tmp/MT10_MTSAC_k5', sample_gap=10 * 12 * 2, perfermance_gap=10 * 12, delta_gap=10 * 12, show=False, percent=1)
    pass

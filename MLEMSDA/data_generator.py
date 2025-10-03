import torch
import numpy as np


def task_generator(data, label, iteration):
    srcdat = data
    srclbl = label

    num_tasks = 7
    query_set_size = 80


    def create_tasks(srcdat, srclbl, num_tasks, query_set_size):
        total_samples = srcdat.shape[0]
        support_set_size = total_samples - query_set_size

        tasks_data = []
        tasks_labels = []

        for task_idx in range(num_tasks):
            start_idx = task_idx * query_set_size
            end_idx = start_idx + query_set_size

            query_set_data = srcdat[start_idx:end_idx]
            query_set_labels = srclbl[start_idx:end_idx]

            support_set_data = torch.cat([srcdat[:start_idx], srcdat[end_idx:]], dim=0)
            support_set_labels = torch.cat([srclbl[:start_idx], srclbl[end_idx:]], dim=0)

            indices = torch.randperm(support_set_data.size(0))
            shuffled_spt_data = support_set_data[indices]
            shuffled_spt_labels = support_set_labels[indices]

            indices1 = torch.randperm(query_set_data.size(0))
            shuffled_qur_data = query_set_data[indices1]
            shuffled_qur_labels = query_set_labels[indices1]


            task_data = (shuffled_spt_data, shuffled_qur_data)
            task_labels = (shuffled_spt_labels, shuffled_qur_labels)
            tasks_data.append(task_data)
            tasks_labels.append(task_labels)

        return tasks_data, tasks_labels


    tasks_data, tasks_labels = create_tasks(srcdat=srcdat, srclbl=srclbl, num_tasks=num_tasks,
                                            query_set_size=query_set_size)


    if iteration == 0:
        print("Task 1 - Support Set Data:")
        print(tasks_data[0][0].shape)
        print("Task 1 - Query Set Data:")
        print(tasks_data[0][1].shape)
        print("Task 1 - Support Set Labels:")
        print(tasks_labels[0][0].shape)
        print("Task 1 - Query Set Labels:")
        print(tasks_labels[0][1].shape)

    return [tasks_data, tasks_labels]

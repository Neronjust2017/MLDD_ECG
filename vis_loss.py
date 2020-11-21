import numpy as np
import csv
import os
import matplotlib.pyplot as plt


def plot(losses, title=None):
    fig = plt.figure(figsize=(10, 4))
    labels = ['loss_baseline', 'loss_meta_train', 'loss_meta_val']
    for i, loss in enumerate(losses):
        plt.plot(loss, label=labels[i])
    plt.title(title)
    plt.legend(labels)
    plt.show()

run_ids = ['1', '2', '3', '4']
log_ids = ['0', '1', '2', '3']

root = './'
# root = '../MLDG/'

for run_id in run_ids:
    for log_id in log_ids:
        loss_b = np.loadtxt(root+'run_%s/logs_%s/loss_log.txt' %(run_id, log_id))
        loss_m = np.loadtxt(root+'run_%s/logs_mldg_%s/loss_log.txt' %(run_id, log_id))
        loss_m_train = loss_m[:,0]
        loss_m_val = loss_m[:,1]
        # plot([loss_b], title='run_%s/logs_%s/' % (run_id, log_id))
        plot([loss_b, loss_m_train, loss_m_val], title='run_%s/logs_%s/'%(run_id, log_id))
        # plot([loss_b, loss_m_train], title='run_%s/logs_%s/' % (run_id, log_id))

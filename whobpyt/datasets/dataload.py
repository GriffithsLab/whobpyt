"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
function for data preparation

emp: length_ts x node_size or data_size x length_ts x node_size
"""


'''
import numpy as np


def dataloader(emp, epoch_size, TRperwindow):
    window_size = int(emp.shape[0] / TRperwindow)
    data_out = 0
    if len(emp.shape) == 2:
        node_size = emp.shape[1]
        length_ts = emp.shape[0]
        window_size = int(length_ts / TRperwindow)
        data_out = np.zeros((epoch_size, window_size, node_size, TRperwindow))
        for i_epoch in range(epoch_size):
            for i_win in range(window_size):
                data_out[i_epoch, i_win, :, :] = emp.T[:, i_win * TRperwindow:(i_win + 1) * TRperwindow]
    if len(emp.shape) == 3:
        node_size = emp.shape[2]
        length_ts = emp.shape[1]
        data_size = emp.shape[0]
        window_size = int(length_ts / TRperwindow)
        data_out = np.zeros((epoch_size, window_size, node_size, TRperwindow))
        for i_epoch in range(epoch_size):
            for i_win in range(window_size):
                data_out[i_epoch, i_win, :, :] = \
                    emp[i_epoch % data_size, i_win * TRperwindow:(i_win + 1) * TRperwindow, :].T
    return data_out
'''
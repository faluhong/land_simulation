
import numpy as np
import os
from os.path import join
import sys


pwd = os.getcwd()
rootpath_project = os.path.abspath(os.path.join(pwd, '../..'))
path_pythoncode = join(rootpath_project, 'pythoncode')
sys.path.append(path_pythoncode)


def get_inverse_transition_matrix(transition_prob_matrix_accumulate_direct):
    """
    get the transition matrix based on the current year, i.e., normalize the current transition matrix to identify matrix
    """
    transition_prob_matrix_accumulate_direct_inverse = np.zeros((np.shape(transition_prob_matrix_accumulate_direct)),
                                                                dtype=float)
    for i in range(0, np.shape(transition_prob_matrix_accumulate_direct)[0]):
        transition_prob_matrix_accumulate_direct_inverse[i] = transition_prob_matrix_accumulate_direct[i] @ \
                                                              np.linalg.pinv(
                                                                  transition_prob_matrix_accumulate_direct[-1])

    return transition_prob_matrix_accumulate_direct_inverse


def read_prob_matrix(predict_flag, country_flag='hispaniola'):
    """
        read the observed change matrix
        :param predict_flag:
        :param country_flag:
        :return:
    """

    path_change_matrix = join(rootpath_project, 'results', 'change_matrix')

    output_adjacent_prob_matrix = join(path_change_matrix, '{}_{}_adjacent_matrix.npy'.format(predict_flag, country_flag))
    transition_prob_matrix_adjacent = np.load(output_adjacent_prob_matrix)

    output_accumulate_prob_matrix_indirect = join(path_change_matrix, '{}_{}_accumulate_matrix_indirect.npy'.format(predict_flag,
                                                                                                country_flag))
    transition_prob_matrix_accumulate_indirect = np.load(output_accumulate_prob_matrix_indirect)

    output_accumulate_prob_matrix_direct = join(path_change_matrix, '{}_{}_accumulate_matrix_direct.npy'.format(predict_flag, country_flag))
    transition_prob_matrix_accumulate_direct = np.load(output_accumulate_prob_matrix_direct)

    transition_prob_matrix_accumulate_direct_inverse = get_inverse_transition_matrix(transition_prob_matrix_accumulate_direct)

    return (transition_prob_matrix_adjacent, transition_prob_matrix_accumulate_indirect,
            transition_prob_matrix_accumulate_direct, transition_prob_matrix_accumulate_direct_inverse)

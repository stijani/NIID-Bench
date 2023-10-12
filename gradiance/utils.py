import copy
import torch


def get_avg_of_unbiased_grads(list_of_gradient_dicts):
    """get the average of the unbiased grads
    """
    average_gradient_dict = {}
    num_dicts = len(list_of_gradient_dicts)

    for gradient_dict in list_of_gradient_dicts:
        for layer_name, gradient in gradient_dict.items():
            if layer_name in average_gradient_dict:
                average_gradient_dict[layer_name] += gradient
            else:
                average_gradient_dict[layer_name] = gradient

    for layer_name in average_gradient_dict.keys():
        average_gradient_dict[layer_name] /= num_dicts
    return average_gradient_dict


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(len(w)))
    return w_avg
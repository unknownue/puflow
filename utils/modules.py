
import torch
import torch.nn as nn


def freeze_module(module: nn.Module):
    """
    Freeze all paramaters for any network
    """
    for name, param in module.named_parameters():
        param.requires_grad = False


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def get_num_parameters(model):
    """
    Returns the number of trainable parameters in a model of type nn.Module
    :param model: nn.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def init_params(module: nn.Module, method: str):
    for key, param in module.named_parameters():
        if param.requires_grad and len(param.shape) > 1:
            if method == 'xavier':
                param.data = torch.randn(param.shape) * 0.030
                # nn.init.xavier_uniform(param.data)
            else:
                print(f'Unsupport layer init methods: {method}')


def print_progress_log(epoch: int, logs: dict, extra=None):
    console_print = f'\x1b[2K\rEpoch {epoch:3}:'
    console_print += ''.join(f" [{key}]{value:5.3f}" for key, value in logs.items())

    if extra is not None:
        if isinstance(extra, str):
            console_print += '| ' + extra
        elif isinstance(extra, list) and len(extra) > 0:
            console_print += '  | ' + "".join(f' {info}' for info in extra)

    print(console_print)


def get_tf_num_parameters(model):
    import tensorflow as tf
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)


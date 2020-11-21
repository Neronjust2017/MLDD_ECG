import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable


def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True)[0].data, requires_grad=False)

            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.linear(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt)
    else:
        return F.linear(inputs, weight, bias)


def conv1d(inputs, weight, bias, meta_step_size=0.001, stride=1, padding=0, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:

            # gradient of weight w.r.t. meta_loss
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True)[0]

            if bias is not None:

                # gradient of bias w.r.t. meta_loss
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True)[0].data,
                                   requires_grad=False)
            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        # meta_step_size: hyperparamter α
        return F.conv1d(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt, stride,
                        padding,
                        dilation, groups)
    else:
        return F.conv1d(inputs, weight, bias, stride, padding, dilation, groups)

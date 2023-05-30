import pickle as pkl
import json

from nemo.utils import logging


import math

import torch

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType


class GlowTTSLoss(Loss):
    """
    Loss for the GlowTTS model
    """

    @property
    def input_types(self):
        return {
            "z": NeuralType(('B', 'D', 'T'), NormalDistributionSamplesType()),
            "y_m": NeuralType(('B', 'D', 'T'), NormalDistributionMeanType()),
            "y_logs": NeuralType(('B', 'D', 'T'), NormalDistributionLogVarianceType()),
            "logdet": NeuralType(('B',), LogDeterminantType()),
            "logw": NeuralType(('B', 'T'), TokenLogDurationType()),
            "logw_": NeuralType(('B', 'T'), TokenLogDurationType()),
            "x_lengths": NeuralType(('B',), LengthsType()),
            "y_lengths": NeuralType(('B',), LengthsType()),
            "stoch_dur_loss": NeuralType(optional=True),
        }

    @property
    def output_types(self):
        return {
            "l_mle": NeuralType(elements_type=LossType()),
            "l_length": NeuralType(elements_type=LossType()),
            "logdet": NeuralType(elements_type=VoidType()),
        }

    @typecheck()
    def forward(self, z, y_m, y_logs, logdet, logw, logw_, x_lengths, y_lengths, stoch_dur_loss,):

        logdet = torch.sum(logdet)
        l_mle = 0.5 * math.log(2 * math.pi) + (
            torch.sum(y_logs) + 0.5 * torch.sum(torch.exp(-2 * y_logs) * (z - y_m) ** 2) - logdet
        ) / (torch.sum(y_lengths) * z.shape[1])

        if stoch_dur_loss is None:
            l_length = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)
        else:
            l_length = stoch_dur_loss
        return l_mle, l_length, logdet
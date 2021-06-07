# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .BERTKD import AdvBERTKD

def setup(opt):
    if opt.model == "bert_adv_kd":
        model =AdvBERTKD(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model

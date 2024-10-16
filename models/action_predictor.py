import torch
import torch.nn as nn
import math
from config import Constants

__all__ = ("predict_action")

class predict_action(nn.Module):
    def __init__(self, opt, key_name="verb"):
        super(predict_action, self).__init__()
        # print(opt["is_verb_decoder"])
        if "is_verb_decoder" in opt.keys():
            self.net = nn.Sequential(
                nn.Linear(opt['dim_hidden'], opt['dim_hidden']),
                nn.ReLU(),
                nn.Dropout(opt['hidden_dropout_prob']),
                nn.Linear(opt['dim_hidden'], opt['verb_num']),
                nn.Sigmoid()
            )
            self.proj = nn.Linear(opt["verb_num"], opt["dim_hidden"])
            self.key_name = key_name
            print("这里是ap。py的第22行")

    def forward(self, enc_output, gt_many_hot_verb=None, **kwargs):
        if isinstance(enc_output, list):
            assert len(enc_output) == 1
            enc_output = enc_output[0]
        assert len(enc_output.shape) == 3

        apperance = enc_output[:, 0:8, :]
        motion = enc_output[:, 8:16, :]
        out = self.net(enc_output.mean(1))
        # out = self.net(apperance.mean(1))
        # out = self.net(motion.mean(1))
        out1 = self.proj(out)
        if gt_many_hot_verb != None:
            out2 = self.proj(gt_many_hot_verb)
            return {self.key_name: [out, out1, out2]}
        else:
            return {self.key_name: [out, out1]}

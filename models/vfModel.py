import torch
import os
import copy
import torch.nn as nn
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .base_model import BaseModel
from models import getModels


class Aasm(BaseModel):

    def __init__(self, args, **kwargs):
        super(Aasm, self).__init__(args)
        self.args = args

        if not args.get('Arch', False):
            raise RuntimeError("Model initialization failed!")

        if not (args.Arch.get('vNet', False) and args.Arch.get('aNet', False) and args.Arch.get('eqm', False)):
            raise RuntimeError("The model components are incomplete!")

        self.vNet = getModels.getModel(args.Arch.vNet)
        self.aNet = getModels.getModel(args.Arch.aNet)
        self.eqm = getModels.getModel(args.Arch.eqm)

        for cpt in args.Arch.keys():
            model = getattr(self, cpt)
            if model.args.get('freeze', False):
                for p in model.parameters():
                    p.requires_grad = False

        self.corr_num = nn.Parameter(torch.FloatTensor(924), requires_grad=False)
        nn.init.constant_(self.corr_num, 0.0)
        self.compare_sum = nn.Parameter(torch.FloatTensor(924), requires_grad=False)
        nn.init.constant_(self.compare_sum, 0.0)

    '''
        a REFERS TO AUDIO, v REFERS TO VISUAL
    '''
    def forward(self, samples, iter=None, is_triple=False, V2F=True, **kwargs):

        if is_triple:
            sample, sample_p, sample_n = samples
            if V2F:
                x = self.aNet(sample['audio'])
                x_p = self.vNet(self.normalize(sample_p['image']))
                x_n = self.vNet(self.normalize(sample_n['image']))
            else:
                x = self.vNet(self.normalize(sample['image']))
                x_p = self.aNet(sample_p['audio'])
                x_n = self.aNet(sample_n['audio'])

            dist_pos = cosine_distance(x, x_p)
            dist_neg = cosine_distance(x, x_n)

            pred = dist_pos > dist_neg
            out = {
                'dist_pos': dist_pos,
                'dist_neg': dist_neg,
                'pred': pred.long(),
                'target': torch.ones_like(pred).long().to(pred.device)
            }
            return [out]

        v = self.normalize(samples['image'])
        v_aux = self.normalize(samples['image_aux'])
        a = samples['audio']
        a_aux = samples['audio_aux']

        rd = np.random.choice(range(int(2.5*16000), int(5.0*16000)))
        rd_start = np.random.choice(range(0, a.shape[1] - rd))
        a = a[:,rd_start:rd + rd_start]

        rd = np.random.choice(range(int(2.5 * 16000), int(5.0 * 16000)))
        rd_start = np.random.choice(range(0, a_aux.shape[1] - rd))
        a_aux = a_aux[:, rd_start:rd + rd_start]

        y = samples['ID']
        z_v = self.vNet(v)
        z_a = self.aNet(a)
        z_v_aux = self.vNet(v_aux)
        z_a_aux = self.aNet(a_aux)

        logit_cross, trip_cross, ACR = self.cal(z_v, z_a, True, True, iter).values()
        v_intra, trip_v = self.cal(z_v, z_v_aux, False, False, iter).values()
        a_intra, trip_a = self.cal(z_a, z_a_aux, False, False, iter).values()

        if self.args.weighted_enabled:
            w = self.eqm.get_weight(y)
        else:
            w = None

        outputs = []

        trip_cro = {
            'loss': trip_cross * 10 if iter >= self.args.regauge_iter else None,
            'loss_type': 'zero_avoid',
            'weight': w,
            'loss_name': 'trip_cross'
        }

        trip_intra_v = {
            'loss': trip_v * 10 if iter >= self.args.regauge_iter else None,
            'loss_type': 'zero_avoid',
            'weight': w,
            'loss_name': 'trip_intra_v'
        }
        trip_intra_a = {
            'loss': trip_a * 10 if iter >= self.args.regauge_iter else None,
            'loss_type': 'zero_avoid',
            'weight': w,
            'loss_name': 'trip_intra_a'
        }
        outputs.append(trip_cro)
        outputs.append(trip_intra_v)
        outputs.append(trip_intra_a)

        out_sim = {
            'loss': v_intra + a_intra,
            'loss_type': 'wtd',
            'loss_name': 'sim_loss'
        }
        outputs.append(out_sim)

        out_cross = {
            'loss': logit_cross,
            'loss_type': 'wtd',
            'loss_name': 'crosssp_loss'
        }
        outputs.append(out_cross)

        return outputs


    def cal(self, modal1, modal2, isCross=False, hasACR=False, iter=None):

        npl, tpl, ACR = self.eqm.cross_logit(modal1, modal2, is_cross=isCross, iter=iter)
        npl_, tpl_, _ = self.eqm.cross_logit(modal2, modal1, is_cross=isCross, iter=iter)
        logit_cross = 0.5 * npl + 0.5 * npl_
        nonzero = (tpl != 0).float()
        nonzero_ = (tpl_ != 0).float()
        dived = nonzero + nonzero_
        trip_cross = tpl + tpl_
        non_zero = (trip_cross != 0)
        if torch.count_nonzero(non_zero) != 0:
            trip_cross[non_zero] /= dived[non_zero]
        ret = {}
        ret['npl'] = logit_cross
        ret['tpl'] = trip_cross
        if hasACR:
            ret['ACR'] = ACR
        return ret


# def flushWeight(self):
#     self.eqm.flush()

def Euclidean_distance(x, y):
    return ((x - y)**2).sum(-1)

def cosine_distance(x, y):
    return (F.normalize(x) * F.normalize(y)).sum(-1)
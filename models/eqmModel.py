import torch
import math
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .base_model import BaseModel
from .vfModel import cosine_distance

class eqm(BaseModel):
    def __init__(self, args):
        super(eqm, self).__init__(args)
        num_classes = self.args.num_classes

        self.corr_num = nn.Parameter(torch.FloatTensor(num_classes), requires_grad=False)
        nn.init.constant_(self.corr_num, 0.0)
        self.compare_sum = nn.Parameter(torch.FloatTensor(num_classes), requires_grad=False)
        nn.init.constant_(self.compare_sum, 0.0)

        self.weight = nn.Parameter(torch.FloatTensor(num_classes), requires_grad=False)
        nn.init.constant_(self.weight, 1)

        self.set_metrics()

    def set_metrics(self, metrics = cosine_distance):
        self.metrics = metrics

    def forward(self, ACR, y):
        self.corr_num[y] += ACR['corr']
        self.compare_sum[y] += ACR['sum']

    def flush(self):
        '''
            Dynamic strategy
        '''
        corr_ratio = self.corr_num / self.compare_sum

        sigma = corr_ratio.std()
        miu = corr_ratio.mean() + (-1.0) * sigma
        sigma = math.pow(1, 0.5) * sigma

        Wei_distr = torch.distributions.normal.Normal(loc = miu, scale = sigma)

        nn.init.constant_(self.weight, 0.0)
        weight_sum = 0
        for i in range(self.args.num_classes):
            self.weight[i] = Wei_distr.log_prob(corr_ratio[i]).exp()
            weight_sum += self.weight[i]

        # clr
        nn.init.constant_(self.corr_num, 0.0)
        nn.init.constant_(self.compare_sum, 0.0)

    def get_weight(self, y):
        w = torch.index_select(self.weight, 0, y)
        return w

    def cross_logit(self, x, v, is_cross = False, iter = None):

        dist = self.dist(F.normalize(x).unsqueeze(0), v.unsqueeze(1))
        one_hot = torch.zeros(dist.size()).to(x.device)
        one_hot.scatter_(1, torch.arange(len(x)).view(-1, 1).long().to(x.device), 1)

        pos = (one_hot * dist).sum(-1, keepdim=True)
        logit = (1.0 - one_hot) * (dist - pos) * (1.0 / self.args.temp)
        if is_cross:
            loss = torch.log(1 + torch.exp(logit).sum(-1) + 3.4)
        else:
            loss = torch.log(2 + torch.exp(logit).sum(-1) + 3)

        dist_fake = self.dist(F.normalize(x).unsqueeze(0), F.normalize(v).unsqueeze(1))
        homo_dist = self.dist(F.normalize(x).unsqueeze(0), F.normalize(x).unsqueeze(1))

        one_hot = torch.zeros(dist_fake.size()).to(x.device)
        one_hot.scatter_(1, torch.arange(len(x)).view(-1, 1).long().to(x.device), 1)
        pos_fake = (one_hot * dist_fake).sum(-1, keepdim=True)
        neg_fake = (1.0 - one_hot) * dist_fake
        cond1 = pos_fake > self.args.m_u

        if self.args.H:
            logit_fake = (1.0 - one_hot) * (dist_fake - pos_fake)
            trip_score = logit_fake + (1.0 - one_hot) * self.args.m_l
        else:
            alphap = torch.clamp_min(- pos_fake.detach() + 1 + self.args.m_l, min=0.)
            alphan = torch.clamp_min(neg_fake.detach() + self.args.m_l, min=0.) * (1.0 - one_hot)

            delta_p = 1 - self.args.m_l
            delta_n = self.args.m_l

            logit_p = - alphap * (pos_fake - delta_p)
            logit_n = alphan * (neg_fake - delta_n)
            trip_score = (logit_n + logit_p) * (1.0 - one_hot)

        '''
            REFINEMENT of trip_score
        '''
        is_selected = (trip_score > 0).float()
        add_dis1 = (homo_dist + 0.15 - pos_fake) * is_selected
        add_dis1 = torch.clamp(add_dis1, 0, 1e8)
        add_dis2 = (dist_fake - homo_dist + 0.15) * is_selected
        add_dis2 = torch.clamp(add_dis2, 0, 1e8)
        add_dis = add_dis1 + add_dis2
        trip_score = torch.clamp(trip_score, 0, 1e8)
        trip_score += add_dis

        exi_trip = torch.count_nonzero(trip_score != 0, dim=1)
        cond2 = exi_trip.reshape(trip_score.shape[0], -1).bool()

        logit_list = list()
        for i in range(len(trip_score)):
            if exi_trip[i] == 0:
                logit_list.append(0)
                continue
            else:
                logit_sum = 0
                for j in range(len(trip_score[i])):
                    if trip_score[i][j] != 0:
                        logit_sum += trip_score[i][j]
                logit_list.append(logit_sum / exi_trip[i])
        logit_list = torch.tensor(logit_list).reshape(trip_score.shape[0], -1).to(trip_score.device)
        cond = (cond1 & cond2).int()
        logit_list = logit_list * cond


        '''
            ACR
        '''
        sub_dis = (dist_fake - pos_fake) * (1.0 - one_hot)
        corr_nums = torch.count_nonzero(sub_dis < 0, dim=1)
        compare_sum = dist_fake.shape[-1] - 1
        ACR = {
            'corr' : corr_nums,
            'sum': compare_sum
        }

        return loss, logit_list, ACR

    def dist(self, a, b):
        dist = (a * b).sum(-1)
        return dist
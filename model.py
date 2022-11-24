"""SGRAF model"""

import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np
from collections import OrderedDict

from torch.nn import init

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, 1024)
        self.fc1 = nn.Linear(1024, embed_size)
        self.fc2 = nn.Sequential(nn.Linear(1024+512, embed_size))
        
        #self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb1 = self.fc(images)
        img_emb2 = self.fc1(img_emb1)
        img_emb = torch.cat((img_emb1, img_emb2), dim=-1)
        img_emb = self.fc2(img_emb)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """

    def __init__(self, opt):
        super(EncoderText, self).__init__()

        self.bert = BertModel.from_pretrained('../SGRAF/uncased_L-12_H-768_A-12/')
        print('text-encoder-bert fine-tuning !')
        self.embed_size = opt.embed_size
        self.fc = nn.Sequential(nn.Linear(768 *4, 1024), nn.ReLU(), nn.Dropout(0.1))
        self.LN = nn.LayerNorm(self.embed_size)
        self.fc1 =nn.Linear(1024, self.embed_size)



    def forward(self, captions, lengths):
        all_encoders, pooled = self.bert(captions)
        out = all_encoders[-1]
        out1 = all_encoders[-2]
        out2 = all_encoders[-3]
        out3 = all_encoders[-4]
        outs = torch.cat((out, out1, out2, out3), dim=-1)

        outs = self.fc(outs)
        out = self.fc1(outs)
        out = self.LN(out)
        cap_lens = lengths
        return out, cap_lens



class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """
    def __init__(self, embed_size, sim_dim=256, sgr_step=3):
        super(EncoderSimilarity, self).__init__()


        self.init_weights()
        self.k = embed_size
        self.h_dim = 1
        self.h_out = 9
        self.h_mat = nn.Parameter(torch.Tensor(1, self.h_out, 1, self.h_dim * self.k).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, self.h_out, 1, 1).normal_())
        self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        self.softmax = nn.Softmax(dim=1)
        self.fcc1 = nn.Linear(embed_size, embed_size)
        self.fcc2 = nn.Linear(embed_size, embed_size)
        self.fcc3 = nn.Linear(embed_size, embed_size)


    def forward(self, img_emb, cap_emb, cap_lens):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, img_emb, cap_i_expand)) + self.h_bias
            p = nn.functional.softmax(logits.view(-1, self.h_out, logits.size(2) * logits.size(3)), 2)
            p = p.view(-1, self.h_out, logits.size(2), logits.size(3))

            for g in range(p.size(1) - 1):
                if g == 0:
                    l1 = torch.einsum('bvk,bvq,bqk->bk', (img_emb, p[:, 0, :, :], cap_i_expand))
                    l2 = torch.einsum('bvk,bvq,bqk->bk', (img_emb, p[:, 1, :, :], cap_i_expand))
                    query = l1
                    key = torch.tanh(l2)
                    G = query + key
                    query = self.fcc1(G)
                    key = self.fcc2(G)
                    query = torch.sigmoid(query)
                    key = torch.sigmoid(key)
                    a = torch.tanh(l1) * query + l1
                    b = self.fcc3(l2) * key + l2
                    lts = a + b
                    lts = l2norm(lts, dim=-1)
                else:
                    l1 = torch.einsum('bvk,bvq,bqk->bk', (img_emb, p[:, g+1, :, :], cap_i_expand))
                    l2 = lts
                    query = l1
                    key = torch.tanh(l2)
                    G = query + key
                    query = self.fcc1(G)
                    key = self.fcc2(G)
                    query = torch.sigmoid(query)
                    key = torch.sigmoid(key)
                    a = torch.tanh(l1) * query + l1
                    b = self.fcc3(l2) * key + l2
                    lts = a + b
                    lts = l2norm(lts, dim=-1)

            b_emb = lts.unsqueeze(1)  # b x 1 x d
            b_emb = self.p_net(b_emb).squeeze(1) * self.k
            sim_all.append(b_emb)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)


        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        # print('this si sss:',scores.shape)
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class SGRAF(object):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt)
        self.sim_enc = EncoderSimilarity(opt.embed_size, opt.sim_dim,
                                         opt.module_name)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())

        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.sim_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])
        total = sum(p.numel() for p in self.params)

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs, cap_lens = self.txt_enc(captions, lengths)

        return img_embs, cap_embs, cap_lens

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sims

    def forward_loss(self, sims, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims)
        self.logger.update('Loss', loss.item(), sims.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        # print('thsis mmmL:', max(lengths))
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()


def get_optimizer(params, opt, t_total=-1):
    bertadam = BertAdam(params, lr=opt.learning_rate, warmup=opt.warmup, t_total=t_total)
    return bertadam
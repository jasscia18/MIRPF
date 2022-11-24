"""Data provider"""

import torch
import torch.utils.data as data

import os
import nltk
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, tokenizer):
        #self.vocab = vocab
        loc = data_path + '/'

        # load the raw captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        # load the image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        self.tokenizer = tokenizer
        self.max_seq_len = 32

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index/self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        #print('this si ca:', caption)

        # vocab = self.vocab
        #
        # # convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(
        #     str(caption).lower())
        # caption = []
        # caption.append(vocab('<start>'))
        # caption.extend([vocab(token) for token in tokens])
        # caption.append(vocab('<end>'))
        # target = torch.Tensor(caption)

        target = self.get_text_input(caption)

        #target = torch.Tensor(target.float()).long()
        # print('this is sit:',target, target.shape)
        # #print('this is lengths:', lengths)

        return image, target, index, img_id

    def __len__(self):
        return self.length

    def get_text_input(self, caption):
        caption_tokens = self.tokenizer.tokenize(caption)
        caption_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)
        # print('this is ll:', caption_ids)
        #lengths = len(caption_ids)

        # if len(caption_ids) >= self.max_seq_len:
        #     caption_ids = caption_ids[:self.max_seq_len]
        # else:
        #     caption_ids = caption_ids + [0] * (self.max_seq_len - len(caption_ids))
        caption = torch.tensor(caption_ids)

        return caption#, lengths


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids= zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)
    #targets = torch.stack(captions, 0)
    # les = [cap for cap in lengths]
    # print('thissi ll:',les)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    ls = [len(cap) for cap in captions]
    # print('thissi ls:', ls)
    # print('thissi max(ls):', max(ls))
    targets = torch.zeros(len(captions), max(ls)).long()
    for i, cap in enumerate(captions):
        end = ls[i]
        targets[i, :end] = cap[:end]

    #print('this targets:', targets.shape)
    #lengths = torch.Tensor(lengths)

    return images, targets, ls, ids

def get_tokenizer(bert_path):
    tokenizer = BertTokenizer(bert_path + 'vocab.txt')
    return tokenizer

def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    bert_path = '../SGRAF/uncased_L-12_H-768_A-12/'
    dset = PrecompDataset(data_path, data_split, tokenizer=get_tokenizer(bert_path))

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    100, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     100, False, workers)
    return test_loader

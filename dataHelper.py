# -*- coding: utf-8 -*-

import os
import numpy as np
import string
from collections import Counter
import pandas as pd
from tqdm import tqdm
import random
import time

import torch
from torch.autograd import Variable

from PWWS.read_files import split_imdb_files, split_yahoo_files, split_agnews_files, split_snli_files
from PWWS.word_level_process import word_process, get_tokenizer, update_tokenizer, text_process_for_single, label_process_for_single, text_process_for_single_bert
from PWWS.neural_networks import get_embedding_index, get_embedding_matrix

from PWWS.paraphrase import generate_synonym_list_from_word, generate_synonym_list_by_dict, get_syn_dict
from PWWS.config import config

from torchvision.datasets.vision import VisionDataset

import torch.utils.data

from codecs import open
try:
    import cPickle as pickle
except ImportError:
    import pickle

class SynthesizedData(torch.utils.data.Dataset):
    
    def __init__(self, opt, x, y, syn_data):
        super(SynthesizedData, self).__init__()
        self.x = x.copy()
        self.y = y.copy()
        self.syn_data = syn_data.copy()

        for x in range(len(self.syn_data)):
            self.syn_data[x] = [syn_word for syn_word in self.syn_data[x] if syn_word!=x]

        self.len_voc = len(self.syn_data)+1

    def transform(self, sent, label, anch, pos, neg, anch_valid):
       
        return torch.tensor(sent,dtype=torch.long), torch.tensor(label,dtype=torch.long),torch.tensor(anch,dtype=torch.long),torch.tensor(pos,dtype=torch.long),torch.tensor(neg,dtype=torch.long),torch.tensor(anch_valid,dtype=torch.float)

    def __getitem__(self, index, max_num_anch_per_sent=100, num_pos_per_anch=20, num_neg_per_anch=100):
        sent = self.x[index]
        label = self.y[index].argmax()

        #for x in sent:
        #    self.syn_data[x] = [syn_word for syn_word in self.syn_data[x] if syn_word!=x]
        #try:
        sent_for_anch = [x for x in sent if x>0 and x<len(self.syn_data) and len(self.syn_data[x]) != 0]
        #except:
        #    print(index)
        #while(len(sent_for_anch) < max_num_anch_per_sent):
        #    sent_for_anch.extend(sent_for_anch)
        
        if len(sent_for_anch) > max_num_anch_per_sent:
            anch = random.sample(sent_for_anch, max_num_anch_per_sent)
        else:
            anch = sent_for_anch

        anch_valid = [1 for x in anch]

        pos = []
        neg = []
        for word in anch:
            syn_set = set(self.syn_data[word])
            if len(self.syn_data[word]) == 0:
                pos.append([word for i in range(num_pos_per_anch)])
            elif len(self.syn_data[word]) < num_pos_per_anch:
                    temp = []
                    for i in range( int(num_pos_per_anch/len(self.syn_data[word])) + 1):
                        temp.extend(self.syn_data[word])
                    #pos.append(temp[:num_pos_per_anch])
                    pos.append(random.sample(temp, num_pos_per_anch))
            elif len(self.syn_data[word]) >= num_pos_per_anch:
                pos.append(random.sample(self.syn_data[word], num_pos_per_anch))

            count=0
            temp = []
            while (count<num_neg_per_anch):
                while (1):
                    neg_word = random.randint(0, self.len_voc)
                    if neg_word not in syn_set:
                        break
                temp.append(neg_word)
                count+=1
            neg.append(temp)

        while(len(anch)<max_num_anch_per_sent):
            anch.append(0)
            anch_valid.append(0)
            pos.append([0 for i in range(num_pos_per_anch)])
            neg.append([0 for i in range(num_neg_per_anch)])

        return self.transform(sent, label, anch, pos, neg, anch_valid)

    def __len__(self):
        return len(self.y)

class SynthesizedData_TextLikeSyn_Bert(SynthesizedData):
    
    def __init__(self, opt, x, y, syn_data, seq_max_len, tokenizer, attack_surface=None, given_y=None):
        self.opt = opt
        self.x = x.copy()
        self.y = y.copy() #y is onehot
        self.syn_data = syn_data.copy()
        self.seq_max_len = seq_max_len
        self.tokenizer = tokenizer

        self.dataset_ids = []
        for i in range(len(x)):
            self.dataset_ids.append(i)

        label = [y.argmax() for y in self.y]

        # for sample idx
        self.cls_positive = [[] for i in range(opt.label_size)]
        for i in range(len(x)):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(opt.label_size)]
        for i in range(opt.label_size):
            for j in range(opt.label_size):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(opt.label_size)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(opt.label_size)]
        #

        if attack_surface is not None:
            xx=[]
            yy=[]
            did=[]
            for i, text in enumerate(self.x):
                if attack_surface.check_in(text.split(' ')):
                    xx.append(text)
                    yy.append(self.y[i])
                    did.append(self.dataset_ids[i])
            self.x = xx
            self.y = yy
            self.dataset_ids = did

        if given_y is not None:
            xx=[]
            yy=[]
            did=[]
            for i, label in enumerate(self.y):
                if self.y[i].argmax()==given_y:
                    xx.append(self.x[i])
                    yy.append(self.y[i])
                    did.append(self.dataset_ids[i])
            self.x = xx
            self.y = yy
            self.dataset_ids = did

    def transform(self, sent, label, text_like_syn, text_like_syn_valid, mask, token_type_ids, dataset_id, sample_idx):
       
        return torch.tensor(sent,dtype=torch.long), torch.tensor(label,dtype=torch.long), torch.tensor(text_like_syn,dtype=torch.long), torch.tensor(text_like_syn_valid,dtype=torch.float), torch.tensor(mask, dtype = torch.long), torch.tensor(token_type_ids, dtype = torch.long), torch.tensor(dataset_id, dtype = torch.long), torch.tensor(sample_idx, dtype = torch.long)

    def __getitem__(self, index, num_text_like_syn=10, K=1000):

        if self.opt.bert_type=="bert":
            encoded = self.tokenizer.encode_plus(self.x[index], None, add_special_tokens = True, max_length = self.seq_max_len, pad_to_max_length = True)
        elif self.opt.bert_type=="roberta":
            encoded = self.tokenizer(self.x[index], None, max_length = self.seq_max_len, pad_to_max_length = True)
            encoded["token_type_ids"] = [0 for i in range(len(encoded["attention_mask"]))]

        elif self.opt.bert_type=="xlnet":
            encoded = self.tokenizer(self.x[index], None, max_length = self.seq_max_len, pad_to_max_length = True)
            

        sent = encoded["input_ids"]
        mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"]

        text_like_syn=[]
        text_like_syn_valid=[]
        for token in sent:
            text_like_syn_valid.append([])
            text_like_syn.append([token])

        splited_words = self.x[index].split()

        if self.opt.bert_type=="bert" or self.opt.bert_type=="roberta":
            for i in range(min(self.seq_max_len-2, len(splited_words))):
                word = splited_words[i]
                if word in self.syn_data:
                    text_like_syn[i+1].extend(self.syn_data[word])
        elif self.opt.bert_type=="xlnet":
            start = self.seq_max_len-sum(mask)
            for i in range(min(self.seq_max_len-2, len(splited_words))):
                word = splited_words[i]
                if word in self.syn_data:
                    text_like_syn[i+start].extend(self.syn_data[word])

        label = self.y[index].argmax()
        dataset_id = self.dataset_ids[index]

        for i, x in enumerate(sent):
            text_like_syn_valid[i] = [1 for times in range(len(text_like_syn[i]))]
            
            while(len(text_like_syn[i])<num_text_like_syn):
                text_like_syn[i].append(0)
                text_like_syn_valid[i].append(0)

            assert(len(text_like_syn[i])==num_text_like_syn)
            assert(len(text_like_syn_valid[i])==num_text_like_syn)


        replace = True if K > len(self.cls_negative[label]) else False
        neg_idx = np.random.choice(self.cls_negative[label], K, replace=replace)
        sample_idx = np.hstack((np.asarray([dataset_id]), neg_idx))

        return self.transform(sent, label, text_like_syn, text_like_syn_valid, mask, token_type_ids, dataset_id, sample_idx)


def imdb_bert_make_synthesized_iter_giveny(opt):
    dataset=opt.dataset
    opt.label_size = 2
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)

    from modified_bert_tokenizer import ModifiedBertTokenizer
    tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    if opt.synonyms_from_file:
        filename= opt.imdb_bert_synonyms_file_path
        f=open(filename,'rb')
        saved=pickle.load(f)
        f.close()
        syn_data = saved["syn_data"]
        #x_train=saved['x_train']
        #x_test=saved['x_test']
        #x_dev=saved['x_dev']
        y_train=saved['y_train']
        y_test=saved['y_test']
        y_dev=saved['y_dev']

    else:
        #tokenizer = get_tokenizer(opt)
        #print("len of tokenizer before updata.", len(tokenizer.index_word))
        print("Preparing synonyms.")

        syn_dict = get_syn_dict(opt)
        syn_data = {} # key is textual word

        # Tokenize syn data
        print("Tokenize syn data.")
        for key in syn_dict:
            if len(syn_dict[key])!=0:
                temp = tokenizer.encode_plus(syn_dict[key], None, add_special_tokens=False, pad_to_max_length=False)['input_ids']

                token_of_key = tokenizer.encode_plus(key, None, add_special_tokens=False, pad_to_max_length=False)["input_ids"][0]

                syn_data[key] = temp

                #if not token_of_key in syn_data:
                #    syn_data[token_of_key] = temp
                #else:
                #    syn_data[token_of_key].append(temp)

        # Tokenize the training data
        print("Tokenize training data.")
        #x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        
        #x_train = text_process_for_single_bert(tokenizer, train_texts, opt.dataset)
        y_train = label_process_for_single(tokenizer, train_labels, opt.dataset)

        #x_dev = text_process_for_single_bert(tokenizer, dev_texts, opt.dataset)
        y_dev = label_process_for_single(tokenizer, dev_labels, opt.dataset)

        #x_test = text_process_for_single_bert(tokenizer, test_texts, opt.dataset)
        y_test = label_process_for_single(tokenizer, test_labels, opt.dataset)
 

        filename= opt.imdb_bert_synonyms_file_path
        f=open(filename,'wb')
        saved={}
        saved['syn_data']=syn_data
        #saved['x_train']=x_train
        #saved['x_test']=x_test
        #saved['x_dev']=x_dev
        saved['y_train']=y_train
        saved['y_test']=y_test
        saved['y_dev']=y_dev
        pickle.dump(saved,f)
        f.close()

    from PWWS.config import config
    seq_max_len = config.word_max_len[dataset]

    opt.n_training_set = len(train_texts)

    train_data_y0 = SynthesizedData_TextLikeSyn_Bert(opt, train_texts, y_train, syn_data, seq_max_len, tokenizer, given_y=0)
    train_data_y0.__getitem__(0)
    train_loader_y0 = torch.utils.data.DataLoader(train_data_y0, opt.batch_size//2, shuffle=True, num_workers=16)

    train_data_y1 = SynthesizedData_TextLikeSyn_Bert(opt, train_texts, y_train, syn_data, seq_max_len, tokenizer, given_y=1)
    train_loader_y1 = torch.utils.data.DataLoader(train_data_y1, opt.batch_size//2, shuffle=True, num_workers=16)

    # use training data as dev
    dev_data = SynthesizedData_TextLikeSyn_Bert(opt, dev_texts, y_dev, syn_data, seq_max_len, tokenizer)
    dev_loader = torch.utils.data.DataLoader(dev_data, opt.test_batch_size, shuffle=False, num_workers=16)

    from from_certified.attack_surface import WordSubstitutionAttackSurface, LMConstrainedAttackSurface
    attack_surface = LMConstrainedAttackSurface.from_files(opt.certified_neighbors_file_path, opt.imdb_lm_file_path)

    test_data = SynthesizedData_TextLikeSyn_Bert(opt, test_texts, y_test, syn_data, seq_max_len, tokenizer, attack_surface=attack_surface)
    test_loader = torch.utils.data.DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=16)
    return train_loader_y0, train_loader_y1, test_loader

# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import pandas as pd
from six.moves import cPickle
import time,os,random
import itertools

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import NLLLoss,MultiLabelSoftMarginLoss,MultiLabelMarginLoss,BCELoss

from dataHelper import imdb_bert_make_synthesized_iter_giveny
from PWWS.fool_pytorch import genetic_attack

import opts
import models
import utils

from solver.lr_scheduler import WarmupMultiStepLR

try:
    import cPickle as pickle
except ImportError:
    import pickle

def set_params(net, resume_model_path, data_parallel=False, bert=False):
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume_model_path), 'Error: ' + resume_model_path + 'checkpoint not found!'
    checkpoint = torch.load(resume_model_path)
    state_dict = checkpoint['net']
    from collections import OrderedDict
    sdict = OrderedDict()
    for key in state_dict.keys():
        if data_parallel:
            new_key = key
        else:
            if 'hidden2label' in key:
                new_key = key
            else:
                key1, key2 = key.split('module.')[0], key.split('module.')[1]
                new_key = key1+key2

        sdict[new_key]=state_dict[key]
    try:
        net.load_state_dict(sdict)
    except:
        print("WARNING!!!!!!!! MISSING PARAMETERS. LOADING ANYWAY.")
        net.load_state_dict(sdict,strict=False)
    return net

def train(opt, train_iter_y0, train_iter_y1, test_iter, verbose=True):
    global_start= time.time()
    logger = utils.getLogger()
    model=models.setup(opt)

    from from_certified.attack_surface import WordSubstitutionAttackSurface, LMConstrainedAttackSurface
    if opt.lm_constraint:
        attack_surface = LMConstrainedAttackSurface.from_files(opt.certified_neighbors_file_path, opt.imdb_lm_file_path)
    else:
        attack_surface = WordSubstitutionAttackSurface.from_files(opt.certified_neighbors_file_path, opt.imdb_lm_file_path)
    
    if opt.resume != None:
        model = set_params(model, opt.resume, data_parallel=True, bert=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()
        #model=torch.nn.DataParallel(model)

    params = [param for param in model.parameters() if param.requires_grad] 
    from transformers import AdamW
    no_decay = ['bert_stu','bert_tea']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': opt.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    if opt.optimizer=='adamw':
        optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate)
    else:
        optimizer = utils.getOptimizer(optimizer_grouped_parameters,name=opt.optimizer, lr=opt.learning_rate,weight_decay=opt.weight_decay)

    if opt.optimizer=='sgd':
        scheduler = WarmupMultiStepLR(optimizer, (10, 15), 0.1, 1.0/10.0, 2, 'linear')
    else:
        scheduler = WarmupMultiStepLR(optimizer, (50, 80), 0.1, 1.0/10.0, 2, 'linear')

    from modified_bert_tokenizer import ModifiedBertTokenizer

    tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    loss_fun = F.cross_entropy

    start= time.time()
    best_adv_acc = 0
    best_save_dir = None

    for epoch in range(opt.imdb_epochs):

        sum_loss  = sum_loss_kl = sum_loss_clean  = sum_loss_mi_giveny_adv = sum_loss_params_l2 = 0
        total = 0

        print('w_clean: ', opt.weight_clean)
        print('w_kl: ', opt.weight_kl)
        print('weight_mi_giveny_adv', opt.weight_mi_giveny_adv)
        print('weight_params_l2', opt.weight_params_l2)

        for iters, (batch_y0, batch_y1) in enumerate(zip(train_iter_y0, train_iter_y1)):

            text = torch.cat((batch_y0[0].to(device),batch_y1[0].to(device)), 0)
            label = torch.cat((batch_y0[1].to(device),batch_y1[1].to(device)), 0)
            text_like_syn= torch.cat((batch_y0[2].to(device),batch_y1[2].to(device)), 0)
            text_like_syn_valid= torch.cat((batch_y0[3].to(device),batch_y1[3].to(device)), 0)
            bert_mask= torch.cat((batch_y0[4].to(device),batch_y1[4].to(device)), 0)
            bert_token_id= torch.cat((batch_y0[5].to(device),batch_y1[5].to(device)), 0)
            dataset_id= torch.cat((batch_y0[6].to(device),batch_y1[6].to(device)), 0)
            sample_idx = torch.cat((batch_y0[7].to(device),batch_y1[7].to(device)), 0)

            bs, sent_len = text.shape

            model.train()
            # zero grad
            optimizer.zero_grad()

            with torch.no_grad():
                embd = model(mode="text_to_embd", input=text, bert_mask=bert_mask, bert_token_id=bert_token_id) #in bs, len sent, vocab
            n,l,s = text_like_syn.shape

            with torch.no_grad():
                text_like_syn_embd = model(mode="text_to_embd", input=text_like_syn.permute(0, 2, 1).reshape(n*s,l), bert_mask=bert_mask.reshape(n,l,1).repeat(1,1,s).permute(0, 2, 1).reshape(n*s,l), bert_token_id=bert_token_id.reshape(n,l,1).repeat(1,1,s).permute(0, 2, 1).reshape(n*s,l)).reshape(n,s,l,-1).permute(0,2,1,3)

            if opt.ascc_mode=="comb_p":
                attack_type_dict = {
                    'num_steps': opt.train_attack_iters,
                    'loss_func': 'kl',
                    'w_optm_lr': opt.bert_w_optm_lr,
                    'sparse_weight': opt.train_attack_sparse_weight,
                    'out_type': "comb_p"
                }
                adv_comb_p = model(mode="get_adv_by_convex_syn", input=embd, label=label, text_like_syn_embd=text_like_syn_embd, text_like_syn_valid=text_like_syn_valid, attack_type_dict=attack_type_dict, bert_mask=bert_mask, bert_token_id=bert_token_id)
                predicted_adv = model(mode="text_syn_p_to_logit", input=text_like_syn, comb_p=adv_comb_p, bert_mask=bert_mask, bert_token_id=bert_token_id)
            
            elif opt.ascc_mode=="text":
                attack_type_dict = {
                'num_steps': opt.train_attack_iters,
                'loss_func': 'kl',
                'w_optm_lr': opt.bert_w_optm_lr,
                'sparse_weight': opt.train_attack_sparse_weight,
                'out_type': "text"
                }
                adv_text = model(mode="get_adv_by_convex_syn", input=embd, label=label, text_like_syn_embd=text_like_syn_embd, text_like_syn_valid=text_like_syn_valid, attack_type_dict=attack_type_dict, bert_mask=bert_mask, bert_token_id=bert_token_id)
                predicted_adv = model(mode="text_to_logit", input=adv_text, bert_mask=bert_mask, bert_token_id=bert_token_id)

            optimizer.zero_grad()
            # clean loss
            predicted = model(mode="text_to_logit", input=text, bert_mask=bert_mask, bert_token_id=bert_token_id)
            loss_clean= loss_fun(predicted,label)

            # kl loss
            criterion_kl = nn.KLDivLoss(reduction="sum")
            if opt.weight_kl == 0:
                loss_kl = torch.zeros(1).to(device)
            else:
                loss_kl = (1.0 / bs) * criterion_kl(F.log_softmax(predicted_adv, dim=1),
                                                            F.softmax(predicted, dim=1))

            if opt.weight_mi_giveny_adv == 0:
                loss_mi_giveny_adv = torch.zeros(1).to(device)
            else:
                if opt.ascc_mode=="comb_p":
                    loss_mi_giveny_adv = model(mode="text_syn_p_to_infonce_giveny", input=text_like_syn, comb_p=adv_comb_p, bert_mask=bert_mask, bert_token_id=bert_token_id, sim_metric=opt.infonce_sim_metric, orig_text_for_info=text)       
                elif opt.ascc_mode=="text":
                    loss_mi_giveny_adv = model(mode="text_to_infonce_giveny", input=adv_text, bert_mask=bert_mask, bert_token_id=bert_token_id, sim_metric=opt.infonce_sim_metric, orig_text_for_info=text)  
            
            # optimize
            loss =  opt.weight_kl * loss_kl + opt.weight_clean * loss_clean + loss_mi_giveny_adv*opt.weight_mi_giveny_adv 
            loss.backward() 
            optimizer.step()

            sum_loss += loss.item()
            sum_loss_clean += loss_clean.item()
            sum_loss_kl += loss_kl.item()
            sum_loss_mi_giveny_adv += loss_mi_giveny_adv.item()
            predicted, idx = torch.max(predicted, 1) 
            precision=(idx==label).float().mean().item()
            predicted_adv, idx = torch.max(predicted_adv, 1)
            precision_adv=(idx==label).float().mean().item()
            total += 1

            out_log = "%d epoch %d iters: loss: %.3f, loss_kl: %.3f, loss_clean: %.3f, loss_mi_giveny_adv: %.3f | acc: %.3f acc_adv: %.3f | in %.3f seconds" % (epoch, iters, sum_loss/total, sum_loss_kl/total, sum_loss_clean/total, sum_loss_mi_giveny_adv/total, precision, precision_adv, time.time()-start)
            start= time.time()
            print(out_log)
                
        scheduler.step()
       
        current_model_path=os.path.join(opt.out_path, "{}_epoch{}.pth".format(opt.model, epoch))
        state = {
                'net': model.state_dict(),
                'epoch': epoch,
            }
        torch.save(state, current_model_path)

        if opt.if_test and epoch>=10: 
            acc=utils.imdb_evaluation_bert(opt, device, model, test_iter)
            out_log="%d epoch acc %.4f" % (epoch, acc)
            print(out_log)
            print("Best Acc under genetic: ", best_adv_acc)
            print("Running Genetic Attack for this epoch")
            adv_acc=genetic_attack(opt, device, model, attack_surface, dataset=opt.dataset, genetic_test_num=opt.genetic_test_num, test_bert=True)
            out_log="%d epoch genetic acc %.4f" % (epoch, adv_acc)
            print(out_log)
            
            if adv_acc>=best_adv_acc:
                best_adv_acc = adv_acc
                best_save_dir=os.path.join(opt.out_path, "{}_best.pth".format(opt.model))
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state, best_save_dir)

            print("Best Acc under genetic: ", best_adv_acc)

def main():
    opt = opts.parse_opt()
    print(opt)

    torch.manual_seed(opt.torch_seed)

    assert(opt.dataset == "imdb")
    if opt.bert_type=="bert":
        syn_train_iter_y0, syn_train_iter_y1, syn_test_iter = imdb_bert_make_synthesized_iter_giveny(opt)

    train(opt, syn_train_iter_y0, syn_train_iter_y1, syn_test_iter)
    
if __name__=="__main__": 
    main()
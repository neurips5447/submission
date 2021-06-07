# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import BertModel,BertForSequenceClassification
import os
import math

class BertModel_forward_modified(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if input_ids is not None:
            embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )
            return embedding_output

        elif inputs_embeds is not None:
            embedding_output = inputs_embeds
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

            assert(not return_dict)
            return (sequence_output, pooled_output) + encoder_outputs[1:]

class AdvBERTKD(nn.Module): 
    def __init__(self, opt ):
        super(AdvBERTKD, self).__init__()
        self.opt=opt
        
        if opt.bert_type=="bert":
            self.bert_stu = BertModel_forward_modified.from_pretrained('bert-base-uncased')
            self.bert_tea = BertModel_forward_modified.from_pretrained('bert-base-uncased')

        tea_params_l2 = 1e-10
        for (param_name_stu, param_stu), (param_name_tea, param_tea) in zip(self.bert_stu.named_parameters(),self.bert_tea.named_parameters()):
            assert(param_name_stu==param_name_tea)
            tea_params_l2 += (param_tea**2).sum()
        tea_params_l2 = tea_params_l2.sqrt()
        print("----------teacher params l2: ", tea_params_l2, "----------------")

        self.bert_stu = torch.nn.DataParallel(self.bert_stu)
        self.bert_tea = torch.nn.DataParallel(self.bert_tea)
      
        if opt.freeze_bert_stu:
            for name, param in self.bert_stu.named_parameters():
                param.requires_grad=False
        else:
            for name, param in self.bert_stu.named_parameters():
                param.requires_grad=True

        if opt.freeze_bert_tea:
            for name, param in self.bert_tea.named_parameters():
                param.requires_grad=False
        else:
            for name, param in self.bert_tea.named_parameters():
                param.requires_grad=True

        self.t_stu_gy_list = nn.ModuleList([])
        self.t_tea_gy_list = nn.ModuleList([])

        for y in range(self.opt.label_size):
            self.t_stu_gy_list.append(nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Linear(512,256)))
            self.t_tea_gy_list.append(nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Linear(512,256)))

        self.RP_hidden2label_tea = nn.Sequential(nn.Linear(768,256), nn.ReLU(), nn.Linear(256,opt.label_size))
        self.RP_hidden2label_stu = nn.Sequential(nn.Linear(768,256), nn.ReLU(), nn.Linear(256,opt.label_size))

        self.eval_adv_mode = True
        self.temperature = 0.2

    
    def embd_to_logit_tea(self, embd, attention_mask):
        with torch.no_grad():
            _, pooled_tea = self.bert_tea(inputs_embeds=embd, attention_mask=attention_mask)
        logits_tea = self.RP_hidden2label_tea(pooled_tea.detach())
        return logits_tea

    def embd_to_logit_stu(self, embd, attention_mask):
        out, pooled = self.bert_stu(inputs_embeds=embd, attention_mask=attention_mask) 
        fea = pooled
        logits = self.RP_hidden2label_stu(fea)
        return logits


    def embd_to_infonce(self, embd, embd_tea, attention_mask, sim_metric='projected_cossim', y=None):

        bs = (embd.shape)[0]

        _, pooled_stu, hidden_stu = self.bert_stu(inputs_embeds=embd, attention_mask=attention_mask, output_hidden_states=True) 
        
        with torch.no_grad():
            _, pooled_tea, hidden_tea = self.bert_tea(inputs_embeds=embd_tea, attention_mask=attention_mask, output_hidden_states=True)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        if sim_metric=='projected_cossim':

            fea_stu = pooled_stu
            fea_tea = pooled_tea

            fea_stu = self.t_stu_gy_list[y](fea_stu)
            fea_tea = self.t_tea_gy_list[y](fea_tea)

            fea_stu = torch.nn.functional.normalize(fea_stu)
            fea_tea = torch.nn.functional.normalize(fea_tea)

            bs, fs = fea_stu.shape
            f_a_b = torch.mm(fea_stu, torch.transpose(fea_tea, 0, 1) )#bs*768 * 768*bs = bs * bs

            f_a_b = f_a_b/self.temperature

            lsoftmax_0 = nn.LogSoftmax(0)
            loss = - torch.sum(torch.diag(lsoftmax_0(f_a_b)))/bs

        else:
            raise(NotImplementedError)

        return loss
    
    def text_to_embd(self, input_ids, token_type_ids, model=None):
        if model is None:
            model=self.bert_stu

        embedding_output = model(
            input_ids=input_ids, token_type_ids=token_type_ids)
        return embedding_output
    
    def get_adv_by_convex_syn(self, embd, y, syn, syn_valid, text_like_syn, attack_type_dict, bert_mask, text_for_vis, record_for_vis):
        
        # record context
        self_training_context = self.training
        # set context

        self.eval()


        device = embd.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        loss_func=attack_type_dict['loss_func']
        w_optm_lr=attack_type_dict['w_optm_lr']
        sparse_weight = attack_type_dict['sparse_weight']
        out_type = attack_type_dict['out_type']

        batch_size, text_len, embd_dim = embd.shape
        batch_size, text_len, syn_num, embd_dim = syn.shape

        w = torch.empty(batch_size, text_len, syn_num, 1).to(device).to(embd.dtype)
        #ww = torch.zeros(batch_size, text_len, syn_num, 1).to(device).to(embd.dtype)
        #ww = ww+500*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
        nn.init.kaiming_normal_(w)
        w.requires_grad_()
        
        import utils
        params = [w] 
        optimizer = utils.getOptimizer(params,name='adam', lr=w_optm_lr,weight_decay=2e-5)

        def get_comb_p(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, syn_num, 1) + 500*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
            return F.softmax(ww, -2)

        def get_comb_ww(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, syn_num, 1) + 500*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
            return ww

        def get_comb(p, syn):
            return (p* syn.detach()).sum(-2)


        embd_ori=embd.detach()
        with torch.no_grad():
            logit_ori = self.embd_to_logit_stu(embd_ori, bert_mask)

        for _ in range(num_steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                ww = get_comb_ww(w, syn_valid)
                #comb_p = get_comb_p(w, syn_valid)
                embd_adv = get_comb(F.softmax(ww, -2), syn)
                if loss_func=='ce':
                    logit_adv = self.embd_to_logit_stu(embd_adv, bert_mask)
                    loss = -F.cross_entropy(logit_adv, y, reduction='sum')
                elif loss_func=='kl':
                    logit_adv = self.embd_to_logit_stu(embd_adv, bert_mask)
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = -criterion_kl(F.log_softmax(logit_adv, dim=1),
                                        F.softmax(logit_ori.detach(), dim=1))

                #print("ad loss:", loss.data.item())
                ad_loss = loss.detach()
                                    
                if sparse_weight !=0:
                    #loss_sparse = (comb_p*comb_p).mean()
                    loss_sparse = (-F.softmax(ww, -2)*F.log_softmax(ww, -2)).sum(-2).sum() / (300*128)
                    #loss -= sparse_weight*loss_sparse
                    
                    loss = loss + sparse_weight*loss_sparse
                    #print(loss_sparse.data.item())

            #loss*=1000
            loss.backward()
            optimizer.step()

        #print((ww-w).max())

        comb_p = get_comb_p(w, syn_valid)

        if out_type == "text":
            # need to be fix, has potential bugs. the trigger dependes on data.
            assert(text_like_syn is not None) # n l synlen
            comb_p = comb_p.reshape(batch_size* text_len, syn_num)
            ind = comb_p.max(-1)[1] # shape batch_size* text_len
            out = (text_like_syn.reshape(batch_size* text_len, syn_num)[np.arange(batch_size*text_len), ind]).reshape(batch_size, text_len)
        elif out_type == "comb_p":
            out = comb_p
        elif out_type == "loss":
            out = ad_loss

        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        return out.detach()

    def forward(self, mode, input, dataset_id=None, sample_idx=None, comb_p = None, label=None, text_like_syn_embd=None, text_like_syn_valid=None, text_like_syn=None, attack_type_dict=None, bert_mask=None, bert_token_id=None, text_for_vis=None, record_for_vis=None, sim_metric='cos', orig_text_for_info=None):
       
        if mode == "get_embd_adv":
            assert(attack_type_dict is not None)
            out = self.get_embd_adv(input, label, attack_type_dict)

        if mode == "get_adv_by_convex_syn":
            assert(attack_type_dict is not None)
            assert(text_like_syn_embd is not None)
            assert(text_like_syn_valid is not None)
            out = self.get_adv_by_convex_syn(input, label, text_like_syn_embd, text_like_syn_valid, text_like_syn, attack_type_dict, bert_mask, text_for_vis, record_for_vis)
        
        if mode == "embd_to_logit":
            out = self.embd_to_logit_stu(input, bert_mask)
        if mode == "text_to_embd":
            out = self.text_to_embd(input, bert_token_id, self.bert_stu) 
        if mode == "text_to_logit":
            embd = self.text_to_embd(input, bert_token_id, self.bert_stu)
            out = self.embd_to_logit_stu(embd, bert_mask)

        if mode == "embd_to_logit_tea":
            out = self.embd_to_logit_tea(input, bert_mask)

        if mode == "text_to_embd_tea":
            out = self.text_to_embd(input, bert_token_id, self.bert_tea) 

        if mode == "text_to_logit_tea":
            embd = self.text_to_embd(input, bert_token_id, self.bert_tea) 
            out = self.embd_to_logit_tea(embd, bert_mask)

        if mode == "text_syn_p_to_logit":
            assert(comb_p is not None)
            n, l, s = input.shape
            text_like_syn_embd = self.text_to_embd(input.permute(0, 2, 1).reshape(n*s,l), bert_token_id.reshape(n,l,1).repeat(1,1,s).permute(0, 2, 1).reshape(n*s,l)).reshape(n,s,l,-1).permute(0,2,1,3)
            embd = (comb_p*text_like_syn_embd).sum(-2)
            out = self.embd_to_logit_stu(embd, bert_mask)

        if mode == "text_to_infonce_giveny":
            self.bert_stu.train()
            self.bert_tea.eval()

            n, l = input.shape
            bs_per_y = n//self.opt.label_size
            embd = self.text_to_embd(input, bert_token_id)
            embd_tea = self.text_to_embd(orig_text_for_info, bert_token_id, self.bert_tea)
            loss = 0
            for i in range(self.opt.label_size):
                loss += self.embd_to_infonce(embd[i*bs_per_y:(i+1)*bs_per_y], embd_tea[i*bs_per_y:(i+1)*bs_per_y], bert_mask[i*bs_per_y:(i+1)*bs_per_y], sim_metric=sim_metric,  y=i)
            out = loss/self.opt.label_size

        if mode == "text_syn_p_to_infonce_giveny":
            assert(comb_p is not None)
            assert(orig_text_for_info is not None)
            self.bert_tea.eval()
            self.bert_stu.train()
            n, l, s = input.shape
            text_like_syn_embd = self.text_to_embd(input.permute(0, 2, 1).reshape(n*s,l), bert_token_id.reshape(n,l,1).repeat(1,1,s).permute(0, 2, 1).reshape(n*s,l)).reshape(n,s,l,-1).permute(0,2,1,3)
            embd = (comb_p*text_like_syn_embd).sum(-2)
            embd_tea = self.text_to_embd(orig_text_for_info, bert_token_id, self.bert_tea) 

            bs_per_y = n//self.opt.label_size
            loss = 0
            for i in range(self.opt.label_size):
                loss += self.embd_to_infonce(embd[i*bs_per_y:(i+1)*bs_per_y], embd_tea[i*bs_per_y:(i+1)*bs_per_y], bert_mask[i*bs_per_y:(i+1)*bs_per_y], sim_metric=sim_metric, y=i)
            out = loss/self.opt.label_size

        if mode == "params_l2":
            loss = 1e-10
            for (param_name_stu, param_stu), (param_name_tea, param_tea) in zip(self.bert_stu.named_parameters(),self.bert_tea.named_parameters()):
                #assert(param_name_stu==param_name_tea)
                if param_name_stu==param_name_tea:
                    loss += ((param_stu-param_tea)**2).sum()
            out = loss.sqrt()

        return out

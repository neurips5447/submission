# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import argparse
import os
import numpy as np
from .read_files import split_imdb_files, split_yahoo_files, split_agnews_files, split_snli_files
from .word_level_process import word_process, get_tokenizer
from .char_level_process import char_process
from .adversarial_tools import ForwardGradWrapper, ForwardGradWrapper_pytorch, adversarial_paraphrase, ForwardGradWrapper_pytorch_snli, adversarial_paraphrase_snli
import time
import random
from .unbuffered import Unbuffered
import torch
try:
    import cPickle as pickle
except ImportError:
    import pickle

sys.stdout = Unbuffered(sys.stdout)

def write_origin_input_texts(origin_input_texts_path, test_texts, test_samples_cap=None):
    if test_samples_cap is None:
        test_samples_cap = len(test_texts)
    with open(origin_input_texts_path, 'a') as f:
        for i in range(test_samples_cap):
            f.write(test_texts[i] + '\n')

def genetic_attack(opt, device, model, attack_surface, dataset='imdb', genetic_test_num=100, test_bert=False):

    if test_bert:
        if opt.bert_type=="bert":
            from modified_bert_tokenizer import ModifiedBertTokenizer
            tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        elif opt.bert_type=="roberta":
            from modified_bert_tokenizer  import ModifiedRobertaTokenizer
            tokenizer = ModifiedRobertaTokenizer.from_pretrained("roberta-base",add_prefix_space=True)
        elif opt.bert_type=="xlnet":
            from modified_bert_tokenizer  import ModifiedXLNetTokenizer
            tokenizer = ModifiedXLNetTokenizer.from_pretrained("xlnet-base-cased", do_lower_case=True)
    else:
        # get tokenizer
        tokenizer = get_tokenizer(opt)

    if dataset == 'imdb':
        train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
        #x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    indexes = [i for i in range(len(test_texts))]
    random.seed(opt.rand_seed)
    random.shuffle(indexes)
    test_texts = [test_texts[i] for i in indexes]
    test_labels = [test_labels[i] for i in indexes]
    indexes = []
    for i, x in enumerate(test_texts):
        words = x.split()
        if attack_surface.check_in(words):
            indexes.append(i)
    test_texts = [test_texts[i] for i in indexes]
    test_labels = [test_labels[i] for i in indexes]

    #print("all_genetic_in_lm", len(test_texts))
    test_num = min(len(test_texts), genetic_test_num)

    test_texts = test_texts[:test_num]
    test_labels = test_labels[:test_num]

    from .attacks import GeneticAdversary, AdversarialModel
    adversary = GeneticAdversary(opt, attack_surface, num_iters=opt.genetic_iters, pop_size=opt.genetic_pop_size)
    from .config import config
    wrapped_model = AdversarialModel(tokenizer, config.word_max_len[dataset], test_bert=test_bert, bert_type=opt.bert_type)

    from multiprocessing import Process, Pipe
    conn_main = []
    conn_p = []
    for i in range(test_num):
        c1, c2 = Pipe()
        conn_main.append(c1)
        conn_p.append(c2)

    process_list = []
    for i in range(test_num):
        p = Process(target=adversary.run, args=(conn_p[i], wrapped_model, test_texts[i], test_labels[i], 'cpu'))
        p.start()
        process_list.append(p)

    tested = 0
    acc = 0
    bs = 100
    model.eval()
    p_state=[1 for i in range(test_num)]
    t_start = time.time()


    test_text = torch.zeros(bs, config.word_max_len[dataset],dtype=torch.long).to(device)
    test_mask = torch.zeros(bs, config.word_max_len[dataset],dtype=torch.long).to(device)
    test_token_type_ids = torch.zeros(bs, config.word_max_len[dataset],dtype=torch.long).to(device)

    i=0
    while(1):
        if tested == test_num:
            break
        #for i in range(genetic_test_num):
        bs_j=0
        res_dict = {}
        while(1):
            i=(i+1)%test_num
            if p_state[i] == 1:
                cm=conn_main[i]
                if cm.poll():
                    msg = cm.recv()
                    if msg == 0 or msg == 1:
                        tested+=1
                        acc+=msg
                        cm.close()
                        p_state[i]=0
                        process_list[i].join()
                    else:
                        text, mask, token_type_ids = msg
                        test_text[bs_j] = text.to(device)
                        test_mask[bs_j] = mask.to(device)
                        test_token_type_ids[bs_j] = token_type_ids.to(device)
                        res_dict[i]=bs_j

                        bs_j +=1

                    if bs_j==bs or bs_j>=(test_num-tested):
                        break

        with torch.no_grad():
            logits = model(mode="text_to_logit", input=test_text, bert_mask=test_mask, bert_token_id=test_token_type_ids).detach().cpu()
                
        for key in res_dict.keys():
            cm=conn_main[key]
            cm.send(logits[res_dict[key]])

        bs_j = 0
    #print("final genetic", acc/tested)
    print("genetic time:", time.time()-t_start)
    return acc/tested


def genetic_attack_snli(opt, device, model, attack_surface, dataset='snli', genetic_test_num=100, split="test", test_bert=False):
    
    if test_bert:
        if opt.bert_type=="bert":
            from modified_bert_tokenizer import ModifiedBertTokenizer
            tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        elif opt.bert_type=="roberta":
            from modified_bert_tokenizer  import ModifiedRobertaTokenizer
            tokenizer = ModifiedRobertaTokenizer.from_pretrained("roberta-base",add_prefix_space=True)
    else:
        # get tokenizer
        tokenizer = get_tokenizer(opt)

    # Read data set

    if dataset == 'snli':
        train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)
    else:
        raise NotImplementedError

    indexes = [i for i in range(len(test_labels))]
    random.seed(opt.rand_seed)
    random.shuffle(indexes)
    test_perms = [test_perms[i] for i in indexes]
    test_hypos = [test_hypos[i] for i in indexes]
    test_labels = [test_labels[i] for i in indexes]

    from .attacks import GeneticAdversary_Snli, AdversarialModel_Snli
    adversary = GeneticAdversary_Snli(attack_surface, num_iters=opt.genetic_iters, pop_size=opt.genetic_pop_size)

    from .config import config
    wrapped_model = AdversarialModel_Snli(model, tokenizer, config.word_max_len[dataset], test_bert=test_bert, bert_type=opt.bert_type)

    adv_acc = adversary.run(wrapped_model, test_perms, test_hypos, test_labels, device, genetic_test_num, opt)
    print("genetic attack results:", adv_acc)

    return adv_acc


def fool_text_classifier_pytorch(opt, device,model, dataset='imdb', clean_samples_cap=50, test_bert=False):
    print('clean_samples_cap:', clean_samples_cap)

    if test_bert:
        if opt.bert_type=="bert":
            from modified_bert_tokenizer import ModifiedBertTokenizer
            tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        elif opt.bert_type=="roberta":
            from modified_bert_tokenizer  import ModifiedRobertaTokenizer
            tokenizer = ModifiedRobertaTokenizer.from_pretrained("roberta-base",add_prefix_space=True)
        elif opt.bert_type=="xlnet":
            from modified_bert_tokenizer  import ModifiedXLNetTokenizer
            tokenizer = ModifiedXLNetTokenizer.from_pretrained("xlnet-base-cased", do_lower_case=True)
    else:
        # get tokenizer
        tokenizer = get_tokenizer(opt)

    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)

    # if opt.synonyms_from_file:

    #     if dataset == 'imdb':
    #         train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
    #     elif dataset == 'agnews':
    #         train_texts, train_labels, test_texts, test_labels = split_agnews_files()
    #     elif dataset == 'yahoo':
    #         train_texts, train_labels, test_texts, test_labels = split_yahoo_files()

    #     filename= opt.imdb_synonyms_file_path
    #     f=open(filename,'rb')
    #     saved=pickle.load(f)
    #     f.close()
    #     #syn_data = saved["syn_data"]
    #     #opt.embeddings=saved['embeddings']
    #     #opt.vocab_size=saved['vocab_size']
    #     x_train=saved['x_train']
    #     x_test=saved['x_test']
    #     y_train=saved['y_train']
    #     y_test=saved['y_test']

    # else:
    #     if dataset == 'imdb':
    #         train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = split_imdb_files(opt)
    #         x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    #     elif dataset == 'agnews':
    #         train_texts, train_labels, test_texts, test_labels = split_agnews_files()
    #         x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    #     elif dataset == 'yahoo':
    #         train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
    #         x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    #shuffle test_texts
    indexes = [i for i in range(len(test_texts))]
    random.seed(opt.rand_seed)
    random.shuffle(indexes)
    test_texts = [test_texts[i] for i in indexes]
    test_labels = [test_labels[i] for i in indexes]

    from .config import config
    grad_guide = ForwardGradWrapper_pytorch(dataset, config.word_max_len[dataset],  model, device, tokenizer, test_bert, opt.bert_type)
    classes_prediction = [grad_guide.predict_classes(text) for text in test_texts[: clean_samples_cap]]

    print(sum([classes_prediction[i]==np.argmax(test_labels[i]) for i in range(clean_samples_cap)])/clean_samples_cap)

    print('Crafting adversarial examples...')
    successful_perturbations = 0
    failed_perturbations = 0
    all_test_num =0

    sub_rate_list = []
    NE_rate_list = []

    fa_path = r'./fool_result/{}'.format(dataset)
    if not os.path.exists(fa_path):
        os.makedirs(fa_path)
    adv_text_path = r'./fool_result/{}/adv_{}.txt'.format(dataset, str(clean_samples_cap))
    change_tuple_path = r'./fool_result/{}/change_tuple_{}.txt'.format(dataset, str(clean_samples_cap))
    #file_1 = open(adv_text_path, "a")
    #file_2 = open(change_tuple_path, "a")

    for index, text in enumerate(test_texts[opt.h_test_start: opt.h_test_start+clean_samples_cap]):
        sub_rate = 0
        NE_rate = 0
        all_test_num+=1
        print('_____{}______.'.format(index))

        if np.argmax(test_labels[index]) == classes_prediction[index]:
            print('do')
            start_cpu = time.clock()
            # If the ground_true label is the same as the predicted label
            adv_doc, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase(opt,
                                                                                          input_text=text,
                                                                                          true_y=np.argmax(test_labels[index]),
                                                                                          grad_guide=grad_guide,
                                                                                          tokenizer=tokenizer,
                                                                                          dataset=dataset,
                                                                                          level='word')
            if adv_y != np.argmax(test_labels[index]):
                successful_perturbations += 1
                print('{}. Successful example crafted.'.format(index))
            else:
                failed_perturbations += 1
                print('{}. Failure.'.format(index))

            print("r acc", 1.0*failed_perturbations/all_test_num)

            text = adv_doc
            sub_rate_list.append(sub_rate)
            NE_rate_list.append(NE_rate)

            end_cpu = time.clock()
            print('CPU second:', end_cpu - start_cpu)

    #mean_sub_rate = sum(sub_rate_list) / len(sub_rate_list)
    #mean_NE_rate = sum(NE_rate_list) / len(NE_rate_list)
    print('substitution:', sum(sub_rate_list))
    print('sum substitution:', len(sub_rate_list))
    print('NE rate:', sum(NE_rate_list))
    print('sum NE:', len(NE_rate_list))
    print("succ attack %d"%(successful_perturbations))
    print("fail attack %d"%(failed_perturbations))
    #file_1.close()
    #file_2.close()




def fool_text_classifier_pytorch_snli(opt, device,model, dataset='snli', clean_samples_cap=50, test_bert=False):
    print('clean_samples_cap:', clean_samples_cap)

    if test_bert:
        if opt.bert_type=="bert":
            from modified_bert_tokenizer import ModifiedBertTokenizer
            tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        elif opt.bert_type=="roberta":
            from modified_bert_tokenizer  import ModifiedRobertaTokenizer
            tokenizer = ModifiedRobertaTokenizer.from_pretrained("roberta-base",add_prefix_space=True)
    else:
        # get tokenizer
        tokenizer = get_tokenizer(opt)

    if dataset == 'snli':
        train_perms, train_hypos, train_labels, dev_perms, dev_hypos, dev_labels, test_perms, test_hypos, test_labels = split_snli_files(opt)
    else:
        raise NotImplementedError

    #shuffle test_texts
    indexes = [i for i in range(len(test_perms))]
    random.seed(opt.rand_seed)
    random.shuffle(indexes)
    test_perms = [test_perms[i] for i in indexes]
    test_hypos = [test_hypos[i] for i in indexes]
    test_labels = [test_labels[i] for i in indexes]
    
    #indexes = []
    #for i, x in enumerate(test_hypos):
    #    words = x.split()
    #    if attack_surface.check_in(words):
    #        indexes.append(i)
    #test_texts = [test_texts[i] for i in indexes]
    #test_labels = [test_labels[i] for i in indexes]

    from .config import config
    grad_guide = ForwardGradWrapper_pytorch_snli(dataset, config.word_max_len[dataset],  model, device, tokenizer, test_bert, opt.bert_type)
    classes_prediction = [grad_guide.predict_classes(test_perm, test_hypo) for test_perm, test_hypo in zip(test_perms[: clean_samples_cap], test_hypos[: clean_samples_cap])]

    print(sum([classes_prediction[i]==np.argmax(test_labels[i]) for i in range(clean_samples_cap)])/clean_samples_cap)

    print('Crafting adversarial examples...')
    successful_perturbations = 0
    failed_perturbations = 0
    all_test_num =0

    sub_rate_list = []
    NE_rate_list = []

    for index in range(opt.h_test_start, opt.h_test_start+clean_samples_cap, 1):

        text_p=test_perms[index]
        text_h=test_hypos[index]

        sub_rate = 0
        NE_rate = 0
        all_test_num+=1
        print('_____{}______.'.format(index))
        if np.argmax(test_labels[index]) == classes_prediction[index]:
            print('do')
            start_cpu = time.clock()
            # If the ground_true label is the same as the predicted label
            adv_doc_p, adv_doc_h, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase_snli(opt, input_text_p=text_p, input_text_h=text_h,
                                                                                          true_y=np.argmax(test_labels[index]),
                                                                                          grad_guide=grad_guide,
                                                                                          tokenizer=tokenizer,
                                                                                          dataset=dataset,
                                                                                          level='word')
            if adv_y != np.argmax(test_labels[index]):
                successful_perturbations += 1
                print('{}. Successful example crafted.'.format(index))
            else:
                failed_perturbations += 1
                print('{}. Failure.'.format(index))

            print("r acc", 1.0*failed_perturbations/all_test_num)
            sub_rate_list.append(sub_rate)
            NE_rate_list.append(NE_rate)

            end_cpu = time.clock()
            print('CPU second:', end_cpu - start_cpu)

    print("PWWS acc:", 1.0*failed_perturbations/all_test_num)

    #print('substitution:', sum(sub_rate_list))
    #print('sum substitution:', len(sub_rate_list))
    #print('NE rate:', sum(NE_rate_list))
    #print('sum NE:', len(NE_rate_list))
    #print("succ attack %d"%(successful_perturbations))
    #print("fail attack %d"%(failed_perturbations))


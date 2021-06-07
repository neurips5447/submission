# coding: utf-8
from .config import config
import copy
import spacy
from .word_level_process import text_to_vector
from .char_level_process import doc_process, get_embedding_dict

nlp = spacy.load('en_core_web_sm')


def evaluate_word_saliency(input_text, grad_guide, tokenizer, input_y, dataset, level):
    word_saliency_list = []
    input_text_splited = input_text.split(" ")
    # zero the code of the current word and calculate the amount of change in the classification probability
    if level == 'word':
        max_len = config.word_max_len[dataset]
        #origin_vector = text_to_vector(text, tokenizer, dataset)
        origin_prob = grad_guide.predict_prob(input_text)
        for position in range(len(input_text_splited)):
            if position >= max_len:
                break
            # get x_i^(\hat)
            #without_word_vector = copy.deepcopy(origin_vector)
            #without_word_vector[0][position] = 0
            #without_word_text = [doc[position].text for position in range(len(doc))]
            temp = input_text_splited[position]
            input_text_splited[position] = "NULL"
            without_word_text = ' '.join(input_text_splited)
            input_text_splited[position] = temp

            prob_without_word = grad_guide.predict_prob(without_word_text)

            # calculate S(x,w_i) defined in Eq.(6)
            word_saliency = origin_prob[input_y] - prob_without_word[input_y]
            word_saliency_list.append((position, input_text_splited[position], word_saliency, None))

    return word_saliency_list


def evaluate_word_saliency_snli(input_text_p, input_text_h, grad_guide, tokenizer, input_y, dataset, level):
    word_saliency_list = []
    input_text_h_splited = input_text_h.split(" ")
    # zero the code of the current word and calculate the amount of change in the classification probability
    if level == 'word':
        max_len = config.word_max_len[dataset]
        #origin_vector = text_to_vector(text, tokenizer, dataset)
        origin_prob = grad_guide.predict_prob(input_text_p, input_text_h)
        for position in range(len(input_text_h_splited)):
            if position >= max_len:
                break
            # get x_i^(\hat)
            #without_word_vector = copy.deepcopy(origin_vector)
            #without_word_vector[0][position] = 0
            #without_word_text = [doc[position].text for position in range(len(doc))]
            temp = input_text_h_splited[position]
            input_text_h_splited[position] = "NULL"
            without_word_text_h = ' '.join(input_text_h_splited)
            input_text_h_splited[position] = temp

            prob_without_word = grad_guide.predict_prob(input_text_p, without_word_text_h)

            # calculate S(x,w_i) defined in Eq.(6)
            word_saliency = origin_prob[input_y] - prob_without_word[input_y]
            word_saliency_list.append((position, input_text_h_splited[position], word_saliency, None))

    return word_saliency_list
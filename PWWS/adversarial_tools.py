import sys
import keras
import spacy
import numpy as np
import tensorflow as tf
import os
from .config import config
from keras import backend as K
from .paraphrase import _compile_perturbed_tokens, PWWS, PWWS_snli
from .word_level_process import text_to_vector
from .char_level_process import doc_process, get_embedding_dict
from .evaluate_word_saliency import evaluate_word_saliency, evaluate_word_saliency_snli
#from keras.backend.tensorflow_backend import set_session
from .unbuffered import Unbuffered
from keras.preprocessing import sequence
import torch.nn.functional as F
import torch

sys.stdout = Unbuffered(sys.stdout)
nlp = spacy.load('en', tagger=False, entity=False)


class ForwardGradWrapper:
    '''
    Utility class that computes the classification probability of model input and predict its class
    '''

    def __init__(self, model):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''
        input_tensor = model.input

        self.model = model
        self.input_tensor = input_tensor
        self.sess = K.get_session()

    def predict_prob(self, input_vector):
        prob = self.model.predict(input_vector).squeeze()
        return prob

    def predict_classes(self, input_vector):
        prediction = self.model.predict(input_vector)
        classes = np.argmax(prediction, axis=1)
        return classes


class ForwardGradWrapper_pytorch_snli:
    '''
    Utility class that computes the classification probability of model input and predict its class
    '''

    def __init__(self, dataset, maxlen, model, device, tokenizer, test_bert, bert_type):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''
        model.eval()
        self.dataset = dataset
        self.model=model
        self.device=device
        self.tokenizer = tokenizer
        self.test_bert = test_bert
        self.maxlen = maxlen
        self.bert_type=bert_type

    def get_mask(self, tensor):
        #mask = 1- (tensor==0)
        mask = ~(tensor==0)
        mask=mask.to(self.device).to(torch.float)
        return mask

    def text_to_vector(self, text):
        vector = tokenizer.texts_to_sequences([text])
        vector = sequence.pad_sequences(vector, maxlen=self.maxlen, padding='post', truncating='post')
        return vector

    def predict_prob(self, text_p, text_h):
        if self.test_bert:
            if self.bert_type=="bert":
                token = self.tokenizer.encode_plus(text_p+" [SEP] "+text_h, None, add_special_tokens=True, max_length=self.maxlen, pad_to_max_length=True)
            elif self.bert_type=="roberta":
                token = self.tokenizer(text_p+"</s>"+text_h, None, max_length=self.maxlen, pad_to_max_length=True)
                token["token_type_ids"] = [0 for i in range(len(token["attention_mask"]))]
                
            text = np.array([token['input_ids']])
            text = torch.tensor(text,dtype=torch.long).to(self.device)
            mask = np.array([token['attention_mask']])
            #print(mask.sum())
            mask = torch.tensor(mask,dtype=torch.long).to(self.device)
            token_type_ids = np.array([token["token_type_ids"]])
            token_type_ids = torch.tensor(token_type_ids,dtype=torch.long).to(self.device)
            logit = self.model(mode="text_to_logit", input=text, bert_mask=mask, bert_token_id=token_type_ids).squeeze(0)
        else:
            input_vector_p = self.text_to_vector(text_p)
            input_vector_h = self.text_to_vector(text_h)
            input_vector_p=torch.from_numpy(input_vector_p).to(self.device).to(torch.long)
            input_vector_h=torch.from_numpy(input_vector_h).to(self.device).to(torch.long)
            mask_p = self.get_mask(input_vector_p)
            mask_h = self.get_mask(input_vector_h)
            
            logit = self.model(mode="text_to_logit",x_p=input_vector_p, x_h=input_vector_h, x_p_mask=mask_p, x_h_mask=mask_h).squeeze(0)

        return F.softmax(logit).detach().cpu().numpy()

    def predict_classes(self, text_p, text_h):
        prediction = self.predict_prob(text_p, text_h)
        classes = np.argmax(prediction, axis=-1)
        return classes

class ForwardGradWrapper_pytorch:
    '''
    Utility class that computes the classification probability of model input and predict its class
    '''

    def __init__(self, dataset, maxlen, model, device, tokenizer, test_bert, bert_type):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''
        model.eval()
        self.dataset = dataset
        self.model=model
        self.device=device
        self.tokenizer = tokenizer
        self.test_bert = test_bert
        self.maxlen = maxlen
        self.bert_type=bert_type

    def text_to_vector(self, text):
        vector = tokenizer.texts_to_sequences([text])
        vector = sequence.pad_sequences(vector, maxlen=self.maxlen, padding='post', truncating='post')
        return vector

    def predict_prob(self, text):
        assert(type(text) == str)
        if self.test_bert:
            if self.bert_type=="bert":
                token = self.tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=self.maxlen, pad_to_max_length=True)
            elif self.bert_type=="roberta":
                token = self.tokenizer(text, None, max_length=self.maxlen, pad_to_max_length=True)
                token["token_type_ids"] = [0 for i in range(len(token["attention_mask"]))]
            elif self.bert_type=="xlnet":
                token = self.tokenizer(text, None, max_length=self.maxlen, pad_to_max_length=True)

            #token = self.tokenizer.encode_plus(text, None, add_special_tokens=True, max_length=self.maxlen, pad_to_max_length=True)

            text = np.array([token['input_ids']])
            text = torch.tensor(text,dtype=torch.long).to(self.device)
            mask = np.array([token['attention_mask']])
            #print(mask.sum())
            mask = torch.tensor(mask,dtype=torch.long).to(self.device)
            token_type_ids = np.array([token["token_type_ids"]])
            token_type_ids = torch.tensor(token_type_ids,dtype=torch.long).to(self.device)
            logit = self.model(mode="text_to_logit", input=text, bert_mask=mask, bert_token_id=token_type_ids).squeeze(0)
        else:
            input_vector = self.text_to_vector(text)
            input_vector=torch.from_numpy(input_vector).to(self.device).to(torch.long)
            logit = self.model(mode="text_to_logit",input=input_vector).squeeze(0)

        return F.softmax(logit).detach().cpu().numpy()

    def predict_classes(self, text):
        prediction = self.predict_prob(text)
        classes = np.argmax(prediction, axis=-1)
        return classes


def adversarial_paraphrase(opt, input_text, true_y, grad_guide, tokenizer, dataset, level, verbose=True):
    '''
    Compute a perturbation, greedily choosing the synonym if it causes the most
    significant change in the classification probability after replacement
    :return perturbed_text: generated adversarial examples
    :return perturbed_y: predicted class of perturbed_text
    :return sub_rate: word replacement rate showed in Table 3
    :return change_tuple_list: list of substitute words
    '''

    def halt_condition_fn(perturbed_text):
        '''
        Halt if model output is changed.
        '''
        #perturbed_vector = None
        #if level == 'word':
        #    perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)

        adv_y = grad_guide.predict_classes(perturbed_text)
        if adv_y != true_y:
            return True
        else:
            return False

    def heuristic_fn(text, candidate):
        '''
        Return the difference between the classification probability of the original
        word and the candidate substitute synonym, which is defined in Eq.(4) and Eq.(5).
        '''
        if level == 'word':
            candidate.candidate_word
            text_splited = text.split()
            perturbed_text_splited = text_splited
            perturbed_text_splited[candidate.token_position]=candidate.candidate_word
            perturbed_text = " ".join(perturbed_text_splited) 

        origin_prob = grad_guide.predict_prob(text)
        perturbed_prob = grad_guide.predict_prob(perturbed_text)
        delta_p = origin_prob[true_y] - perturbed_prob[true_y]

        return delta_p

    #doc = nlp(input_text)

    # PWWS
    word_saliency_list = evaluate_word_saliency(input_text, grad_guide, tokenizer, true_y, dataset, level)
    perturbed_text, sub_rate, NE_rate, change_tuple_list = PWWS(opt,
                                                                input_text,
                                                                true_y,
                                                                dataset,
                                                                word_saliency_list=word_saliency_list,
                                                                heuristic_fn=heuristic_fn,
                                                                halt_condition_fn=halt_condition_fn,
                                                                verbose=verbose)

    # print("perturbed_text after perturb_text:", perturbed_text)
    origin_vector = perturbed_vector = None
    #if level == 'word':
        #origin_vector = text_to_vector(input_text, tokenizer, dataset)
        #perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)

    perturbed_y = grad_guide.predict_classes(perturbed_text)
    if verbose:
        origin_prob = grad_guide.predict_prob(input_text)
        perturbed_prob = grad_guide.predict_prob(perturbed_text)
        raw_score = origin_prob[true_y] - perturbed_prob[true_y]
        print('Prob before: ', origin_prob[true_y], '. Prob after: ', perturbed_prob[true_y],
              '. Prob shift: ', raw_score)
    return perturbed_text, perturbed_y, sub_rate, NE_rate, change_tuple_list



def adversarial_paraphrase_snli(opt, input_text_p, input_text_h, true_y, grad_guide, tokenizer, dataset, level, verbose=True):
    '''
    Compute a perturbation, greedily choosing the synonym if it causes the most
    significant change in the classification probability after replacement
    :return perturbed_text: generated adversarial examples
    :return perturbed_y: predicted class of perturbed_text
    :return sub_rate: word replacement rate showed in Table 3
    :return change_tuple_list: list of substitute words
    '''

    def halt_condition_fn(input_text_p, perturbed_text_h):
        '''
        Halt if model output is changed.
        '''
        adv_y = grad_guide.predict_classes(input_text_p, perturbed_text_h)
        if adv_y != true_y:
            return True
        else:
            return False

    def heuristic_fn(text_p, text_h, candidate_h):
        '''
        Return the difference between the classification probability of the original
        word and the candidate substitute synonym, which is defined in Eq.(4) and Eq.(5).
        '''
        if level == 'word':
            candidate_h.candidate_word
            text_h_splited = text_h.split()
            perturbed_text_h_splited = text_h_splited
            perturbed_text_h_splited[candidate_h.token_position]=candidate_h.candidate_word
            perturbed_text_h = " ".join(perturbed_text_h_splited) 

        origin_prob = grad_guide.predict_prob(text_p, text_h)
        perturbed_prob = grad_guide.predict_prob(text_p, perturbed_text_h)
        delta_p = origin_prob[true_y] - perturbed_prob[true_y]

        return delta_p

    # PWWS
    word_saliency_list_h = evaluate_word_saliency_snli(input_text_p, input_text_h, grad_guide, tokenizer, true_y, dataset, level)
    text_p, perturbed_text_h, sub_rate, NE_rate, change_tuple_list = PWWS_snli(opt, input_text_p, input_text_h,
                                                                true_y,
                                                                dataset,
                                                                word_saliency_list_h=word_saliency_list_h,
                                                                heuristic_fn=heuristic_fn,
                                                                #halt_condition_fn=halt_condition_fn,
                                                                halt_condition_fn=halt_condition_fn,
                                                                verbose=verbose)

    origin_vector = perturbed_vector = None
    perturbed_y = grad_guide.predict_classes(text_p, perturbed_text_h)
    if verbose:
        origin_prob = grad_guide.predict_prob(text_p, input_text_h)
        perturbed_prob = grad_guide.predict_prob(text_p, perturbed_text_h)
        raw_score = origin_prob[true_y] - perturbed_prob[true_y]
        print('Prob before: ', origin_prob[true_y], '. Prob after: ', perturbed_prob[true_y],
              '. Prob shift: ', raw_score)
    return text_p, perturbed_text_h, perturbed_y, sub_rate, NE_rate, change_tuple_list

# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import attr
from .config import config
import nltk
import spacy
from functools import partial
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from .get_NE_list import NE_list
import numpy as np
from .unbuffered import Unbuffered

sys.stdout = Unbuffered(sys.stdout)
# from pywsd.lesk import simple_lesk as disambiguate

nlp = spacy.load('en_core_web_sm')
# Penn TreeBank POS tags:
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

supported_pos_tags = [
    'CC',  # coordinating conjunction, like "and but neither versus whether yet so"
    # 'CD',   # Cardinal number, like "mid-1890 34 forty-two million dozen"
    # 'DT',   # Determiner, like all "an both those"
    # 'EX',   # Existential there, like "there"
    # 'FW',   # Foreign word
    # 'IN',   # Preposition or subordinating conjunction, like "among below into"
    'JJ',  # Adjective, like "second ill-mannered"
    'JJR',  # Adjective, comparative, like "colder"
    'JJS',  # Adjective, superlative, like "cheapest"
    # 'LS',   # List item marker, like "A B C D"
    # 'MD',   # Modal, like "can must shouldn't"
    'NN',  # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS',  # Proper noun, plural
    # 'PDT',  # Predeterminer, like "all both many"
    # 'POS',  # Possessive ending, like "'s"
    # 'PRP',  # Personal pronoun, like "hers herself ours they theirs"
    # 'PRP$',  # Possessive pronoun, like "hers his mine ours"
    'RB',  # Adverb
    'RBR',  # Adverb, comparative, like "lower heavier"
    'RBS',  # Adverb, superlative, like "best biggest"
    # 'RP',   # Particle, like "board about across around"
    # 'SYM',  # Symbol
    # 'TO',   # to
    # 'UH',   # Interjection, like "wow goody"
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
    # 'WDT',  # Wh-determiner, like "that what whatever which whichever"
    # 'WP',   # Wh-pronoun, like "that who"
    # 'WP$',  # Possessive wh-pronoun, like "whose"
    # 'WRB',  # Wh-adverb, like "however wherever whenever"
]


@attr.s
class SubstitutionCandidate:
    token_position = attr.ib()
    similarity_rank = attr.ib()
    original_token = attr.ib()
    candidate_word = attr.ib()


def vsm_similarity(doc, original, synonym):
    window_size = 3
    start = max(0, original.i - window_size)
    return doc[start: original.i + window_size].similarity(synonym)


def _get_wordnet_pos(spacy_token):
    '''Wordnet POS tag'''
    pos = spacy_token.tag_[0].lower()
    if pos in ['r', 'n', 'v']:  # adv, noun, verb
        return pos
    elif pos == 'j':
        return 'a'  # adj


def _synonym_prefilter_fn(token, synonym):
    '''
    Similarity heuristics go here
    '''
    if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
            synonym.lemma == token.lemma) or (  # token and synonym are the same
            synonym.tag != token.tag) or (  # the pos of the token synonyms are different
            token.text.lower() == 'be')):  # token is be
        return False
    else:
        return True



def _generate_synonym_candidates(token, token_position, rank_fn=None):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''
    if rank_fn is None:
        rank_fn = vsm_similarity
    candidates = []
    if token.tag_ in supported_pos_tags:
        wordnet_pos = _get_wordnet_pos(token)  # 'r', 'a', 'n', 'v' or None
        wordnet_synonyms = []

        synsets = wn.synsets(token.text, pos=wordnet_pos)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
            synonyms.append(spacy_synonym)

        synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)

        candidate_set = set()
        for _, synonym in enumerate(synonyms):
            candidate_word = synonym.text
            if candidate_word in candidate_set:  # avoid repetition
                continue
            candidate_set.add(candidate_word)
            candidate = SubstitutionCandidate(
                token_position=token_position,
                similarity_rank=None,
                original_token=token,
                candidate_word=candidate_word)
            candidates.append(candidate)
    return candidates



def get_syn_dict(opt):
    import json
    with open(opt.certified_neighbors_file_path) as f:
        syn_dict = json.load(f)
    return syn_dict


def _generate_synonym_candidates_from_dict(opt, word, token_position, syn_dict, rank_fn=None):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''
    if rank_fn is None:
        rank_fn = vsm_similarity

    candidates = []

    """
    candidate = SubstitutionCandidate(
        token_position=token_position,
        similarity_rank=None,
        original_token=token,
        candidate_word=word)
    candidates.append(candidate)
    """

    if not word in syn_dict:
        return candidates
    else:
        for candidate_word in syn_dict[word]:
            candidate = SubstitutionCandidate(
                token_position=token_position,
                similarity_rank=None,
                original_token=word,
                candidate_word=candidate_word)
            candidates.append(candidate)
        return candidates

def generate_synonym_list_from_word(word):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''

    token = nlp(word)[0]

    candidate_set = set()

    #if token.tag_ in supported_pos_tags:
    if True:
        wordnet_pos = _get_wordnet_pos(token)  # 'r', 'a', 'n', 'v' or None
        wordnet_synonyms = []

        synsets = wn.synsets(token.text)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
            synonyms.append(spacy_synonym)

        #synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)

        for _, synonym in enumerate(synonyms):
            candidate_word = synonym.text
            if candidate_word in candidate_set:  # avoid repetition
                continue
            candidate_set.add(candidate_word)

    return list(candidate_set)



def generate_synonym_list_by_dict(syn_dict, word):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''

    if not word in syn_dict:
        return [word]
    else:
        return syn_dict[word]
    

def _compile_perturbed_tokens(doc, accepted_candidates):
    '''
    Traverse the list of accepted candidates and do the token substitutions.
    '''
    candidate_by_position = {}
    for candidate in accepted_candidates:
        candidate_by_position[candidate.token_position] = candidate

    final_tokens = []
    for position, token in enumerate(doc):
        word = token.text
        if position in candidate_by_position:
            candidate = candidate_by_position[position]
            word = candidate.candidate_word
            #word = candidate.candidate_word.replace('_', ' ')
        final_tokens.append(word)

    return final_tokens


def PWWS(
        opt,
        input_text,
        true_y,
        dataset,
        word_saliency_list=None,
        rank_fn=None,
        heuristic_fn=None,  # Defined in adversarial_tools.py
        halt_condition_fn=None,  # Defined in adversarial_tools.py
        verbose=True):

    # defined in Eq.(8)
    def softmax(x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    heuristic_fn = heuristic_fn or (lambda _, candidate: candidate.similarity_rank)
    halt_condition_fn = halt_condition_fn or (lambda perturbed_text: False)

    substitute_count = 0  # calculate how many substitutions used in a doc
    substitute_tuple_list = []  # save the information of substitute word

    word_saliency_array = np.array([word_tuple[2] for word_tuple in word_saliency_list])
    word_saliency_array = softmax(word_saliency_array)

    NE_candidates = NE_list.L[dataset][true_y]

    NE_tags = list(NE_candidates.keys())
    use_NE = False  # whether use NE as a substitute #Changed by dxs

    max_len = config.word_max_len[dataset]

    import json
    with open(opt.certified_neighbors_file_path) as f:
        syn_dict = json.load(f)

    # for each word w_i in x, use WordNet to build a synonym set L_i
    for (position, word, word_saliency, tag) in word_saliency_list:
        if position >= max_len:
            break

        candidates = []
        candidates = _generate_synonym_candidates_from_dict(opt, word, position, syn_dict, rank_fn=rank_fn)

        if len(candidates) == 0:
            continue

        # The substitute word selection method R(w_i;L_i) defined in Eq.(4)
        sorted_candidates = zip(map(partial(heuristic_fn, input_text), candidates), candidates)
        # Sorted according to the return value of heuristic_fn function, that is, \Delta P defined in Eq.(4)
        sorted_candidates = list(sorted(sorted_candidates, key=lambda t: t[0]))

        # delta_p_star is defined in Eq.(5); substitute is w_i^*
        delta_p_star, substitute = sorted_candidates.pop()

        # delta_p_star * word_saliency_array[position] equals H(x, x_i^*, w_i) defined in Eq.(7)
        substitute_tuple_list.append(
            (position, word, substitute, delta_p_star * word_saliency_array[position], None))

    # sort all the words w_i in x in descending order based on H(x, x_i^*, w_i)
    sorted_substitute_tuple_list = sorted(substitute_tuple_list, key=lambda t: t[3], reverse=True)

    # replace w_i in x^(i-1) with w_i^* to craft x^(i)
    NE_count = 0  # calculate how many NE used in a doc
    change_tuple_list = []

    #substitute_list = [ substitute for position, token, substitute, score, tag in sorted_substitute_tuple_list]
    #substitute_count = len(substitute_list)
    #perturbed_text_list = _compile_perturbed_tokens(perturbed_doc, substitute_list)

    #perturbed_text = gen(perturbed_text_list)
    #perturbed_text = " ".join(perturbed_text_list)
    #perturbed_doc = nlp(perturbed_text)
    perturbed_text =  input_text
    for (position, word, substitute, score, tag) in sorted_substitute_tuple_list:
        # if score <= 0:
        #     break
        #perturbed_text = ' '.join(_compile_perturbed_tokens(perturbed_doc, [substitute]))
        #perturbed_text_list = _compile_perturbed_tokens(perturbed_doc, [substitute])
        perturbed_text = perturbed_text.split(" ")
        perturbed_text[substitute.token_position]=substitute.candidate_word
        perturbed_text = " ".join(perturbed_text)
        
        substitute_count += 1
        if halt_condition_fn(perturbed_text):
            if verbose:
                print("use", substitute_count, "substitution; use", NE_count, 'NE')
            sub_rate = substitute_count / len(perturbed_text.split())
            NE_rate = NE_count / substitute_count
            return perturbed_text, sub_rate, NE_rate, change_tuple_list

    if verbose:
        print("use", substitute_count, "substitution; use", NE_count, 'NE')
    sub_rate = substitute_count / len(perturbed_text.split())
    NE_rate = NE_count / (substitute_count+1)
    return perturbed_text, sub_rate, NE_rate, change_tuple_list


def PWWS_snli(
        opt,
        input_text_p,
        input_text_h,
        true_y,
        dataset,
        word_saliency_list_h=None,
        rank_fn=None,
        heuristic_fn=None,  # Defined in adversarial_tools.py
        halt_condition_fn=None,  # Defined in adversarial_tools.py
        verbose=True):

    # defined in Eq.(8)
    def softmax(x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    heuristic_fn = heuristic_fn or (lambda _, candidate: candidate.similarity_rank)
    halt_condition_fn = halt_condition_fn or (lambda perturbed_text: False)


    substitute_count = 0  # calculate how many substitutions used in a doc
    substitute_tuple_list = []  # save the information of substitute word

    word_saliency_array_h = np.array([word_tuple[2] for word_tuple in word_saliency_list_h])
    word_saliency_array_h = softmax(word_saliency_array_h)

    #NE_candidates = NE_list.L[dataset][true_y]

    #NE_tags = list(NE_candidates.keys())
    use_NE = False  # whether use NE as a substitute #Changed by dxs

    max_len = config.word_max_len[dataset]

    import json
    with open(opt.certified_neighbors_file_path) as f:
        syn_dict = json.load(f)

    # for each word w_i in x, use WordNet to build a synonym set L_i
    for (position, word, word_saliency, tag) in word_saliency_list_h:
        if position >= max_len:
            break

        candidates = []
        candidates = _generate_synonym_candidates_from_dict(opt, word, position, syn_dict, rank_fn=rank_fn)

        if len(candidates) == 0:
            continue

        # The substitute word selection method R(w_i;L_i) defined in Eq.(4)
        sorted_candidates = zip(map(partial(heuristic_fn, input_text_p, input_text_h), candidates), candidates)
        # Sorted according to the return value of heuristic_fn function, that is, \Delta P defined in Eq.(4)
        sorted_candidates = list(sorted(sorted_candidates, key=lambda t: t[0]))

        # delta_p_star is defined in Eq.(5); substitute is w_i^*
        delta_p_star, substitute = sorted_candidates.pop()

        # delta_p_star * word_saliency_array[position] equals H(x, x_i^*, w_i) defined in Eq.(7)
        substitute_tuple_list.append(
            (position, word, substitute, delta_p_star * word_saliency_array_h[position], None))

    # sort all the words w_i in x in descending order based on H(x, x_i^*, w_i)
    sorted_substitute_tuple_list = sorted(substitute_tuple_list, key=lambda t: t[3], reverse=True)

    # replace w_i in x^(i-1) with w_i^* to craft x^(i)
    NE_count = 0  # calculate how many NE used in a doc
    change_tuple_list = []

    #substitute_list = [ substitute for position, token, substitute, score, tag in sorted_substitute_tuple_list]
    #substitute_count = len(substitute_list)
    #perturbed_text_list = _compile_perturbed_tokens(perturbed_doc_h, substitute_list)
    perturbed_text_h =  input_text_h

    for (position, word, substitute, score, tag) in sorted_substitute_tuple_list:
        # if score <= 0:
        #     break
        #perturbed_text = ' '.join(_compile_perturbed_tokens(perturbed_doc, [substitute]))
        #perturbed_text_list = _compile_perturbed_tokens(perturbed_doc, [substitute])
        perturbed_text_h = perturbed_text_h.split(" ")
        perturbed_text_h[substitute.token_position]=substitute.candidate_word
        perturbed_text_h = " ".join(perturbed_text_h)
        
        substitute_count += 1
        if halt_condition_fn(input_text_p, perturbed_text_h):
            if verbose:
                print("use", substitute_count, "substitution; use", NE_count, 'NE')
            sub_rate = substitute_count / len(perturbed_text_h.split())
            NE_rate = NE_count / substitute_count
            return input_text_p, perturbed_text_h, sub_rate, NE_rate, change_tuple_list

    if verbose:
        print("use", substitute_count, "substitution; use", NE_count, 'NE')
    sub_rate = substitute_count / len(perturbed_text_h.split())
    NE_rate = NE_count / (substitute_count+1)
    return input_text_p, perturbed_text_h, sub_rate, NE_rate, change_tuple_list
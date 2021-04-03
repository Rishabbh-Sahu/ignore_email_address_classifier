# -*- coding: utf-8 -*-
"""
@author: rishabbh-sahu
"""

import bert
import tensorflow_hub as hub

def create_bert_tokenizer(model_path):
    '''
    Languate en, bert tokenizer to split words in their appropriate sub-tokens and
    use the these as a part of embedding layer
    Bert variant tensorflow-hub model path : param model_path
    Bert tokenizer:return
    '''
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1",trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    print('vocabulary_file:',type(vocabulary_file),'\nto_lower_case:',type(to_lower_case))
    print('tokenizer.vocab:',len(tokenizer.vocab))
    return tokenizer

def tokenize_text(text,tokenizer):
    '''
    Text to tokenize : param text
    tokenizer used for word splitting : param tokenizer
    Stream of sub-tokens after tokenization:return
    '''
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


# ignore_email_address_classifier

#### About this project
This repo can be used for any text(english lanugage) classification task leveraging Bert-tokenizer, CNN layers and framework "Tensorflow 2.4". I've extended the vary concept of tokenization to email addresses and split them into sub-tokens. By using few specific keyworks like "donotreply", "no-reply", "unsubscribe" etc. in the email address, i'd created labels as important/ignore emails to enable supervised learning. Using the bert sub-tokens embedding and training them using CNN architecture, i'd able to achieve very decent accuracy that too with small_bert L2-H128 model. 

#### Getting started
- create virtual environment
- install tensorflow==2.4
- install requirements 
- Open config.yaml file and modify parameters as per your setup

#### For training
- python training.py 

#### Reference:
https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/


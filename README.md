# ignore_email_address_classifier

#### About this project
This repo can be used for any text (english lanugage) classification task leveraging **Bert-tokenizer, CNN layers and framework "Tensorflow 2.4"**. By using few specific keyworks like "donotreply", "no-reply", "unsubscribe" etc. in the email address, I'd created labels as important/ignore-emails to enable supervised learning. Extended the vary concept of tokenization to email addresses and split them into sub-tokens (sub-words). Using the bert sub-tokens embedding and training them using CNN architecture, achieved very good accuracy with very few epochs. To enable GPU support, please do enable CUDA-11 in windows/linux/mac virtual environment for tf2.4 or use CUDA 10.1 for tf2.3. 

#### Getting started
- create virtual environment
- install tensorflow==2.4
- install requirements 
- Open **config.yaml** file and modify parameters as per your setup

#### Data source
https://www.kaggle.com/wcukierski/enron-email-dataset - After close look for some random emails, created the label (important vs ignore) based on email-ids. for e.g. some keywords like noreply, unsubscribe, information etc. relates to marketing or emails from unknown sources at times hence falls under the category of ignore-emails. 

#### For training
- python training.py 

#### Model validation
- **Training accuracy after 1-epoch ~99%** 
- **Validation accuracy ~99%** <br>
*Note: Try running for more epochs if you have very less data to get good results*

#### Experiments:
1) Instead of using bert tokenizer, we can create our own word embeddings based on the dataset by using embedding layer and tune embedding's dimension based on f1-score OR used pre-learned ones like word2vec, GloVe etc. - for the reference, pls do visit model.py (updated links)
2) BPE (byte pair encoding) can be explored for text tokenization with vocab size 2k, 5k, 10k etc. Use this link to explore it further - https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/learn_bpe.py

#### Other use-cases:
To perform any text classification tasks like intent, sentiments, toxic comments identification etc. Below are some open source datasets for the reference.

#### Future scope:
Add manual intervention to input the email-id for classification. 

#### Datasets to explore:
- Toxic-comment-classification: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data <br>
- Sentiment analysis of IMDB reviews: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

#### Reference:
https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/


# API for Sentiment Analysis

This repo is an implementation of Google's paper on *Bidirectional Encoder Representations from Transformers*(BERT) for sentiment analysis on articles, using *huggingface*'s Pytorch NLP wrapper: https://github.com/huggingface/pytorch-transformers.

# Usage
The BERT model was placed in a RESTFul API which can run on any server to output the overall sentiment of the article.

**Model weights:** https://drive.google.com/file/d/1pBvpez7OTFjE7UAdeyDS0pqhA0PkXjvv/view?usp=sharing


1) Clone/download this repository
2) Install dependencies
```javascript
pip install -r requirements.txt
```
3) Download the model weights and add it to the folder
4) Launch the API
```javascript
python bert_inference.py 
```
Use **curl** or **POSTMAN** to post your article to the API. A sentiment(positive/negative) along with the confidence will be returned.

**_e.g. curl (IP address):8000 -F input=@path_to_your_article.txt_**

# Training Data
The model was fine-tuned using the training data of the IMDB movie review dataset which consist of 50k labelled training/testing data (evenly split). 
Download link for dataset: https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset

# Accuracy
The model achieved an accuracy of **92.6%** when tested on the 25k instances of test data in the IMDB dataset.
Model was trained on a GeForce GTX 1660 with sentence length of 256 and batch size of 6 for 4 epochs with a learning rate of 2e-5.

# Test on articles
The model was subsequently tested on 3 random news articles (in the articles folder), results displayed shows the overall sentiment of the article being positive/negative along with the confidence level:

1) Banks in Hong Kong condemn violence, urge restoration of 'harmony'
  **_Result: Negative (0.72)_**
2) Pompeo praises ‘US ally’ Denmark after Trump cancels visit
  **_Result: Positive (0.95)_**
3) Evidence suggests microplastics in water pose ‘minimal health risk’
  **_Result: Positive (0.89)_**


This repo is an implementation of Google's paper on *Bidirectional Encoder Representations from Transformers*(BERT) for sentiment analysis on articles, using *huggingface*'s Pytorch NLP wrapper: https://github.com/huggingface/pytorch-transformers.

The BERT model was placed in a Flask RESTFul API which can run on a server to output the overall sentiment of the article via any curl command
**_e.g. curl (IP address):8000 -F input=@article.txt_** 

**Data:**
The model was fine-tuned using the training data of the IMDB movie review dataset which consist of 50k labelled training/testing data (evenly split). 
Download link for dataset: https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset

**Accuracy:**
The model achieved an accuracy of **92.6%** when tested on the 25k instances of test data in the IMDB dataset.
Model was trained on a GeForce GTX 1660 with sentence length of 256 and batch size of 6 for 4 epochs with a learning rate of 2e-5.

**Test on articles:**
I proceeded to test this model on 3 random news articles (in the articles folder), results displayed is the argmax of the softmax scores of the overall sentiment of the article being positive/negative:

1) Banks in Hong Kong condemn violence, urge restoration of 'harmony'
  **_Result: Negative (0.72)_**
2) Pompeo praises ‘US ally’ Denmark after Trump cancels visit
  **_Result: Positive (0.95)_**
3) Evidence suggests microplastics in water pose ‘minimal health risk’
  **_Result: Positive (0.89)_**
  
**Weight for fine-tuned model:** Please head to this link to download the weight file for the fine-tuned model as described above: https://drive.google.com/file/d/1pBvpez7OTFjE7UAdeyDS0pqhA0PkXjvv/view?usp=sharing


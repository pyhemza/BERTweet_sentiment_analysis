# BERTweet_sentiment_analysis
<hr>
The language model BERT, the Bidirectional Encoder Representations from transformers and its variants have helped produce the state of the art performance results for various NLP tasks. 
These models are trained on the common English domains such as Wikipedia, news and books. The idea behind BERTweet is to train a model using the BERT architecture on a specific domain, 
which is twitter one of the most popular micro-blogging platforms, where users can share real time information related to all kind of topics events. 
Note that the characteristics of Tweets are generally different from those traditional written text such as Wikipedia and news articles, due to the typical short length of Tweets and frequent use of informal grammar as well as irregular vocabulary.
BERTweet authors has decided to train a language model for English Tweets using a 80Gb corpus of 850M English Tweets.

# Data 

In addition to the 3-class sentiment analysis dataset
from the **SemEval2017** Task 4A (Rosenthal et al., 2017)[6] 
and the 2-class irony detection dataset from the **SemEval2018** Task 3A (Van Hee et al., 2018)[7].
we evaluate and compare the performance of BERTweet,
on:
- **SemEval-2016** Task 6: Detecting Stance in Tweets (Mohammad et al, 2016)[8].
- **SemEval-2019** Task 5, subtask A : Hate Speech Detection against immigrants and women (Basile et al)[9]. 
- **SemEval-2019** Task 6, Sub-task A: Offensive language identification (Zampieri et al)[10]. 
- **SemEval-2020** Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets (Patwa et al)[11].
- **SemEval-2020** Task 12: Offensive Language Detection (Zampieri et al)[12].

![Performance of BERTweet](https://user-images.githubusercontent.com/56854458/116765121-04bd6e00-aa24-11eb-9612-b8235ef8698e.png)


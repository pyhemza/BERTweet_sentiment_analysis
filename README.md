# BERTweet_sentiment_analysis
<hr>
The language model BERT, the Bidirectional Encoder Representations from transformers and its variants have helped produce the state of the art performance results for various NLP tasks. 
These models are trained on the common English domains such as Wikipedia, news and books. The idea behind BERTweet is to train a model using the BERT architecture on a specific domain, 
which is twitter one of the most popular micro-blogging platforms, where users can share real time information related to all kind of topics events. 
Note that the characteristics of Tweets are generally different from those traditional written text such as Wikipedia and news articles, due to the typical short length of Tweets and frequent use of informal grammar as well as irregular vocabulary.
BERTweet authors has decided to train a language model for English Tweets using a 80Gb corpus of 850M English Tweets.

### Data 

In addition to the 3-class sentiment analysis dataset
from the **SemEval2017** Task 4A (Rosenthal et al., 2017).
and the 2-class irony detection dataset from the **SemEval2018** Task 3A (Van Hee et al., 2018).
we evaluate and compare the performance of BERTweet,
on:
- **SemEval-2016** Task 6: Detecting Stance in Tweets (Mohammad et al, 2016).
- **SemEval-2019** Task 5, subtask A : Hate Speech Detection against immigrants and women (Basile et al). 
- **SemEval-2019** Task 6, Sub-task A: Offensive language identification (Zampieri et al). 
- **SemEval-2020** Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets (Patwa et al).
- **SemEval-2020** Task 12: Offensive Language Detection (Zampieri et al).

We used the same procedure as in the paper, for each
dataset, we merge the training and the validation data
(if existed), and we sample 10% of the resulted set for
validation and use the remaining 90% for training. And
we use the test set provided by each of the SemEval
tasks.

### Normalization 

We use “soft” normalization strategy to all of the
experimental datasets, by translating word to tokens of
`user mentions` and `web/url` links into special tokens
`@USER`, and `HTTP/URL`, respectively, and the `emoji`
package to translate emotion icons into text strings

We employ the `transformers` library to preprocess the
data, the `tokenizer` has an option to normalize data before
tokenization.

As `BERT` architecture works with `fixed-length`
sequences. To choose the maximum length, we tokenize
the training set and take the length of the maximum
entry if it doesn’t exceed 128 tokens.

### Fine-tuning

We employ the `transformers` library (Wolf et al,
2019), to independently fine-tune `BERTweet` for each
task and each dataset in 30 training epochs. As in the
paper we append a linear prediction layer on the top of
the pooled output. (`class SentimentClassifier`)

- Optimizer : `AdamW` (Adam with L2 regularization
and weight decay) with a fixed learning rate of 1e-5.

- Loss function : Cross entropy loss

- Batch size : 32

### Results 

![Performance of BERTweet](https://user-images.githubusercontent.com/56854458/116765121-04bd6e00-aa24-11eb-9612-b8235ef8698e.png)
![Accuracy score comparaison of BERT, RoBERTa, BERTweet](https://user-images.githubusercontent.com/56854458/116765800-603d2b00-aa27-11eb-8058-fd81b4e65af5.png)
![F1 score comparaison of BERT, RoBERTa, BERTweet](https://user-images.githubusercontent.com/56854458/116765842-91b5f680-aa27-11eb-9c5f-6c2ad2698f0c.png)

On average BERTweet outperforms, the generic language models BERT and RoBERTa, by around 4% and
5%, respectively. The results reported above, confirm the
effectiveness of the large scale BERTweet for the tweet
classification tasks.

On average BERTweet outperforms, the generic language models BERT and RoBERTa, by around 4% and 5%, respectively. The results reported above, confirm the effectiveness of the large scale BERTweet for the tweet
classification tasks.

### Example:

Our results show that BERT model slightly outperform
RoBERTa on average, The results of BERTweet confirms
the effectiveness of a large scale specific pre-trained
language model for English tweets. To further understand
why BERTweet have better result.

we analysed the tweets correctly classified by BERTweet
and wrongly classified by the other two models

- Example well classified by BERTweet
![image](https://user-images.githubusercontent.com/56854458/116765945-3a645600-aa28-11eb-8192-63471f63f062.png)

- Definition of shweet in the Urban dictionnary :
![image](https://user-images.githubusercontent.com/56854458/116765971-58ca5180-aa28-11eb-92a8-b8aba38e2348.png)

### Implementation

- Python 3.6+, and PyTorch 1.1.0+.
- Install transformers: 
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip3 install --upgrade.
```
- Install emoji: `pip3 install emoji`

### Notebooks
- BERTweet finetuning implementation on google colaboratory with `PyTorch` in the nootebook `BERTweet.ipynb`.
- BERT base finetuning implementation on google colaboratory with `PyTorch` in the nootebook `BERT_base vs BERTweet.ipynb`.
- RoBERTa finetuning implementation on google colaboratory with `PyTorch` in the nootebook `RoBERTa vs BERTweet.ipynb`.

###
### References:

[1] J. Devlin, M.-W. Chang, K. Lee, and K.
Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human LanguageTechnologies, Volume 1 (Long and Short Papers), Minneapolis, Minnesota: Association for Computational Linguistics, Jun. 2019, pp. 4171–4186. DOI: 10.18653/v1/N19-1423. [Online]. Available: https://www.aclweb.org/anthology/N19-1423.

[2] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C.Delangue, A. Moi, P. Cistac, T. Rault, R. Louf,M. Funtowicz, and J. Brew, “Huggingface’s transformers: State-of-the-art natural language processing,” CoRR, vol. abs/1910.03771, 2019. arXiv: 1910.03771. [Online]. Available: http://arxiv.org/abs/1910.03771.

[3] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen,
O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized BERT pretraining approach,” CoRR, vol. abs/1907.11692,
2019. [Online]. Available: http://arxiv.org/abs/1907.11692.1908.

[4] D. Q. Nguyen, T. Vu, and A. T. Nguyen,
“BERTweet: A pre-trained language model for
English Tweets,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2020,pp. 9–14.

[5] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C.
Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M.
Funtowicz, J. Davison, S. Shleifer, P. von Platen,
C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao,
S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush, “Transformers: State-of-the-art natural language processing,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, Online: Association for Computational Linguistics, Oct. 2020, pp. 38–45. [Online]. Available: https://www.aclweb.org/anthology/2020.emnlp-demos.6.

[6] S. Rosenthal, N. Farra, and P. Nakov, “SemEval-
2017 task 4: Sentiment analysis in Twitter,” in
Proceedings of the 11th International Workshop
on Semantic Evaluation (SemEval-2017), Vancouver, Canada: Association for Computational Linguistics, Aug. 2017, pp. 502–518. DOI: 10.18653/v1/S17- 2088. [Online]. Available: https://www.aclweb.org/anthology/S17-2088.

[7] C. Van Hee, E. Lefever, and V. Hoste, “SemEval-
2018 task 3: Irony detection in English tweets,” in
Proceedings of The 12th International Workshop
on Semantic Evaluation, New Orleans, Louisiana:
Association for Computational Linguistics, Jun.
2018, pp. 39–50. DOI: 10.18653/v1/S18- 1005.
[Online]. Available: https : //www.aclweb.org/anthology/S18-1005.

[8] S. Mohammad, S. Kiritchenko, P. Sobhani, X.
Zhu, and C. Cherry, “SemEval-2016 task 6: Detecting stance in tweets,” in Proceedings of the
10th International Workshop on Semantic Evaluation (SemEval-2016), San Diego, California:
Association for Computational Linguistics, Jun.
2016, pp. 31–41. DOI: 10.18653/v1/S16- 1003.
[Online]. Available: https : www.aclweb.org/anthology/S16-1003.

[9] V. Basile, C. Bosco, E. Fersini, D. Nozza, V. Patti,
F. M. Rangel Pardo, P. Rosso, and M. Sanguinetti,
“SemEval-2019 task 5: Multilingual detection of
hate speech against immigrants and women in
Twitter,” in Proceedings of the 13th International
Workshop on Semantic Evaluation, Minneapolis,
Minnesota, USA: Association for Computational
Linguistics, Jun. 2019, pp. 54–63. DOI: 10.18653/v1/S19- 2007. 
[Online]. Available: https://www.aclweb.org/anthology/S19-2007.

[10] M. Zampieri, S. Malmasi, P. Nakov, S. Rosenthal,
N. Farra, and R. Kumar, “SemEval-2019 task 6:
Identifying and categorizing offensive language
in social media (OffensEval),” in Proceedings
of the 13th International Workshop on Semantic
Evaluation, Minneapolis, Minnesota, USA: Association for Computational Linguistics, Jun. 2019,
pp. 75–86. DOI: 10.18653/v1/S19-2010. [Online].
Available: https://www.aclweb.org/anthology/S19-2010


[11] P. Patwa, G. Aguilar, S. Kar, S. Pandey, S. PYKL,
B. Gamback, T. Chakraborty, T. Solorio, and A. ¨
7Das, “SemEval-2020 task 9: Overview of sentiment analysis of code-mixed tweets,” in Proceedings of the Fourteenth Workshop on Semantic Evaluation, Barcelona (online): International
Committee for Computational Linguistics, Dec.
2020, pp. 774–790. [Online]. Available: https://www.aclweb.org/anthology/2020.semeval-1.100.

[12] M. Zampieri, P. Nakov, S. Rosenthal, P.
Atanasova, G. Karadzhov, H. Mubarak, L. Derczynski, Z. Pitenis, and C¸ . C¸ oltekin, “SemEval- ¨
2020 task 12: Multilingual offensive language
identification in social media (OffensEval 2020),”
in Proceedings of the Fourteenth Workshop on
Semantic Evaluation, Barcelona (online): International Committee for Computational Linguistics,
Dec. 2020, pp. 1425–1447. [Online]. Available:https://www.aclweb.org/anthology/2020.semeval-1.188.










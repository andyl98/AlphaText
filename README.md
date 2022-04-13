**CSCI 2470 Project Checkpoint #2**

**Title:** AlphaText - A Text Summarization and Q&amp;A Chatbot for Extracting Information From Papers

**Who:**

- Bochen Fu - bfu18
- Jiatao Yuan - jyuan34
- Tianqi Liu - tliu31
- Zhuoran Han - zhan16

**Introduction:**

Researchers spend a great deal of time reading research papers. After reading a large number of papers, it probably becomes hard to remember every key information in all of the papers. When the researchers want to find a specific point within the papers, reading all of them from the beginning over and over again could be wasting time, so a chatbot that can answer questions about the papers would come in handy. In this paper, we applied Natural Language Processing to achieve this goal. We first used a text summarization model to extract the key points from the papers and then used a Q&amp;A model to answer the questions like what a key term in this paper means. We fine-tuned two pretrained models T5 and GPT3, then combined them together to get our final model.

**Related Works:**

**Text Summarization:**

- 1. T5: T5 stands for Text-To-Text Transfer Transformer, a transformer model built by Google AI. Unlike BERT, T5 takes both the input and the output in the format of text strings. This means the model can be used for almost any NLP related tasks.

Source: Raffel, Shazeer, et al. &quot;Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer&quot; arXiv preprint arXiv:1910.10683 (2019).

- 2. BERT: BERT is a state-of-the-art transformer-based language model. Unlike standard language models, which are unidirectional, BERT is bidirectional, which means it has more the choice of architectures that can be used during pre-training.

Source: Devlin, Jacob, et al. &quot;Bert: Pre-training of deep bidirectional transformers for language understanding.&quot; arXiv preprint arXiv:1810.04805 (2018).

- 3. PEGASUS: PEGASUS is a pre-training method of automatic summarization using Transformer. There are two main approaches to automatic summarization: extractive and abstractive.The automatic summarization in PEGASUS uses the abstractive approach. BERT&#39;s pre-training is based on the Masked Language Model (MLM), a method in which a portion of the input word is changed into a masked token and input into the model, then predicts the word based on the context before and after it. In this paper, it creates an abstract summary model. In addition to MLM, Gap Sentences Generation (GSG) is used to predict sentences by masking.

Source: Zhang, Jingqing, et al. &quot;Pegasus: Pre-training with extracted gap-sentences for abstractive summarization.&quot; International Conference on Machine Learning. PMLR, 2020

**Text Generation &amp; QA:**

- 1. Few-shot (GPT3): There are some approaches in pre-training models: Fine-tuning, few-shot, zero-shot. Among them, few-shot means giving a description of the task and a few examples of the task. Few-shot eliminates the need for training with task-specific unsupervised data. By training a large model on a large corpus, we were able to build a model that fits generally.

Source: Brown, Tom, et al. &quot;Language models are few-shot learners.&quot; Advances in neural information processing systems 33 (2020): 1877-1901.

**Methodology:**

**What is the architecture of your model?**

Our model architecture consists of two main parts: text summarization and Q&amp;A. In the first part (text summarization), we will be working with **T5** from Hugging Face Library. In the second part (Q&amp;A), we will be fine-tuning OpenAI&#39;s **GPT3** to adapt our dataset.

**How are you training the model?**

In the current stage of NLP, training a model that gives decent results most likely requires a huge amount of data as well as computational resources. Due to the tight schedule and limited amount of accessible resources, it is more realistic for us to focus on fine-tuning pretrained models as the building blocks for our final model.

**If you are doing something new, justify your design. Also note some backup ideas you may have to experiment with if you run into issues.**

The reason that we came up with this idea is about scale. Instead of storing entire research articles for the model to remember, we can first apply a text summarization model to extract the key information from articles, and then use a Q&amp;A model to demonstrate the &quot;knowledge&quot; the model has mastered. Intuitively, this combination can lead to two advantages. It reduces manual work to filter out insignificant information from papers. It also expands the amount of knowledge that a model can store in a limited capacity.

However, despite that state-of-the-art models achieve competitive results in each of the subcategories, combining multiple models together can lead to several problems. One main challenge is to correctly evaluate the performance of each of the two models. For example, if our model produces incorrect answers, finding out if this error is related with the Q&amp;A model or summarization model is complicated. Thus, we decide to separately evaluate the performance of the two models

**Metrics:**

We plan to run two separate experiments for our summary-Q&amp;A model.

For the text summarization model, we will tune the hyperparameters by feeding a series of research articles into the model. This model&#39;s performance will be evaluated based on a variety of metrics - ROGUE &amp; BLEU (F1 score), and brevity penalty. This set of standards can be further researched and depends on the interests of the stakeholders.

For the second part, we will use existing Q&amp;A datasets such as _SQuAD2.0_. For performance evaluation, similar to the previous model, we will also be using ROGUE &amp; BLEU (F1 score). In addition, the EM (Exact Match) metric might be used to find the proportion of predictions that match the ground truth exactly. We would also experiment with the _top\_k_ parameter in the Q&amp;A pipeline.

Our base goal is to successfully implement our pipeline and make sure our summary-Q&amp;A model is generating reasonable results. Our target goal is to fine-tune the model to achieve good performance on both the summarization and Q&amp;A part, with F1 score (~0.4). Our stretch goal is to improve the entire pipeline to be able to achieve a relatively high F1 score (~0.6+) on mainstream testing datasets.

**Dataset:**

**Text Summarization:**

- NIPS papers:[https://www.kaggle.com/datasets/benhamner/nips-papers](https://www.kaggle.com/datasets/benhamner/nips-papers)
  - Neural Information Processing Systems (NIPS) is one of the top machine learning conferences in the world. This dataset includes the title, authors, abstracts, and extracted text for all NIPS papers ranging from the first 1987 conference to the 2016 conference.

- ScisummNet Corpus: [https://www.kaggle.com/datasets/jawakar/scisummnet-corpus](https://www.kaggle.com/datasets/jawakar/scisummnet-corpus)
  - The ScisummNet(Summary of scientific papers)dataset provides over 1,000 papers in the ACL anthology network with their citation networks (e.g. citation sentences, citation counts) and their comprehensive, manual summaries.

**Q&amp;A:**

- SQuAD2.0 :[https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
  - Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles. SQuAD2.0 consists of over 150k questions, of which more than 35% are unanswerable in relation to their associated passage.
- Natural Questions: [https://ai.google.com/research/NaturalQuestions](https://ai.google.com/research/NaturalQuestions)
  - Natural Questions (NQ) is a new, large-scale corpus for training and evaluating open-domain question answering systems. Presented by Google, this dataset is the first to replicate the end-to-end process in which people find answers to questions. It contains 300,000 naturally occurring questions, along with human-annotated answers from Wikipedia pages, to be used in training QA systems.

**Ethics:**

**Why is Deep Learning a good approach to this problem?**

Research papers can have many complex features and as an end-to-end learning structure that has many layers of networks, deep learning can automatically extract and learn these features from texts or images. Therefore, using deep learning, we can effectively extract key information from research papers and generate answers for questions on these papers.

**Who are the major &quot;stakeholders&quot; in this problem, and what are the consequences of mistakes made by your algorithm?**

The major stakeholders will be the researchers who use the model to do their research, some research-related companies who want to use the model to make profit, the users who use the model for educational purposes. In addition to end users, other stakeholders may include engaging internal stakeholders, such as those who will maintain the chatbot or receive the chatbot&#39;s outputs. The consequences of mistakes made by the algorithm may lead to an inaccurate description or summarization of the papers, which will cause the researcher to be unable to grasp the key information of the article and maybe spend more time on finding the correct definition. And it will cause a bad experience for the user if it outputs some non-reasonable results.

**Division of labor:**

We will be working equally on the following sub-tasks:

- Data preprocessing
- Architecture design
- Text-summarization model
- Q&amp;A model
- Additional features
- Poster design

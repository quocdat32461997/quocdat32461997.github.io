---
layout: post
---

# Zero-Shot (0S) Learning in Natural Language Processing

## What is Zero-Shot Learning?
> If you don't know something, you can refer it to familar ones for guessing

Zero-shot learning aims at correctly predicting instances whose may not have been seen during training. 
The *0S* setting maps additional data attributes into high-dimensional features during **1. training stage** which could then be referred 
to during **2. inference stage** for unseen labels. Hence 0S is done in 2 stages:
- *Training* - knowledge about attributes are captured
- *Inference* - knowledge is used to categorize instances among a new set of classes without the need of fine-tuning

### Zero-Shot Learning in NLP
Speaking of the *0S* applications in NLP, *0S* has been recently formulated as:
- **Embedding approach** that [Chaudhary](https://amitness.com/2020/05/zero-shot-text-classification/) discusses using NN architectures
(e.g. Fully-Connected, LSTM) to embed both text and label (also in text format) as data attributes to perform *binary classification to decide
if label entails text*. Even though the concatenation of text and label embedding is not new, the entailment-like binary
classification avoids the challenge of softmax classification: *huge of number of classes*.


- **Transfer Learning** that utilizes pretrained Transformer-based language models
(e.g. BART, GPT-3) to read and link context of text and labels for data attributes. Two good examples are:
  - [GPT3 by OpenAI](https://arxiv.org/pdf/2005.14165.pdf) is pretrained and evaluated on few-shot and zero-shot settings.
  Look at the below figure, instead of fine-tuning, GPT3 is prompted for downstream tasks (e.g. translation) given
  data attributes in form of *description of task* and *few complete examples*.  
    ![Few-Shot Leaerning Evaluationo Settings in GPT-3](/assets/few-shot-learning-gpt3.png)
  - [Natural-Language-Inference-based for Text Classification by Yin et al.](https://huggingface.co/facebook/bart-large-mnli)
  take the concatenation of text and label (viewed as data attributes) as input and performs the entailment-like text classification. 

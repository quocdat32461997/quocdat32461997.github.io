---
layout: page
title: Portfolio
permalink: /portfolio/
---

This consists:

- [My timeline-based AI/ML/DL projects](#Projects)
- [My blogs/presentations/contributions](#Blogs-&-Workshops)
- [My publications](#Publications)
- [Interesting Work](#Interesting-Work)
- [Interview prep](#Interview-Prep)
- [What I have learnt](#What-I-have-learnt)

**Latest version** is at [here](https://kind-ginger-256.notion.site/machine-learning-portfolio-448355abb41b4b728989b5249f1dffa5)
---

## Projects

### 2018

- [Intro to Machine Learning final project: Titanic Survival Prediction](https://github.com/quocdat32461997/Vacation2AI/blob/master/others/titanic_survival_rate_prediction.pdf) Trained and validated multiple ML algorithms using *Scikit-learn* (**SVM, Random Forest, Decision Tree, Logistic Regression**) on Titatnic dataset to predict the survival rate of Titatnic passengers
- [TCReepy](https://github.com/NCBI-Hackathons/TCRecePy)
    - Implemented **K-Nearest-Neighbor** to learn informative positions of proteins in amino acids to distinguish two types of T cell receptor hypervariable CDR3 sequences.
    - Achived: Best Desk to Best Bedside Award at 2018 Med U-Hack at UT Southwestern

### 2019

- Neural Engineering research projects (2018 - 2019) advised by Prof. Tan Chin-Tuan at the Auditory Perception Engineering Laboratory, UT Dallas
- [Chest X-Ray Abnormal Detection](https://github.com/quocdat32461997/HealthCareAI_2019)
    - Applied multiple Transfer Learning models and hyper-parameter tuning to detect abnormalities in chest x-rays with 88% accuracy.
    - Achived: 1st Prize at the HealthCare AI 2019 Hackathon at Uni. of Texas at Dallas

### 2020

- [MoCV](https://pypi.org/project/MoCV/)
    - An open-source Python package implementing **Computer Vision and Image Processing algorithms**.
- [Senior Co-op Project: Interaction Tunning Tool](https://quocdat32461997.github.io/2020/05/05/interaction-tuning-tool-the-end_to_end-deep-learning_system-for-intent-detection.html)
    - Led a team of 6 engineers to build and deploy **a no-code Intent Extraction system** to **reduce the manual intent labeling tasks** (no coding and domain knowledge required) for Chatbot data preparation.
    - Contribution: utilized *StanfordNLP and Tensorflow* to develop a Deep Learning model (**LSTM-Attention + MLP**) to extract intents from raw utterances (**75% accuracy in development and 20% in deployment**). Unlike Google Dialogflow using a fixed intent list, our system forms VERB-NOUN intents that it does **not limit iteself by industry domains**
- [Name Entity Recognizer](https://github.com/quocdat32461997/NER) Implemented **BiLSTM-CRF** for **Name Entity Recognition**, built the data pipeline in *Tensorflow*, and deploy in *Flask*
- [Intent Classifier](https://github.com/quocdat32461997/intent_classifier)
    - Trained **Suport-Vector-Machine (SVM) and GradientBoosting** on text features extracted by **TF-IDF** for Intent-Classification tasks. Accuracy: 97% for training and 80% validation
- [Borot](https://github.com/quocdat32461997/borot)
    - Built **Chatbot Question & Answering** with *Flask, Scikit-learn, Tensorflow, and SQL*.
    - Implemented **Information Retrieval** with **Intent Classifier (SVM), Name-Entity-Recognizer (BiLSTM-CRF) and TF-IDF** to retrieve answers in response to questions. Implemented OOP to collect users’ QA queries for personalization.
- [Mask-RCNN](https://github.com/quocdat32461997/Mask_RCNN) - Implementation of Mask-RCNN in Tensorflow >= 2.0.0

### 2021

- [Emorecom, ICDAR2021 Competition – Multimodal Emotion Recognition on Comic scenes](https://github.com/aisutd/emorecom)
    - Developed a multimodal Deep Learning model composed of CNN (ResNet, FaceNet) for visual features and RNN/BERT for textual features to detect emotion on comic scenes. Ranked 13th.
    - Utilized Tensorflow Data/string/image and OpenCV to build image/text augmentation pipeline and the TFRecord data pipeline.
- [h-at (Information Extraction)](https://github.com/quocdat32461997/h-at)
    - Developed a Python program to extract template-based information by designing rules and utilizing Dependency Parsing, Name Entities, and POS tags.
- [Find-unicorns (search engine)](https://github.com/quocdat32461997/find_unicorns)
    - Developed a search engine by crawling & scrapping (>100k links), implementing ranking algorithms (TF-IDF, HITS, PageRank).

---

## Blogs and Workshops

[Model Compression](https://www.notion.so/Model-Compression-ed06ffecfe0b4942936923c86453552b)

[Machine Learning with Google Cloud Platform (GCP)](/gcp)

[Install Tensorflow on AMD GPUs](https://medium.com/analytics-vidhya/install-tensorflow-2-for-amd-gpus-87e8d7aeb812)

[Blogs on Deep Learning](https://datngo-79115.medium.com/)

[Entry-ML Workshops @DSC-UTD](https://github.com/DSC-UTDallas/explore-ml) - workshops over fundamental machine learning algorithms

[Introduction to Python](https://github.com/pekkalacd/Introduction-to-Python)

[Workshop on Multimodal Emotion Recognition @AIS](https://www.youtube.com/watch?v=4XAO4IOH5ks&t=52s)

---

[Zero-Shot Learning in NLP](https://www.notion.so/Zero-Shot-Learning-in-NLP-2f0f65ec06dc4dc589f23cc8856df57f)

[Model optimization](https://www.notion.so/Model-optimization-4eaea05ebf26415780f7a528f3138bf4)

## Publications

- [Neural Entrainment to Speech Envelope in response to Perceived Sound Quality](https://ieeexplore.ieee.org/abstract/document/8717078/)
    - Authors: **Dat Quoc Ngo**, Garret Oliver, Gleb Tcheslavski, Chin-Tuan Tan.
    - Affiliation: Undergraduate Research Assistant at Auditory Perception Engineering Laboratory, UT Dallas
    - Status: [Accepted to IEEE Neural Engineering Conference 2019](https://ieeexplore.ieee.org/abstract/document/8717078/)
- [Linear and Nonlinear Reconstruction of Speech Envelope from EEG](https://quocdat32461997.github.io/assets/linear_and_nonlinear_reconstruction_of_speech_envelope_from_eeg.pdf)
    - Authors: **Dat Quoc Ngo**, Garret Oliver, Gleb Tcheslavski, Fei Chen, Chin-Tuan Tan.
    - Affiliation: Undergraduate Research Assistant at Auditory Perception Engineering Laboratory, UT Dallas
    - Status: [Preprint](https://quocdat32461997.github.io/assets/linear_and_nonlinear_reconstruction_of_speech_envelope_from_eeg.pdf)
- [Depression Detection: Text Augmentation for Robustness to Label Noise in Self-reprots](https://github.com/quocdat32461997/quocdat32461997.github.io/blob/master/assets/ACL_IJCNLP_2021___Final.pdf)
    - Authors: **Dat Quoc Ngo**, Aninda Bhattacharjee, Tannistha Maiti, Tarry Singh, Jie Mei
    - Affiliation: Visiting AI Researcher at [deepkapha.ai](http://deepkapha.ai/)
    - Status: [in submission](https://github.com/quocdat32461997/quocdat32461997.github.io/blob/master/assets/ACL_IJCNLP_2021___Final.pdf)

---

## Interesting Work

### Computer Vision

- [You Look Only ONce](https://arxiv.org/abs/1506.02640)
    - A unified Object Detection model consisting of DarkNet (Deep CNN) and Non-Max-Suppression at the output layer to group multiple bounding-boxes for the final object. Multi losses (object loss, class loss, and position loss) were computed.
- [Mask RCNN](https://arxiv.org/pdf/1703.06870.pdf)
    - An instance segmentation model composed of backbone (ResNet or Feature Pyramid Network), Region Proposal Network (simple CNN), and ROIAlign layer (Non-Max-Suppression) that both share hidden feature from the backbone.
    - Prone to overfitting since using Fully Connected layer for prediction.
- [Single Image Super Resolution based on a Modified U-NET with Mixed Gradient Loss](https://arxiv.org/pdf/1911.09428.pdf)
    - Introduction to loss functions (MGE and MixGE) for Super-Image-Super-Resolution problems using U-NET.
    - MSE, a common loss function is limited to learn errors based on pixel values, not the object curve (aka gradient error)
    - To solve gradient error introduced by Super-Resolution, Mean-Gradient-Error (MGE) utilizes Sobel operator to shapren curves of objects in predicted and true images which are then computed into difference-square

### Natural Language Processing

- [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991v1.pdf)
    - Implementation of BiLSTM-CRF for Sequence Tagging tasks.
- [Attention is all you need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
    - Attention-based Seq-2seq model for state-of-the-art Neural Machine Translation
- [BERT: Pretraining of Deep Bidirectional Transformers for Langauge Understanding](https://arxiv.org/pdf/1810.04805.pdf)
    - A Deep Transformer-based Encoder pretrained on Wikipedia and BookCorpus allows easiy fine-tuning for downstream tasks.
    - Make text-based Transfer Learning feasible
    - Pros: effectively extract context
    - Cons: due to unsupervised pretraining tasks (i.e. random input masks) and individually attention-weight computing of each token across text, BERT ignores the sequential dependency of tokens (aka autogressive learning). Hence, BERT is not suitable for several text generation tasks such as Machine Translation.
- [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
    - XLNet (Autogressive Language Modeling) implements [Permutation Language Modeling](https://jmlr.org/papers/volume17/16-272/16-272.pdf) to generate permutation of token positions. This allows XLNet to learn dependency between a token and tokens before it only which represent for Autogressive Language Modeling (remember RNN and LSTM?).
- [Conditional BERT Contextual Augmentation](https://arxiv.org/pdf/1812.06705.pdf)
    - Proposed Conditional Mask Language Modeling (Conditional MSM) and a novel text data augmentation for labeled sntences to train Conditinoal BERT.
    - Idea : replace Segmentation Embedding with vocabulary-indexed labels when pretraining.
    - Result : Conditional BERT is applicable for contextual data augmentation that improves BERT in several NLP tasks. Also, Conditional BERT could be used for text style transfer (replacing words w/o changing context).
- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper, and lighter](https://arxiv.org/abs/1910.01108v1)
    - A smaller version of BERT with equivalent performance but less number of layers (6 in Distil BERT and 12 in BERT)
    - More practical for real-time inference
- [COMET: Commonsense Transformers for Automatic Knowledge Graph Construction](https://arxiv.org/pdf/1906.05317.pdf)
    - A Transformer-based model to generate/complete knowledge graphs.
    - Input encoding:
        - Consists of a 3-element tuple {subject, relation, object}
        - Unlike BERT w/ 3 embedings, COMET has 2 embeddings: tokens and positions
    - Architecture is similar to BERT, except the output layer of Transformer
- [Unsupervised Commonsense Question Answering with Self-Talk](https://arxiv.org/pdf/2004.05483.pdf)
    - To be added
- [Visual Question Answering To-read-list](https://github.com/zixuwang1996/VQA-reading-list)
    - Collection of to-read publications regarding Visual Question Answering

### Others

### Label Noise

- [Label Noise Types and Their Effects on Deep
Learning](https://arxiv.org/pdf/2003.10471.pdf)
    - Investigate the effects of label noise types to Deep Learning
    - Label noise is common in large-scale datasets or in active learning when samples are labeled by non-experts
- [Exploiting Context for Robustness to Label Noise in Active Learning](https://arxiv.org/pdf/2010.09066.pdf)
    - Proposed to exploit graphical context to improve CNN's performance in classification tasks.

### Knowledge Distillation

- [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
    - A literature review of Knowlege Distillation techniques
    - By:
        - Knowledge: response-based, feature-based, and relation-based
        - Schemes: offline-distillation, online-distillation, and self-distllation
        - Algorithms: attention-based, adversarial-based, cross-modal, multi-teacher, graph-based, data-free, quantization-based, NAS-based
    - Knowledge distillation's applications in CV, NLP, Speech Recognition
    - I personally found **data-free** algo useful when huge distilling huge models to smaller models (e.g. T5-3B to T5-small) without data in-hand.
- [Adversarial Self-Supervised Data-Free Distillation for Text Classification](https://www.aclweb.org/anthology/2020.emnlp-main.499.pdf)
    - A data-free distillation algorithm to distill BERT-large to smaller BERTs without ready in-hand.
    - Idea: use BERT-teacher to generate and optimize synthetic embeddings (i.e. ignore discrete text that is hard to generate) that serve as inputs to BERT-student.
    - Update: I utilized this method to distil T5-3B to T5-small with minor modifications for Grammar Error Correction.
- [Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation](https://arxiv.org/pdf/1905.08094.pdf)
    - An self-distillation scheme for ResNet that each low-level module is a student network. In training, the teacher (whole ResNet) and students (sub-ResNet modules) are trained and optimized simulatenously by losses: KL-divergence & cross-entropy
    - Notes: useful when distilling huge models (e..g T5-3B, T5-11B, or GPT-#)

---

## Interview Prep

- [ML Engineer Notes](https://docs.google.com/document/d/1mo1edEotJDpvT4fxL8-_VFBwS0e4OHqLnxd5I0yy-ps/edit?usp=sharing)

## What I have learnt

Graduate courses lised only

- [Natural Language Processing](/nlp-course)
- [Machine Learning](/ml-course)
- [Information Retrieval](/ir-course)
- [Machine Learning System Design](https://docs.google.com/document/d/1yOJagVolemWpgms_TkBlEi866qLi7p76m4aMjkPcU6Q/edit?usp=sharing)
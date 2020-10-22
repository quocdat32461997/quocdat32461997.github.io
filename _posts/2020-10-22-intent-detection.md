---
layout: post
---

## Time
Spring 2020 @University of Texas at Dalls

## Members
My team of 6 Senior CS/SE students in Senior Project course:
* Matthew Kunjammen, Backend Engineer (Database)
* Johnathan Kim, Backend Engineer
* Lan Vu, Frontend Engineer
* Tung Vu, Fronend Engineer
* Frank Yang, Full-Stack Engineer
* Dat ngo, Machine Learning Engineer

## Goals
* To improve the customer service in multiple industry domains (e.g. call bots in banking system)
* Cooperated with Concentrix supervisors to build a web-based tool to detect and generate intents withdrawn from given utterances.
* Allow Speech scientists without strong coding skills easily remove noise, correct misspelling, and detect intents. 

## Achievements
* An end-to-end web tool to automate and reduce the manual intent-labeling workload of scientists and engineers
* Web tool requires zero coding skills
* Core is a hybrid and Attention-based model to detect natural intents from text (not pre-defined intents as Google DiagFlow) with accuracy 70%


## Tech in use:
* Frontend: VueJS
* Backend: NodeJS, Flask, SQL
* AI/ML: Tensorflow/Keras

## Challenges:
* Dataset size: 10MB corpus of text translated from customer-agent conversaions. Available data is expensive to gather
* Limited labeled data: given 15000 intents, there are 5000 labeled intents
* Need an end-to-end system with GUI for scientists without coding skills to use

## Solutions:
* A web tool that allows scientists to upload corpus and retrieve intents. The corpus/detected intents are saved for each scientist account for later use.
 - Architecture (modules communicate by RESTful API): 
  - Backend (NodeJS + SQL)
  - AI/ML module (Tensorflow/Keras)
  - Frontend (VueJS)
* The AI module:
 * Intent is defined as tuple of Vert-Noun and Noun-Noun.
 * A hybrid model of 2 modules:
  - Module 1: detect tuples of Verb-Noun and Noun-Nouns as potential intents
  - Module 2: given potential intents, finalize the correct intent
  -  The model is built around:
   - Pre-trained Word Embeddings from Stanford NLP
   - Bidirectional Long-Short-Term-Memory (BiLSTM)
   - Conditional Random Field (CRF)
   - Multilayer Perceptron
 * The model is implemented in OOP content that allows easy maintaince 

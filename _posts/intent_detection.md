---
layout: post
---

# Intent Detection

* Time: Spring 2020 @University of Texas at Dalls
* Team: My team of 6 Senior CS/SE students
 - Matthew Kunjammen, Backend Engineer (Databse)
 - Johnathan Kim - Backend Engineer
 - Lan Vu - Frontend Engineer
 - Tung Vu - Frontend Engineer
 - Frank Yang - Full-Stack Developer
 - Dat Ngo - Machine Learning Engineer

* Goal: Cooperated with Concentrix supervisors to build a web-based tool to detect and generate intents withdrawn from given utterances. Our tool helps Speech scientists without strong coding skills easily remove noise, correct misspelling, and detect intents. Our core AI section is developed around 2 separate models, Potential Intents and True Intents models which respectively generate potential intents and finalize mostly correct intents.
* Achievement:
 - Developed **a hybrid Attention-based model** to retrieve **natural intents (from text, not predefiend intents) with accuracy at 70%**
 - Autoamted the process to **reduce the manual intent-labeling workload of scientists and engineers**
 - A web tool does **not require coding skills**


## Tech in use:
* Frontend : VueJS
* Backend: NodeJS, Flask, SQL RESTful API 
* Machine Learning: Tensorflow/Keras
* NLP/NLU model: 
 - Pre-trained Word Embedding from Standford NLP
 - Bidirectional Long-Short-Term-Memory (BiLSTM)
 - Self-Attentin
 - Conditional Random Field 
 - Multilayer Perceptron 

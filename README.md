# Calpric: Inclusive and Fine-grain Labeling of Privacy Policies with Crowdsourcing and Active Learning
This GitHub page serves as the AE of Usenix Security 2023 for Calpric: Inclusive and Fine-grain Labeling of Privacy Policies with Crowdsourcing and Active Learning.


## Introduction
We include three artifacts here: the Calpric Privacy Policy Corpus (CPPS), a customized pre-trained BERT-based embedding using privacy policy texts, and a source code example of the crowdsourcing and active learning components of the Calpric category model.  


## Dataset
The CPPS data set includes privacy policy segment labels covering 9 data categories (contact, device, location, health, financial, demographic, survey, social media, and personally identifiable information) with 3 data actions (collect/use, share, and store). For clarity purposes, duplicated labels have been removed, resulting in a total of 12,585 labels. The dataset is in CSV format. 


## Required packages for the source code example
- Python standard library 
- re
- langdetect 
- numpy  
- os 
- pandas 
- math
- keras
- modAL
- tensorflow 

## Required packages for CPPS:
- Python standard library 
- csv

## Functionality Test
1. CPPS check

## This repository is based on the following work:
Wenjun Qiu, David Lie and Lisa Austin, “Calpric: Inclusive and Fine-grained Labeling of Privacy Policies with Crowdsourcing and Active Learning”, In Proceedings of the 32th USENIX Security Symposium, 2023. (To appear.)


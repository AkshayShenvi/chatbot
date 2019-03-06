# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:40:34 2019

@author: Akshay Shenvi
"""

# importing Libraries
import numpy as np
import tensorflow as tf
import re
import time

# importing the dataset
lines=open('movie_lines.txt',encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations=open('movie_conversations.txt',encoding = 'utf-8', errors = 'ignore').read().split('\n')

#creating a dictionary
idtoline={}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line)== 5:
        idtoline[_line[0]]=_line[4]
        
# creating a list of the conversations
conversations_id=[]
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_id.append(_conversation.split(','))
    
#seperating questions and answers
questions=[]
answers=[]
for conversation in conversations_id:
    for i in range(len(conversation)-1):
        questions.append(idtoline[conversation[i]])
        answers.append(idtoline[conversation[i+1]])
# First clean of text
def clean_text(text):
    text= text.lower()
    text= re.sub(r"i'm","i am",text)
    text= re.sub(r"he's","he is",text)
    text= re.sub(r"she's","she is",text)
    text= re.sub(r"that's","that is",text)
    text= re.sub(r"what's","what is",text)
    text= re.sub(r"\'ll","will",text)
    text= re.sub(r"\'ve","have",text)
    text= re.sub(r"\'re","are",text)
    text= re.sub(r"\'d","would",text)
    text= re.sub(r"won't","will not",text)
    text= re.sub(r"can't","cannot",text)
    text= re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","",text)
    return text
                    
# Clean Questions
clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))
    
# Clean Questions
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))
    
    
# Creating a dictionary that will count the number of occurences
wordtocount={}
for question in clean_questions:
    for word in question.split():
        if word not in wordtocount:
            wordtocount[word]= 1
        else:
            wordtocount[word]+=1

for answer in clean_answers:
    for word in answer.split():
        if word not in wordtocount:
            wordtocount[word]= 1
        else:
            wordtocount[word]+=1
    
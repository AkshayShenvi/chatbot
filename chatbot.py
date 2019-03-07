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
    
# Creating dictionaries that map the question words and answer words
threshold = 20
questionswordstoint = {}
word_number=0
for word, count in wordtocount.items():
    if count >= threshold:
        questionswordstoint[word]=word_number
        word_number+=1
answerswordstoint ={}
word_number=0
for word, count in wordtocount.items():
    if count >= threshold:
        answerswordstoint[word]=word_number
        word_number+=1
        
#Adding tokens to dictionaries
tokens=['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionswordstoint[token] = len(questionswordstoint)+1
for token in tokens:
    answerswordstoint[token]= len(answerswordstoint)+1  
    
    
# creating inverse dictionary for words to answers
answersinttowords={w_i:w for w,w_i in answerswordstoint.items()}

# Addind EOS to clean_answers
for i in range(len(clean_answers)):
    clean_answers[i]+=' <EOS>'

# Give all quest and ans into integers
# replace the words that were filtered by <OUT>
questions_to_int= []
for question in clean_questions:
    ints=[]
    for word in question.split():
        if word not in questionswordstoint:
            ints.append(questionswordstoint['<OUT>'])
        else:
            ints.append(questionswordstoint[word])
    questions_to_int.append(ints)
answers_to_int= []
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word not in answerswordstoint:
            ints.append(answerswordstoint['<OUT>'])
        else:
            ints.append(answerswordstoint[word])
    answers_to_int.append(ints)
    
#Sorting questions and answers by length
sort_clean_questions=[]
sort_clean_answers=[]
for length in range(1,25+1):
    for i in enumerate(questions_to_int):
        if len(i[1])==length:
            sort_clean_questions.append(questions_to_int[i[0]])
            sort_clean_answers.append(answers_to_int[i[0]])
    
        
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
    text= re.sub(r"where's","where is",text)
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
    
     
        
#Creating placeholders for inputs and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32,[None,None],name='input')
    targets = tf.placeholder(tf.int32,[None,None],name='target')
    lr = tf.placeholder(tf.float32,name='learning_rate')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    return inputs, targets, lr, keep_prob


# Preprocessing the targets
def preprocess_targets(targets, wordtoint, batch_size):
    left_side = tf.fill([batch_size,1],wordtoint['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size,-1],[1,1])
    preprocess_targets = tf.concat([left_side,right_side],1)
    return preprocess_targets

# Encoder RNN Layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm= tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    encoder_output , encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state
    
#decoding the training set
def decode_training_set(encoder_state,decoder_cell,decoder_embeded_input,sequence_length, decoding_scope,output_function,keep_prob,batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys, attention_values,attention_score_function,attention_construct_function= tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau',num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name='attn_dec_train')
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embeded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)

# Decoding the test set
def decode_test_set(encoder_state,decoder_cell,decoder_embeded_matrix,sos_id,eos_id, maximum_length,num_words,sequence_length, decoding_scope,output_function,keep_prob,batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys, attention_values,attention_score_function,attention_construct_function= tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau',num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeded_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name='attn_dec_inf')
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    
    return test_predictions
    
# Creating the Decoder RNN
def decoder_rnn(decoder_embeded_input, decoder_embeddings_matrix,encoder_state, num_words, sequence_length,rnn_size, num_layers, wordtoint, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm =tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope= decoding_scope,
                                                                      weights_initializer= weights,
                                                                      biases_initializer= biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embeded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           wordtoint['<SOS>'],
                                           wordtoint['<EOS>'],
                                           sequence_length -1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
            
#Seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embidding_size, decoder_embidding_size, rnn_size, num_layers, questionswordstoint):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embidding_size,
                                                              initializer= tf.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocess_targets = preprocess_targets(targets, questionswordstoint, batch_size)
    decoder_embiddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embiddings_matrix, preprocess_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embiddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswordstoint,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
    
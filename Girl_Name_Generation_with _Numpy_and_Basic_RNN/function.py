# -*- coding: utf-8 -*-
"""
"""

from scipy.io import loadmat
import tensorflow as tf
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def RNN_cell(parameters, char_to_idx, seed):
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    # Creating a zero vector x that can be used as the one-hot vector 
    # representing the first character. 
    x = np.zeros((vocab_size,1))
    
    # Initialize hidden state at previous time step as zeros.
    a_prev = np.zeros((n_a,1))
   
    # Create an empty list of indices.
    indices = []
    
    # idx is the index of the one-hot vector x that is set to 1. In this case, initialize idx to -1.
    idx = -1 
    
    counter = 0
    newline_character = char_to_idx['\n']
    
    while (idx != newline_character and counter != 50): # maximize the number of character in each word to 50.
        
        # Forward propagate x.
        a = np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+b)
        z = np.dot(Wya,a)+by
        y = softmax(z)
        
        # Sample the index of a character within the vocabulary from the probability distribution y
        # so that it will not always generate the same character given a previous character.
        idx = np.random.choice(vocab_size, p = y.ravel())

        # Append the index to "indices"
        indices.append(idx)
        
        # Overwrite the input x with one that corresponds to the sampled index `idx`.
        x = np.zeros((vocab_size,1))
        x[idx] = 1
        
        # Update the hidden state to current time steps.
        a_prev = a
        
        counter +=1

    if (counter == 50):
        indices.append(char_to_idx['\n'])
    
    return indices

def initialize_parameters(n_a, n_x, n_y):
    
    Wax = np.random.randn(n_a, n_x)*0.01  # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01  # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01  # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias
    
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) # hidden state
    p_t = softmax(np.dot(Wya, a_next) + by) # unnormalized log probabilities for next chars # probabilities for next chars 
    
    return a_next, p_t

def rnn_forward(X, Y, a0, parameters, vocab_size = 29):
    
    # Initialize x, a and y_hat as empty dictionaries
    x, a, y_hat = {}, {}, {}
    
    a[-1] = np.copy(a0)
    
    # initialize loss to 0
    loss = 0
    
    for t in range(len(X)): #iterate every time step
        
        # Set x[t] to be the one-hot vector representation of the t'th character in X.
        # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector. 
        x[t] = np.zeros((vocab_size,1)) 
        if (X[t] != None):
            x[t][X[t]] = 1
        
        # Run one step forward of the RNN
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        
        # Update the loss by substracting the cross-entropy term of this time-step from it.
        loss -= np.log(y_hat[t][Y[t],0])
    
    # Store the result in a cache which will be useful for backpropagation
    cache = (y_hat, a, x)
        
    return loss, cache

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] 
    daraw = (1 - a * a) * da 
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    
    return gradients

def rnn_backward(X, Y, parameters, cache):
    
    # Initialize gradients as an empty dictionary
    gradients = {}
    
    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    
    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    

    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])

    
    return gradients, a

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters

def gradientClip(gradients, max_value):
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
   
    for gradient in [dWax, dWaa, dWya, db, dby]:
        
        np.clip(gradient, -max_value, max_value, out = gradient)
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    
    # Forward propagate through time
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    
    # Backpropagate through time
    gradients, a = rnn_backward(X, Y, parameters, cache)
    
    # Clip gradients between -5 (min) and 5 (max)
    gradients = gradientClip(gradients, 5)
    
    # Update parameter
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    return loss, gradients, a[len(X)-1]

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    print ('%s' % (txt, ), end='')
    

def model(data, idx_to_char, char_to_idx, epochs = 200001, n_a = 50, girl_names = 8, vocab_size = 29):
    
    # Retrieve n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size
    
    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # Build list of all girl names.
    with open("girl_name.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    
    # Shuffle list of all girl names
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # Initialize the hidden state
    a_prev = np.zeros((n_a, 1))
    
    # Optimization loop
    for j in range(epochs):
       
        # Set the index `idx`
        idx = j%len(examples)
        
        # Set the input X
        single_example = examples[idx]
        single_example_chars = [c for c in single_example]
        
        single_example_idx =  [char_to_idx[c] for c in single_example_chars]
       
        X = [None] + single_example_idx
        
        # Set the labels Y
        idx_newline = char_to_idx['\n']
        Y = single_example_idx+[idx_newline]
        
        
        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        
        loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)

        # Print output
        if j % (epochs-1) == 0:
            
            # The number of girl names to print
            for name in range(girl_names):
                
                # Sample indices and print them
                sampled_indices = RNN_cell(parameters, char_to_idx, 0)
                print_sample(sampled_indices, idx_to_char) 
      
            print('\n')
        
    return parameters
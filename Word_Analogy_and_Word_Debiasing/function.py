# -*- coding: utf-8 -*-
"""
"""

def read_glove_vecs(glove_file):
    
    with open(glove_file,'r', encoding="utf8", ) as f:
        
        words = set()
        word_to_vec_map = {}
        
        print(f)
        for line in f:
            
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

def cosine_similarity (vector_u, vector_v):
    
    dot_product = np.dot(vector_u, vector_v)
    
    norm_u = np.sqrt(np.sum(vector_u**2))
    norm_v = np.sqrt(np.sum(vector_v**2))
    
    cosine_similarity = dot_product/(norm_u*norm_v)
    
    return cosine_similarity

def word_analogy(word_1, word_2, word_3, word_to_vec_map):
    
    word_1, word_2, word_3 = word_1.lower(), word_2.lower(), word_3.lower()
    
    vector_a, vector_b, vector_c = [word_to_vec_map.get(x) for x in [word_1, word_2, word_3]]
    
    bag_of_words = word_to_vec_map.keys()
    max_cos_similarity = -1000
    analogous_word = None
    
    input_words = set([word_1, word_2, word_3])
    
    for i in bag_of_words:
        
        if i in input_words:
            continue
        
        cos_similarity = cosine_similarity((vector_b-vector_a), (word_to_vec_map[i]-vector_c))
        
        if cos_similarity > max_cos_similarity:
            max_cos_similarity = cos_similarity
            analogous_word = i
    
    return analogous_word

def word_debiasing(word, vec_gender, word_to_vec_map):
    
    vec_word = word_to_vec_map[word]
    
    vec_word_biased = np.dot(vec_word, vec_gender)*vec_gender/(np.sum(vec_gender**2))
    
    vec_word_debiased = vec_word - vec_word_biased
    
    return vec_word_debiased

"""未使用该模块"""
import numpy as np
import heapq,re
from gensim.models import word2vec
from nltk import pos_tag

def P_next(P,length,d,A):
    E = np.ones((length,length))
    next_P = ((1-d)/length)*E + d*A*P
    return next_P

def w2f(threshold,sent,model,d):
    total_word_sim_list = []
    N = []
    word_list = []
    bad_words = set()
    # for word in sent.split():
    #     if re.search(r'_+\S+|\S+_+', word):
    #         new_word = word.strip('_')
    #         sent = re.sub(word, new_word, sent)
    # for word, pos in pos_tag(sent.split()):
    #     if pos not in ['NN','VB','NNS','NNP','NNPS','VBD','VBG','VBN','VBP','VBZ']:
    #         bad_words.add(word)
    #         continue
    for word in sent.split():
        try:
            model.similarity(word, word)
        except:
            bad_words.add(word)
            continue
        word_sim_list = []
        n = 0
        other_word_list = []
        for other_word in sent.split():
            if not other_word in bad_words:
                try:
                    sim = model.similarity(word, other_word)
                    if sim >= threshold:
                        word_sim_list.append(1)
                        n += 1
                    else:
                        word_sim_list.append(0)
                    other_word_list.append(other_word)
                except:
                    bad_words.add(other_word)
                    continue
            else:continue
        if n != 0:
            word_sim_list = list(map(lambda x: x / n, word_sim_list))
            total_word_sim_list.append(word_sim_list)
            N.append(n)
            word_list.append(word)
        else:
            continue
    length = len(total_word_sim_list)
    if length != 0:
        P = np.array(total_word_sim_list)
        A = np.transpose(P)
        next_P = P_next(P, length, d, A)
        result = np.linalg.norm(A*P - next_P, 2)
        while result >= 0.000001:
            P = next_P
            next_P = P_next(P, length, d, A)
            result = np.linalg.norm(P - next_P, 2)
        score_list = list(next_P)
        word_score_list = [sum(word_socre) for word_socre in score_list]
        data = heapq.nsmallest(3, enumerate(word_score_list), key=lambda x: x[1])
        topic_word = []
        for word in data:
            topic_word.append(word_list[list(word)[0]])
    else:
        topic_word = None
    return topic_word,sent




if __name__ == '__main__':
    path = 'defence\\train_tweets\\10000_pos_tweets.txt'
    model_path = 'new_100_total_tweets_word2vec.model'
    model = word2vec.Word2Vec.load(model_path)
    # sent = 'USArmy Google offer new program Google Voice active_duty military learn check blog post'
    # topic_word = w2f(0.55, sent, model, 0.85)
    with open(path,'r') as file:
        for sent in file.readlines()[0:100]:
            sent = sent.split(':')[-1]
            print(w2f(0.55, sent, model, 0.85))
#-*- coding:utf-8 -*-
#实现输入一个关键词，返回一个是否与关键词有关的推文分类的SVM的model
from svm.program.Pre_preparation import *
from svm.program.stemmer import *
from svm.program.log import Logger
from svm.program.new_feature_compute import *

import sys
import numpy as np
import functools,time,random,gensim
from gensim.models import word2vec
from scipy.stats import chi2_contingency
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

def half_insert_sort(R):
    num = len(R)
    for i in range(2, num):
        R[0] = R[i]
        if R[i-1] <= R[i]:
            R[i] = R[0]
        else:
            # 找到位置ｍ插入 R[i]
            low = 0
            high = i - 1
            while low <= high:
                m = int((high + low)/2)
                if R[m] > R[0]:
                    high = m - 1
                else:
                    low = m + 1
            R[m+1:i+1] = R[m:i]  #R[m:i]记录后移
            R[m] = R[0]  # insert R[i] on the record m
    return R

def itv2time(iItv):
    iItv = int(iItv)
    if iItv >= 3600:
        h = int(iItv / 3600)
    else:
        h = 0
    sUp_h = iItv - 3600 * h
    if iItv > 60:
        m = int(sUp_h / 60)
    else:
        m = 0
    sUp_m = sUp_h - 60 * m
    s = sUp_m
    return ":".join(map(str, (h, m, s)))

def metric(fn):
    @functools.wraps(fn)
    def wrapper(*args):
        startTime = time.time()
        result = fn(*args)
        useTime = time.time() - startTime
        print('%s executed in %s' % (fn.__name__, itv2time(useTime)))
        return result
    return wrapper

def model_predict(model_name, feature_vecs):
    """
    #将训练好的模型用于新句子，进行测试
    :param model_name: 训练好的模型名字
    :param feature_vecs:用于预测的向量，列表形式
    :return: void
    """
    # 加载本地模型
    clf = joblib.load(model_name)
    # 预测分类结果
    return clf.predict(feature_vecs)

class w2c_tweet_Keyword_SVM:
    def __init__(self,keyword,down_threshold,up_threshold):
        self.keyword = keyword  #关键词
        self.model = gensim.models.KeyedVectors.load('100_word2vec.model')  #训练好的word2vec模型名字
        self.corpus_collect = link_set('Group_Account',keyword + '_tweets')  #训练word2vec模型的所有推文数据库
        self.local_collect = local_link_set('interest',keyword + '_word_analysis')  #本地存储卡方检验数据的数据库
        self.sent_keyword_num = 3  #每条推文根据卡方检验找的关键词的个数
        self.tweet_dim = 100  #Word2vec模型的向量维数
        self.down_threshold = down_threshold  #过滤负例推文的打分阈值
        self.up_threshold = up_threshold  #过滤正例推文的打分阈值
        self.tweets_num = 10000  #过滤得到的每种推文的数量
        self.features_num = 5000  #用于训练SVM的特征数量
        self.sentname = 'text'  #语料的数据库中每条数据的推文的字典的key名称
        self.local_db = 'interest'  #本地数据库的名字
        self.path = './' + keyword   #存放该兴趣的相关文件的路径
        self.keyword_simword_num = 1000  #关键词的相似词的个数，用于寻找训练集

    """下面都是调用函数"""
    #  寻找正负例时候用于计算推文分数
    def score_count(self, model, word_list, simliary_keyword_list, simliary_keyword_dict):
        score = 0
        length = len(word_list)
        for word in word_list:
            try:
                # word_score = 0
                # count = 0
                # sim_words_list = model.most_similar(word, topn = 20)
                # for item in sim_words_list:
                #     item = list(item)
                #     if item[0] in simliary_keyword_list:
                #         word_score += item[1]*simliary_keyword_dict[item[0]]
                #         count += 1
                # word_score = word_score / 20

                word_scores = []
                for topic_word in simliary_keyword_list:
                    cos = round(model.similarity(word, topic_word) * simliary_keyword_dict[topic_word],3)
                    word_scores.append(cos)
                word_score = max(word_scores)
                # 这是之前的计算分数方法

                score += word_score
            except:
                length -= 1
                continue
        if score != 0:
            score = round(score / length,3)

        return score
    # 这里的存储向量是一条向量的存储，并且形式是list
    def save_vector(self, vec, file_path):
        with open(file_path, 'a') as f:
            vec_string = ''
            if type(vec) == list:
                for item in vec:
                    word = str(item) + " "
                    vec_string += word
                vec_string += '\n'
            else:
                pass
            f.write(vec_string)
    # 便于其他模块调用的，输入句子与flag，输出向量
    def kf_feature_compute(self, flag, sent):
        #  注意输入的是一个未处理的句子
        #  flag为数据库中的每个word统计的key值，即：正例：pos_tweets;负例：neg_tweets
        if 'pos' in flag:
            oppo_flag = flag.replace('pos','neg')
        else:
            oppo_flag = flag.replace('neg','pos')
        sent_feature = [0] * self.tweet_dim
        word_kf_scores = {}
        class_tweets_num = self.local_collect.find_one({'word': 'class_tweets_num'})
        sent = tokenize_and_stem(sent)
        for word in sent.split():
            if len(word) > 2:
                word_analysis = self.local_collect.find_one({'word': word})
                if word_analysis:
                    x = [word_analysis[flag], word_analysis[oppo_flag]]
                    y = [class_tweets_num[flag] - word_analysis[flag],
                         class_tweets_num[oppo_flag] - word_analysis[oppo_flag]]
                    X = np.array([x, y])
                    result = list(chi2_contingency(X))[0]
                    if result > 3.84:
                        word_kf_scores[word] = round(result, 5)
                    else:
                        continue
                else:
                    continue
            else:
                continue
        if len(word_kf_scores) >= 3:
            word_kf_scores_dict = dict(sorted(zip(word_kf_scores.values(), word_kf_scores.keys()), reverse=True))
            # 字典的键值对会翻转过来
            word_list = list(word_kf_scores_dict.values())[:3]
            sent_feature = self.w2c_feature(word_list)
            if sent_feature != [0] * self.tweet_dim:
                print('text:', sent, 'word_list:', word_list)
            # print(sent, ':', flag, ':', word_kf_scores_dict, ':', word_list)
        else:
            pass
        return sent_feature
    #  直接用最相似的词来做为判断看看
    def sim_word_vec(self,sent,similary_keyword_list,similary_keyword_dict):
        scores = {}
        for word in set(sent.split()):
            scores[word] = 0
            for sim_word in similary_keyword_list:
                try:
                    scores[word] += self.model.similarity(word, sim_word) * similary_keyword_dict[sim_word]
                except:
                    continue
        scores = dict(sorted(zip(scores.values(), scores.keys()), reverse=True))
        words = list(scores.values())[:3]
        print(words)
        sent_feature = self.w2c_feature(words,len(words))
        return sent_feature
    #  用于预测的推文得到向量
    def predict_kf_feature_compute(self, flag ,sent):
        #  注意输入的是一个句子
        #  flag为数据库中的每个word统计的key值，即：正例：pos_tweets;负例：neg_tweets
        if 'pos' in flag:
            oppo_flag = flag.replace('pos', 'neg')
        else:
            oppo_flag = flag.replace('neg', 'pos')
        sent_feature = [0] * self.tweet_dim
        word_kf_scores = {}
        class_tweets_num = self.local_collect.find_one({'word': 'class_tweets_num'})
        word_list = [word for word in sent.split()]
        for word in word_list:
            if len(word) > 2:
                word_analysis = self.local_collect.find_one({'word': word})
                if word_analysis:
                    x = [word_analysis[flag], word_analysis[oppo_flag]]
                    y = [class_tweets_num[flag] - word_analysis[flag],
                         class_tweets_num[oppo_flag] - word_analysis[oppo_flag]]
                    X = np.array([x, y])
                    result = chi2_contingency(X)
                    word_kf_scores[word] = round(list(result)[0], 3)
                else:
                    continue
            else:
                continue
        if len(word_kf_scores) != 0:
            word_kf_scores_dict = dict(sorted(zip(word_kf_scores.values(), word_kf_scores.keys()), reverse=True))
            word_list = list(word_kf_scores_dict.values())[:3]
            sent_feature = self.w2c_feature(word_list)
            print('text:', sent + '\n', 'word_list:', word_list)
        else:
            word_list = []
        return sent_feature,word_list
    #  未用
    def new_feature_compute_save(self):
        #  这个是直接用推文中的每个词来得到向量来跑的,用来做对比
        traintweet_file_path = self.path + 'train_tweets\\'
        tweet_feature_file_path = self.path + 'sim_tweets_feature\\'
        tweet_feature_filepath_list = []
        mkdir(tweet_feature_file_path)
        for file_name in os.listdir(traintweet_file_path):
            with open(traintweet_file_path + file_name) as re_file:
                new_file_name = 'new_' + file_name
                new_tweet_feature_file_path = tweet_feature_file_path + new_file_name
                for sent in re_file.readlines():
                    sent_word,new_sent = w2f(0.4, sent, self.model, 0.85)
                    if sent_word:
                        sent_feature = self.w2c_feature(sent_word)
                    else:
                        sent_word = random.sample(new_sent.split(),3)
                        sent_feature = self.w2c_feature(sent_word)
                    if sent_feature != [0] * self.tweet_dim:
                        self.save_vector(sent_feature, new_tweet_feature_file_path)
                    else:continue
            tweet_feature_filepath_list.append(new_tweet_feature_file_path)
        return tweet_feature_filepath_list[1],tweet_feature_filepath_list[0]
    #  词列表转化为向量
    def w2c_feature(self,sent_word):
        sent_feature = [0] * self.tweet_dim
        for word in sent_word:
            try:
                sent_feature = [a + b for a, b in zip(self.model[word], sent_feature)]
            except:
                sent_feature = [a + b for a, b in zip([0] * self.tweet_dim, sent_feature)]
        if sent_feature != [0] * self.tweet_dim:
            sent_feature = [round(num, 5) for num in sent_feature]
        return sent_feature
    #  读取特征向量文件
    def read_vec_file(self,filepath,m,n):
        """
        :param filename: 文件名，保存的是用于训练的特征向量
        :return: 特征向量的list
        """
        feature_vec = []
        with open(filepath, 'r') as f:
            for line in f.readlines()[m:n]:
                vec = [float(num) for num in line.split()]
                feature_vec.append(vec)
        return feature_vec
    #  看svm训练效果
    def svm_train(self,train_vec, pos_num, neg_num, test_size, name):
        """
        用特征向量训练分类器,并使用十次交叉验证，查看准确率
        :param train_vec: 用于训练的特征向量，包含正例和负例两部分， 前一部分为正例，后一部分为负例
        :param pos_num: 正例的数量，用于构建标签
        :param neg_num: 负例的数量， 用于构建标签
        :param test_size: 按testSize比例将训练集随机分为两部分，一部分用于训练，一部分用于测试
        :return: void
        """
        label = [1] * pos_num + [-1] * neg_num
        x = np.array(train_vec)
        y = np.array(label)
        scores = []
        for i in range(10):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
            clf = svm.SVC()
            clf.fit(x_train, y_train)
            scores.append(clf.score(x_test, y_test))
        num = pos_num + neg_num
        mean_score = np.mean(scores)
        log = Logger('acurancy.log', level='debug')
        log.logger.info('Interest:%s' % name)
        log.logger.info('num:%d' % num)
        log.logger.info('scores:%s' % scores)
        log.logger.info('mean_scores:%f' % mean_score)
        return mean_score
    #  svm训练并且保存成模型
    def svm_train_dump(self,train_x, pos_num, neg_num, model_name):
        """
        用特征向量训练分类器, 并将模型本地化
        :param train_x: 用于训练的特征向量，包含正例和负例两部分， 前一部分为正例，后一部分为负例
        :param pos_num: 正例的数量，用于构建标签
        :param neg_num: 负例的数量， 用于构建标签
        :return: void
        """
        train_y = [1] * pos_num + [0] * neg_num
        clf = svm.SVC()
        clf.fit(train_x, train_y)
        joblib.dump(clf, model_name)
    #  传入路径，对路径中文档中词做卡方统计的词频统计,可添加新文本模式
    def word_in_tweets_analysis(self,path):
        #  路径中文件的名字：正例：pos_tweets.txt;负例：neg_tweets.txt
        """
        path:文档所在的文件夹路径
        word_analy_collect:词分析结果存入的数据库
        """
        word_analy_collect = self.local_collect
        total_tweets = {}
        key_list = []
        dirs = os.listdir(path)
        for filename in dirs:
            key = filename.replace('.txt', '')
            key_list.append(key)
            filepath = path + '/' + filename
            with open(filepath,'r') as file:
                total_tweets[key] = len(file.readlines())
        for key in key_list:#更新
            word_analy_collect.update({'word':'class_tweets_num'},{'$inc':{key:total_tweets[key]}},upsert=True)
        for filename in dirs:
            file_name = path + '/' + filename
            flag_filename = filename.replace('.txt','')
            if 'pos' in filename:
                oppo_filename = flag_filename.replace('pos','neg')
            else:oppo_filename = flag_filename.replace('neg','pos')
            with open(file_name,'r') as tweets_file:
                for line in tweets_file.readlines():
                    words = [word for word in line.split()]
                    for word in set(words):
                        word_analy_collect.update({'word':word},{'$inc':{flag_filename:1,oppo_filename:0}},upsert=True)


    """下面都是功能函数"""
    #  寻找正负例训练集,
    def find_pos_and_neg_tweets(self):
        pos_id_score_dict = {}  #存放正例推文以及分数
        neg_id_score_dict = {}
        pos_train_tweets_num = 0
        neg_train_tweets_num = 0
        neg_tweets_num = 0
        pos_tweets_num = 0
        simliary_keyword_list = self.model.most_similar(self.keyword, topn=self.keyword_simword_num)
        similary_keyword_dict = {}
        for item in simliary_keyword_list:
            item = list(item)
            similary_keyword_dict[item[0]] = item[1]
        simliary_keyword_list = [list(word)[0] for word in simliary_keyword_list]
        simliary_keyword_list.append(self.keyword)
        similary_keyword_dict[self.keyword] = 1.0
        thre = int(0.7 * self.tweets_num)
        print('与关键词相似的词列表：', simliary_keyword_list)
        stop = input('是否要继续(y/n):')
        if stop == 'y':
            path = self.path + '/train_tweets'
            mkdir(path)
            posttweet_file_path = path + '/pos_tweets.txt'
            negtweet_file_path = posttweet_file_path.replace('pos', 'neg')
            pos_tweets_collect = local_link_set(self.local_db, self.keyword + '_pos_tweets')
            # pos_tweets_analysis_collect = local_link_set(self.local_db,self.keyword + '_pos_tweets_word_analysis')
            neg_tweets_collect = local_link_set(self.local_db, self.keyword + '_neg_tweets')
            neg_tweets_collect.update({'word': 'tweets_num'}, {'$set': {'neg_tweets_num': 0}},
                                      upsert=True)
            pos_tweets_collect.update({'word': 'tweets_num'}, {'$set': {'pos_tweets_num': 0}},
                                      upsert=True)
            self.local_collect.update({'word': 'class_tweets_num'},
                                      {'$set': {'neg_tweets': 0, 'pos_tweets': 0}},
                                      upsert=True)
            with open(posttweet_file_path, 'a', encoding='utf-8') as pos_file:
                with open(negtweet_file_path, 'a', encoding='utf-8') as neg_file:
                    corpus = self.corpus_collect.find(no_cursor_timeout=True)
                    for item in corpus:
                        sent = item[self.sentname]
                        text = tokenize_and_stem(sent)
                        word_list = [word for word in text.split()]
                        length = len(set(word_list))
                        if length >= self.sent_keyword_num:
                            score = self.score_count(self.model, word_list, simliary_keyword_list,
                                                     similary_keyword_dict)
                            if 0 < score <= self.down_threshold and not neg_tweets_collect.find_one({self.sentname: sent}) and neg_train_tweets_num <= self.tweets_num * 2:  # 防止有重复的推文
                                neg_tweets_num += 1
                                neg_tweets_collect.insert_one({'id':neg_tweets_num,self.sentname: sent,'score':score})
                                print('find %dth neg tweet' % neg_tweets_num)
                                if len(word_list) > 7:
                                    neg_train_tweets_num += 1
                                    neg_id_score_dict[neg_tweets_num] = score
                                    for word in set(word_list):
                                        if len(word) > 2:
                                            self.local_collect.update({'word': word},
                                                                      {'$inc': {'neg_tweets': 1, 'pos_tweets': 0}},
                                                                      upsert=True)
                                        else:
                                            continue
                                    if neg_train_tweets_num <= thre:
                                        continue
                                    else:
                                        neg_id_score_dict = dict(
                                            sorted(neg_id_score_dict.items(), key=lambda item: item[1], reverse=False))
                                        neg_id_score_dict.popitem()
                                else:
                                    continue
                            elif score >= self.up_threshold and not pos_tweets_collect.find_one({self.sentname: sent}) and pos_train_tweets_num <= self.tweets_num * 2:
                                pos_tweets_num += 1
                                pos_tweets_collect.insert_one(
                                    {'id':pos_tweets_num, self.sentname: sent, 'score': score})  # 下面要是调用的话，这里不能用异步IO
                                print('find %dth pos tweet' % pos_tweets_num)
                                if len(word_list) > 7:
                                    pos_train_tweets_num += 1
                                    pos_id_score_dict[pos_tweets_num] = score
                                    for word in set(word_list):
                                        if len(word) > 2:
                                            self.local_collect.update({'word': word},
                                                                      {'$inc': {'neg_tweets': 0, 'pos_tweets': 1}},
                                                                      upsert=True)
                                        else:
                                            continue
                                    if pos_train_tweets_num <= thre:
                                        continue
                                    else:
                                        pos_id_score_dict = dict(
                                            sorted(pos_id_score_dict.items(), key=lambda item: item[1], reverse=True))
                                        pos_id_score_dict.popitem()
                                else:
                                    continue
                            elif neg_train_tweets_num >= self.tweets_num * 2 and pos_train_tweets_num >= self.tweets_num * 2:
                                    break
                            else:
                                continue
                        else:
                            continue
                    corpus.close()
                    neg_tweets_collect.update({'word': 'tweets_num'}, {'$set': {'neg_tweets_num': neg_tweets_num}},
                                              upsert=True)
                    pos_tweets_collect.update({'word': 'tweets_num'}, {'$set': {'pos_tweets_num': pos_tweets_num}},
                                              upsert=True)
                    self.local_collect.update({'word': 'class_tweets_num'},
                                              {'$set': {'neg_tweets': neg_train_tweets_num,
                                                        'pos_tweets': pos_train_tweets_num}},
                                              upsert=True)
                    print('Processing train tweets......')
                    for tweet_id, score in pos_id_score_dict.items():
                        text = tokenize_and_stem(pos_tweets_collect.find_one({'id': tweet_id})[self.sentname])
                        pos_file.write(text + '\n')
                    for tweet_id, score in neg_id_score_dict.items():
                        text = tokenize_and_stem(neg_tweets_collect.find_one({'id': tweet_id})[self.sentname])
                        neg_file.write(text + '\n')
                print('Process train tweets done!')
            return path  # 可以用于其他模块
        else:
            return None
    #  未用
    def sent_by_sent_word_analysis(self,sent,file_name,collect):
        # 这是在得到正例推文的同时，将一条推文中的词做分析
        """
        没有用
        sent:推文
        file_name:此推文将存入的文档名字
        collect:存入词分析的数据库
        """
        file_name = file_name.replace('.txt','')
        if 'pos' in file_name:
            posfile_name = file_name.replace('pos','neg')
        else:
            posfile_name = file_name.replace('neg','pos')
        collect.update({'word':'class_tweets_num'},{'$inc':{file_name:1}},True)
        word_list = []
        for word in sent.split():
            if word not in word_list:
                collect.update({'word': 'total_words'}, {'$addToSet': {'total_words': word}},True)
                collect.update({'word':word}, {'$inc': {file_name: 1, posfile_name: 0}},True)
                word_list.append(word)
            else:continue
#  直接对已经找好的正负例文本进行转化为向量，得到分类器的训练向量(5000条）
    #  对训练集进行卡方统计得到向量
    def kf_feature_compute_save(self):
        traintweet_file_path = self.path + '/train_tweets'
        tweet_feature_file_path = self.path + '/kf_tweets_feature'
        mkdir(tweet_feature_file_path)
        # 前面在找训练集的过程中就做了词频统计
        # print('Processing train tweet words......')
        # self.word_in_tweets_analysis(traintweet_file_path)
        # print('Tweet words processed done!')
        """
        similary_keyword_list = self.model.most_similar(self.keyword, topn=2000)
        similary_keyword_dict = {}
        for item in similary_keyword_list:
            item = list(item)
            similary_keyword_dict[item[0]] = item[1]
        similary_keyword_list = [list(word)[0] for word in similary_keyword_list]
        """
        for filename in os.listdir(traintweet_file_path):
            train_filepath = traintweet_file_path + '/' + filename
            feature_filepath = tweet_feature_file_path + '/' + filename
            flag = filename.replace('.txt','')
            feature_count = 0
            with open(train_filepath, 'r') as file:
                for sent in file.readlines():
                    if feature_count < self.features_num:
                        sent_feature = self.kf_feature_compute(flag, sent)
                        # sent_feature = self.sim_word_vec(sent,similary_keyword_list,similary_keyword_dict)
                        if sent_feature != [0] * self.tweet_dim:
                            self.save_vector(sent_feature, feature_filepath)
                            feature_count += 1
                        else:
                            continue
                    else:
                        break
        print('Feature process done!')
    #  集成svm的两个模块，直接训练看效果再保存
    def svm_train_and_save(self,feature_file_path):
        print("************ >>>>  Reading feature files......")
        post_feature_path = None
        neg_feature_path = None
        for filename in os.listdir(feature_file_path):
            if 'pos' in filename:
                post_feature_path = feature_file_path + '/' + filename
            else:
                neg_feature_path = feature_file_path + '/' + filename
        pos_vecs = self.read_vec_file(post_feature_path,0,self.features_num)
        x = len(pos_vecs)
        neg_vecs = self.read_vec_file(neg_feature_path,0,self.features_num)
        y = len(neg_vecs)
        pos_vecs.extend(neg_vecs)
        print("************ >>>>  SVM training......")
        mean_score = self.svm_train(pos_vecs, x, y, 0.1, self.keyword)
        print('mean_score:%f%%'%mean_score*100)
        if mean_score >= 0.9:
            print("************ >>>>  SVM save......")
            model_name = self.keyword + '.model'
            self.svm_train_dump(pos_vecs, x, y, model_name)
            print('Done!!!')
        else:
            pass


if __name__ == "__main__":
    """寻找的正负例以及词频统计没有问题，但是向量计算有些问题，不小心把正负例删除了，重跑看修改向量计算后的情况"""
    start_time_one = time.time()
    print('Starting at:',time.ctime())
    tweet_SVM = w2c_tweet_Keyword_SVM('political',0.39,0.43)
    train_tweets_filepath = tweet_SVM.find_pos_and_neg_tweets()
    if not train_tweets_filepath:
        sys.exit(0)
    # start_time_two = time.time()
    tweet_SVM.kf_feature_compute_save()
    tweets_feature_file = './' + tweet_SVM.keyword + '/kf_tweets_feature'
    tweet_SVM.svm_train_and_save(tweets_feature_file)
    end_time = time.time()
    print('Stop at:',time.ctime())
    print('Total use time(hour shou reduce 8):', time.strftime('%H:%M:%S', time.localtime(end_time - start_time_one)))
    # print('After finding train_tweets use time(hour shou reduce 8):', time.strftime('%H:%M:%S', time.localtime(end_time - start_time_two)))
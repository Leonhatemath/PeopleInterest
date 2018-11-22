"""
输入：账号名称（screen_name)，要测试的兴趣名称（defence,news,political）
输出：账号的兴趣分析结果
"""
from svm.program.LatestSVM import *
from svm.program.stemmer import *
import warnings

def sum(basic,L):
    if not L:
        return basic
    else:
        basic += L[0]
        return sum(basic,L[1:])

def detect_interest(remote_db,screen_name,interest):
    """
    :param remote_db: 测试数据集
    :param screen_name: 测试数据集人名
    :param interest: 测试兴趣
    :return: 更新测试结果
    """
    tweet_SVM = w2c_tweet_Keyword_SVM(interest,0.40,0.46)
    collect_name = screen_name + '_' + interest + '_result'
    interest_collect = local_link_set(interest + '_analysis',collect_name)#测试结果存放数据集
    tweet_num = remote_db.find({"user.screen_name": screen_name}).count()
    if tweet_num == 0:
        print('Can not find screen_name!')
        return
    interest_collect.insert_one({'screen_name': screen_name, 'interest': interest,'total_tweet_num':tweet_num})
    """
    simliary_keyword_list = model.most_similar(interest, topn = 2000)
    similary_keyword_dict = {}
    for item in simliary_keyword_list:
        item = list(item)
        similary_keyword_dict[item[0]] = item[1]
    simliary_keyword_list = [list(word)[0] for word in simliary_keyword_list]
    """
    model_name = interest + '.model'
    count = 0
    score = 0
    for tweet in remote_db.find({"user.screen_name": screen_name}):
        flag = False
        text = tokenize_and_stem(tweet['text'])
        for letter in text:
            if '\u4e00' <= letter <= '\u9fa5':
                flag = True
                break
        if flag:
            continue
        sent_feature,word_list = tweet_SVM.predict_kf_feature_compute('pos_tweets',text)
        # sent_feature = tweet_SVM.sim_word_vec(flag=None,sent=text,similary_keyword_list=simliary_keyword_list,similary_keyword_dict=similary_keyword_dict)
        result = model_predict(model_name,sent_feature)
        if result == [1]:
            count += 1
        interest_collect.insert_one({'text': tweet['text'], 'result': int(result[0]), 'key_words': word_list})
    result = str(round(count/tweet_num*100,3)) + '%'
    interest_collect.update({'screen_name':screen_name},{'$set':{'pos_tweets_num':count,'result':result,'total_score':score}})

def show_result(screen_name,interest):
    collect_name = screen_name + '_' + interest + '_result'
    interest_collect = local_link_set(interest + '_analysis', collect_name)
    result = interest_collect.find_one({"interest": interest})
    print('账号名称：',result['screen_name'])
    print('账号推文数量：',result['total_tweet_num'])
    print('正例推文数量:',result['pos_tweets_num'])
    print('正例推文比例：',result['result'])
    print('账号兴趣得分：',result['total_score'])
    # for tweet in interest_collect.find({'result': 1}):
    #     print(tweet['text'])
    int_result = result['result'].strip('%')
    if float(int_result) > 40 and result['total_tweet_num'] > 50:
        print('判定',result['screen_name'],'对',interest,'感兴趣')
    elif float(int_result) < 15 or (15.0 < float(int_result) < 20.0 and result['total_tweet_num'] <= 200):
        print('判定',result['screen_name'],'对',interest,'不感兴趣')
    else:
        print('判定',result['screen_name'],'对',interest,'兴趣不浓厚')
        stop = input('是否显示判为相关推文？(y/n)：')
        if stop == 'y':
            sort_score = {}
            for tweet in interest_collect.find({'result':1}):
                sort_score[tweet['_id']] = tweet['tweet_score']
            sort_score = dict(sorted(sort_score.items(), key=lambda sort_score:sort_score[1], reverse=True))
            command = 1
            n = 0
            count = 10
            while command != 'n':
                if command == 1:
                    n += 1
                    if n > len(sort_score) // count + 1:
                        n = len(sort_score) // count + 1
                elif command == 2:
                    n -= 1
                    if n < 1:
                        n = 1
                else:pass
                # for id in list(sort_score)[(n-1)*count:n*count]:
                #     text = interest_collect.find_one({'_id':id})['text']
                #     print('text:',text)
                command = input("Up:1;Down:2;Exit:n")
    print('\n')

"""
if __name__ == '__main__':
    for screen_name in ["UB_Economics", "USCG", "Military_Edge", "NationalGuard", "insidedefense", "northropgrumman"]:
        show_result(screen_name,'military')
"""

if __name__ == '__main__':
    """
    remote_db:测试数据集
    
    """
    warnings.filterwarnings("ignore")
    # corpus_db = 'new_tweet'
    corpus_collect = 'tutor_tweets'
    # corpus_collect = 'jiangdu_tweets'
    corpus_db = 'test'
    # corpus_collect = 'tweet_user'
    remote_db = local_link_set(corpus_db,corpus_collect)
    # remote_db = link_set(corpus_db, corpus_collect)
    # screen_name = input('please input screen_name:')
    for interest in ['political']:
        # for screen_name in ["mengley1"]:
        for screen_name in ["AndrewYNg", "prfsanjeevarora", "Reza_Zadeh", "5harad", "benjraphael", "DaphneKoller","ermonste" ]:
            detect_interest(remote_db, screen_name, interest)
            show_result(screen_name, interest)
            # interest_list = ['military']
            # print('here are the interests we can detect:',interest_list)
            # test_intertest = input('please input interest you want to detect(use\',\'for split) or \'all\':').split(',')
            # test_intertest = ['military']
            # if test_intertest == ['all']:
            #     for interest in interest_list:
            #         detect_interest(remote_db, screen_name, interest)
            # else:
            #     have_no_interest = [interest for interest in test_intertest if not local_link_set(local_db_name, interest)]
            #     if len(have_no_interest) > 0:
            #         print('can not detect these interests:', have_no_interest)
            #         test_intertest = [x for x in test_intertest if x not in have_no_interest]
            #     if len(test_intertest) > 0:
            #         for interest in test_intertest:
            #             detect_interest(remote_db, screen_name, interest)
            #     else:
            #         print('sorry can not detect all of these interests you input')

from pymongo import *
import logging,os,motor.motor_asyncio
from gensim.models import word2vec
from svm.program.stemmer import *

def motor_link_db(db_name, collect_name):
    client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://kb314:fzdwxxcl.314@121.49.99.14:30011')
    db = client[db_name]
    collect = db[collect_name]
    return collect

def motor_local_link_db(db_name, collect_name):
    client = motor.motor_asyncio.AsyncIOMotorClient('localhost', 27017)
    db = client[db_name]
    collect = db[collect_name]
    return collect

def link_set(db_name, collect_name):
    client = MongoClient('mongodb://readAnyDatabase:Fzdwxxcl.121@121.49.99.14:30011')
    db = client.get_database(db_name)
    collect = db.get_collection(collect_name)
    return collect

def local_link_set(db_name,collect_name):
    client = MongoClient('localhost', 27017)
    db = client.get_database(db_name)
    collect = db.get_collection(collect_name)
    return collect


def w2vmodel_train(tweetname,filename):
    """训练模型"""
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(filename)
    model = word2vec.Word2Vec(sentences, size=100)
    modelname = tweetname + '.model'
    model.save(modelname)
    return(modelname)

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False

def find_index(list,value):
    high_index = 0
    low_index = len(list) - 1
    while high_index < low_index:
        mid = (low_index + high_index) // 2
        if value <= list[mid]:
            high_index = mid + 1
        else:
            low_index = mid
    return (high_index)

def sort(length, sentence_score, id_list, id, score):
        # 测试，取top6000
        if length == 0:
            sentence_score.append(score)
            id_list.append(id)
            length += 1
        elif length <= 6000:
            if score >= sentence_score[0]:
                sentence_score.insert(0, score)
                id_list.insert(0, id)
                length += 1
            elif score <= sentence_score[-1]:
                sentence_score.append(score)
                id_list.append(id)
                length += 1
            else:
                index = find_index(sentence_score, score)
                sentence_score.insert(index, score)
                id_list.insert(index, id)
                length += 1
        else:
            if score <= sentence_score[-1]:
                pass
            elif score >= sentence_score[0]:
                sentence_score.pop()
                sentence_score.insert(0, score)
                id_list.pop()
                id_list.insert(0, id)
            else:
                index = find_index(sentence_score, score)
                sentence_score.insert(index, score)
                id_list.insert(index, id)
                sentence_score.pop()
                id_list.pop()

        return (length, sentence_score, id_list)



if __name__ == '__main__':
    #  读取数据库的原始文本并训练word2vec模型
    with open('jiangdu.txt', 'a') as f:
        for i in ['jiangdu_tweets']:
            collect = link_set('Group_Account', i)
            corpus = collect.find(no_cursor_timeout=True)
            for item in corpus:
                flag = True
                sent = item['text']
                for letter in sent:
                    if '\u4e00' <= letter <= '\u9fa5':
                        flag = False
                        break
                if flag:
                    try:
                        text = tokenize_and_stem(sent)
                        if len(text.split()) >= 3:
                            f.write(text + '\n')
                        else:
                            continue
                    except:
                        continue
            corpus.close()
    print("Done!")
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # sentences = word2vec.Text8Corpus('corpus.txt')
    # model = word2vec.Word2Vec(sentences, size=100,sg=1)
    # model_name = 'new_' + str(100) + '_word2vec.model'
    # model.save(model_name)
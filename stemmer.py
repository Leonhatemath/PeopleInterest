#-*- coding:utf-8 -*-\
import nltk
import re

from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return res

from nltk.stem.snowball import SnowballStemmer
stemmer  = SnowballStemmer("english")
def tokenize_and_stem(text):
    text = re.sub(r'(\s[^\s]+\.\.\.)|(\s[^\s]+\…)', '', text) # 去掉以...和…结尾的单词
    text = re.sub(r'(\'[^\s]+)|(\’[^\s]+)', '', text) # 去掉所有格与助词缩写
    text = re.sub(r'(RT|([^\w]+BREAKING:)|(www\.[^\s]+))|(https[^\s]+)|(http[^\s]+)', '', text)  # 把链接,RT,BREAKING,等东西去掉
    text = re.sub(r'@','USR_',text)
    # 替换A.B A.B. A.B.C等形式的单词为AB，AB，ABC
    fixtext = re.findall(r'(([A-Z]\.[A-Z]\.?)+([A-Z]?))', text)
    for word in fixtext:
        match = word[0]
        replacetext = ' ' + word[0].replace('.', '') + ' '
        text = re.sub(match, replacetext, text)
    # 找出文本中的基本和扩展拉丁字母，数字和基本的ASCII符号,只保留了-,_
    text = re.findall(r'[\u002D\u0030-\u0039\u0041-\u005A\u005F\u0061-\u007A]+', text)
    text = ' '.join(text)
    # 把首字母大写多词连写拆开
    text_exchange = re.findall(r'[A-Z]+[\S]+[A-Z]+[\S]+', text)
    for word in text_exchange:
        if word[:3] == 'USR':
            continue
        else:
            fixtext = re.findall(r'[A-Z]+[\S]+', word)  # 这句放在最后是因为可以保证句中就只有-,%,$这样的符号了
            replace_text = ' '.join(map(str, fixtext))
            text = re.sub(word, replace_text, text)
    text_exchange = re.findall(r'[A-Z][\S]+', text)
    for word in text_exchange:
        if word[:3] == 'USR':
            continue
        else:
            replace_word = str.lower(word[0]) + word[1:]
            text = re.sub(word, replace_word, text)
    #分词,注意标点符号也会成为一个词
    #tokens = [word for word in nltk.word_tokenize(text)]
    #stems = [stemmer.stem(t) for t in tokens] #词干化
    for word in text.split():
        if re.search(r'-+\S+|\S+-+', word):
            new_word = word.strip('-')
            text = re.sub(word, new_word, text)

    #写一个词形还原来对比
    stems = lemmatize_sentence(text)#函数里面有分词效果

    # #加载停用词
    stopwords = nltk.corpus.stopwords.words('english')
    tokens_filter = [word for word in stems if word and word.lower() not in stopwords]  # 去停等词(修改了一下stopwords文档）
    a = " ".join(tokens_filter)
    return (a)

if __name__ == "__main__":
    # from svm.extraction.w2c import link_set
    # db_name = 'Group_Account'
    # collect_name = 'defence_tweets'
    # collect = link_set(db_name, collect_name)
    # num = 0
    # for item in collect.find()[0:100]:
    #     print(tokenize_and_stem(item['text']))
    #     num += 1
    #     print('num:',num)
    text = "Win 4 10% #$% %% -Free's a @Tickets to A.B Big/Health. Expo on the Queen Mary! http://t.co/cCMjkdZC7D http://t.co/zsZme2R3Wb"
    print(tokenize_and_stem(text))

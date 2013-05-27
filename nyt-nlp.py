# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#import requests
import json
import pandas as pd
import numpy as np
from time import sleep
from itertools import count
#import cPickle as pickle
#from lxml.cssselect import CSSSelector
#from lxml.html import fromstring
#import redis
import pymongo
import re
from operator import itemgetter
from gensim import corpora, models, similarities
import gensim
from collections import Counter
import datetime as dt

# <codecell>

import myutils as mu
mu.psettings(pd)

# <codecell>

%load_ext autosave
%autosave 30

# <markdowncell>

# ##Init

# <codecell>

connection = pymongo.Connection( "localhost", 27017 )
db = connection.nyt

# <codecell>

raw = list(db.raw_text.find())

# <codecell>

for dct in raw:
    if 'title' not in dct:
        dct['title'] = ''

# <codecell>

filter(lambda x: 'text' not in x, db.raw_text.find())

# <codecell>

filter(lambda x: 'title' not in x, raw)

# <rawcell>

# _txt = raw[0]['text']

# <rawcell>

# def catch():
#     for t in map(itemgetter('text'), raw):
#         for w in format(t):
#             if w.startswith("'"):  
#                 print w
#                 return t
# _t = catch()
# _t

# <rawcell>

# c = Counter(txt)

# <markdowncell>

# ##Extraction

# <codecell>

def format(txt):
    tt = re.sub(r'[\.\,\;\:\'\"\(\)\&\%\*\+\[\]\=\?\!/]', '', txt).lower()
    tt = re.sub(r' *\$[0-9]\S* ?', ' <money> ', tt)    
    tt = re.sub(r' *[0-9]\S* ?', ' <num> ', tt)    
    tt = re.sub(r'[\-\s]+', ' ', tt)
    return tt.strip().split()

# <rawcell>

# ' '.join(dols)
# #txt = format(_txt)
# #print txt[:10]
# #t = format(_t)
# #sorted(t)
# #format(' '.join(dols))

# <codecell>

map(itemgetter('url'), raw)

# <codecell>

len(raw)

# <codecell>

texts = [format(doc['text']) for doc in raw]
#opinions = [format(doc['text']) for doc in raw if '/opinion/' in doc['url']]
#articles = [format(doc['text']) for doc in raw if '/opinion/' not in doc['url']]

# <rawcell>

# articles[0]

# <codecell>

dmap = lambda dct, a: [dct[e] for e in a]

# <codecell>

dictionary = corpora.Dictionary(texts)
#odictionary = corpora.Dictionary(opinions)
#adictionary = corpora.Dictionary(articles)

# <rawcell>

# dols = sorted(dictionary.token2id)[-20:]
# dols

# <codecell>

corpus = [dictionary.doc2bow(text) for text in texts]
#ocorpus = [odictionary.doc2bow(text) for text in opinions]
#acorpus = [adictionary.doc2bow(text) for text in articles]

# <codecell>

tfidf = models.TfidfModel(corpus)
#otfidf = models.TfidfModel(ocorpus)
#atfidf = models.TfidfModel(acorpus)

# <rawcell>

# #Common words, count vs tfidf
# sorted([(dictionary[w], ct) for w, ct in tfidf[corpus[0]]], key=itemgetter(1), reverse=1)
# sorted([(dictionary[w], ct) for w, ct in corpus[0]], key=itemgetter(1), reverse=1)

# <codecell>

tcorpus = dmap(tfidf, corpus)
#otcorpus = dmap(otfidf, ocorpus)
#atcorpus = dmap(atfidf, acorpus)

# <codecell>

lda = gensim.models.ldamodel.LdaModel(corpus=tcorpus, id2word=dictionary, num_topics=15, update_every=0, passes=20)

# <rawcell>

# olda = gensim.models.ldamodel.LdaModel(corpus=otcorpus, id2word=odictionary, num_topics=15, update_every=0, passes=20)

# <rawcell>

# alda = gensim.models.ldamodel.LdaModel(corpus=atcorpus, id2word=adictionary, num_topics=15, update_every=0, passes=20)

# <codecell>

np.random.randint(len(raw), size=3)# (raw, replace=0)

# <codecell>


# <codecell>

d = raw[0]['date']
print d
datef = lambda d: dt.datetime.strptime(d, '%Y%m%d').strftime('%b %Y')

# <codecell>

all(map(itemgetter('text'), raw))

# <codecell>

topic_data = []
randi = np.random.randint(len(raw), size=3)
for mod in (lda,):# olda, alda:
    topic_list = [[w for _, w in tups] for tups in mod.show_topics(formatted=0, topn=15, topics=None)]
    for tit, _text, date in map(itemgetter(u'title', 'text', 'date'), (raw[i] for i in randi)):
        text = format(_text)
        #date = datef(_date)
        #date = dt.datetime.strptime(_date, '%Y%m%d')
 #       print date, '--', tit
        
         # [format(doc['text']) for doc in raw]
        #print mod[tcorpus[0]]
    #    print ' '.join(texts[0])
        _srtd = sorted(mod[dictionary.doc2bow(text)], key=itemgetter(1), reverse=1)[:2]
        top, score = _srtd[-1]
        topic_data.append((tit, date, top, score))
#        print top, score
       # continue
        print tit, date
        for top, score in sorted(mod[dictionary.doc2bow(text)], key=itemgetter(1), reverse=1):
            #if top == 11: continue
            print top, '%.2f' % score, ', '.join(topic_list[top])
        print

# <codecell>

_df = pd.DataFrame(topic_data, columns=['Title', 'Date', 'Topic', 'Score'])
df = _df.set_index('Date').sort_index()#.head()

# <codecell>

df

# <codecell>

df.Topic.hist()

# <codecell>

for mod in lda, olda, alda:
    #print mod[tcorpus[0]]
#    print ' '.join(texts[0])
    topic_list = [[w for _, w in tups] for tups in mod.show_topics(formatted=0, topn=15, topics=None)]
    for top, score in sorted(mod[tcorpus[0]], key=itemgetter(1), reverse=1):
        print top, ', '.join(topic_list[top])
    print

# <codecell>

for mod in lda, olda, alda:
    #print mod[tcorpus[0]]
#    print ' '.join(texts[0])
    topic_list = [[w for _, w in tups] for tups in mod.show_topics(formatted=0, topn=15, topics=None)]
    for top, score in sorted(mod[corpus[0]], key=itemgetter(1), reverse=1):
        print top, ', '.join(topic_list[top])
    print

# <codecell>

[topic_list[i] for i in (2, 3, 4, 11)]

# <codecell>

hdp = gensim.models.hdpmodel.HdpModel(corpus=tcorpus, id2word=dictionary, outputdir=)

# <codecell>

hdp.outputdir = '/Users/beard/Dropbox/Engineering/data/nyt-nlp'

# <codecell>

_p = hdp.print_topics(topics=20, topn=10)
!ls

# <codecell>


# <codecell>

#http://comments.gmane.org/gmane.comp.ai.gensim/1572
alpha, beta = hdp.hdp_to_lda()
hda = gensim.models.LdaModel(id2word=hdp.id2word, num_topics=len(alpha), alpha=alpha, eta=hdp.m_eta)
hda.expElogbeta = numpy.array(beta, dtype=numpy.float32)

# <codecell>

hda.show_topics(formatted=0)

# <codecell>

models.hdpmodel?

# <codecell>

ls "/tmp"

# <codecell>

model.outputdir = '.'

# <codecell>

model.print_topics()

# <codecell>

ldh = model.hdp_to_lda()

# <codecell>

pwd

# <codecell>

model.save('hdp')

# <codecell>

lda.show_topics(formatted=0, topn=15)[0]

# <codecell>

pprint([map(itemgetter(1), tups) for tups in lda.show_topics(formatted=0, topn=15)])

# <codecell>

[map(itemgetter(1), tups) for tups in lda.show_topics(formatted=0, topn=15)]

# <codecell>

lda.numworkers

# <codecell>

lda.show_topics() #10 topics

# <codecell>

lda.show_topics()

# <codecell>

#filter(lambda x: x[1] == 1, c.items())

# <codecell>

ttl = format(tt).split()

# <codecell>

c = Counter(_.split())

# <rawcell>

# list(c.keys())

# <codecell>

stopwords = set('to as at a is its in on and that of the'.split())
stopwords

# <codecell>

print sorted(filter(lambda w: w not in stopwords, ttl))


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

_txt = raw[0]['text']

# <codecell>

def catch():
    for t in map(itemgetter('text'), raw):
        for w in format(t):
            if w.startswith("'"):  
                print w
                return t
_t = catch()
_t

# <rawcell>

# c = Counter(txt)

# <codecell>

def format(txt):
    tt = re.sub(r'[\.\,\;\:\'\"\(\)\&\%\*\+\[\]\=\?\!/]', '', txt).lower()
    tt = re.sub(r' *\$[0-9]\S* ?', ' <money> ', tt)    
    tt = re.sub(r' *[0-9]\S* ?', ' <num> ', tt)    
    tt = re.sub(r'\s+', ' ', tt)
    #tt = ' '.join(s.strip() for s in tt.lower().splitlines()).strip()
    return tt.strip().split()

txt = format(_txt)
#print txt[:10]
#t = format(_t)
#sorted(t)
format(' '.join(dols))

# <rawcell>

# ' '.join(dols)

# <codecell>

texts = [format(doc['text']) for doc in raw]

# <codecell>

dictionary = corpora.Dictionary(texts)

# <rawcell>

# dols = sorted(dictionary.token2id)[-20:]
# dols

# <codecell>

corpus = [dictionary.doc2bow(text) for text in texts]

# <codecell>

corpus[0]

# <codecell>

tfidf = models.TfidfModel(corpus)

# <codecell>

dmap = lambda dct, a: [dct[e] for e in a]

# <rawcell>

# #Common words, count vs tfidf
# sorted([(dictionary[w], ct) for w, ct in tfidf[corpus[0]]], key=itemgetter(1), reverse=1)
# sorted([(dictionary[w], ct) for w, ct in corpus[0]], key=itemgetter(1), reverse=1)

# <codecell>

tcorpus = dmap(tfidf, corpus)

# <codecell>

lda = gensim.models.ldamodel.LdaModel(corpus=tcorpus, id2word=dictionary, num_topics=5, update_every=0, passes=20)

# <codecell>

model = gensim.models.hdpmodel.HdpModel(corpus=tcorpus, id2word=dictionary)

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

lda.show_topics(formatted=0, topn=15)

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


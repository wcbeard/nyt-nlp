# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
import json
import pandas as pd
import numpy as np
from time import sleep
from itertools import count, imap, starmap, cycle
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

# <markdowncell>

# ##Extraction

# <codecell>

def format(txt):
    tt = re.sub(r"'s\b", '', txt).lower()
    tt = re.sub(r'[\.\,\;\:\'\"\(\)\&\%\*\+\[\]\=\?\!/]', '', tt)    
    tt = re.sub(r' *\$[0-9]\S* ?', ' <money> ', tt)    
    tt = re.sub(r' *[0-9]\S* ?', ' <num> ', tt)    
    tt = re.sub(r'[\-\s]+', ' ', tt)
    return tt.strip().split()

dmap = lambda dct, a: [dct[e] for e in a]

# <codecell>

texts = [format(doc['text']) for doc in raw]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
tcorpus = dmap(tfidf, corpus)

# <rawcell>

# lda = gensim.models.ldamodel.LdaModel(corpus=tcorpus, id2word=dictionary, num_topics=15, update_every=0, passes=20)

# <codecell>

model = models.lsimodel.LsiModel(corpus=tcorpus, id2word=dictionary, num_topics=6)

# <codecell>

topic_data = []
_topic_stats = []
_kwargs = dict(formatted=0, topn=15, topics=None)
_kwargs = dict(formatted=0, num_words=15)
topic_words = [[w for _, w in tups] for tups in model.show_topics(**_kwargs)]

for tit, _text, date in imap(itemgetter(u'title', 'text', 'date'), raw):
    text = format(_text)
    _srtd = sorted(model[dictionary.doc2bow(text)], key=itemgetter(1), reverse=1)#[:2]
    top, score = _srtd[:2][0]
    topic_data.append((tit, date, top, score))
    _topic_stats.append(_srtd)

topic_stats = [tup for tups in _topic_stats for tup in tups]

# <codecell>

_df = pd.DataFrame(topic_data, columns=['Title', 'Date', 'Topic', 'Score'])
df = _df#.set_index('Date').sort_index()#.head()

# <codecell>

sdf = pd.DataFrame(topic_stats, columns=['Topic', 'Score'])
#df = _df.set_index('Date').sort_index()#.head()
topic_mean = sdf.groupby('Topic').mean()['Score']
#topic_mean

# <codecell>

df.head()

# <codecell>

center = lambda top, score: score - topic_mean[top]
center_list = lambda lst: [(top, center(top, score)) for top, score in lst]
centered_scores = [max(center_list(lst), key=itemgetter(1)) for lst in _topic_stats]
second_scores = [sorted(lst, key=itemgetter(1),  reverse=1)[:2][-1] for lst in _topic_stats]

# <codecell>

df.Topic, df.Score = zip(*centered_scores)
#df.Topic, df.Score = zip(*second_scores)

# <codecell>

df = _df.set_index('Date').sort_index()#.head()

# <codecell>

df.head()

# <codecell>

plt.figsize(6, 4)
df.Topic.hist()

# <codecell>

year = lambda x: x.year
#df.reset_index().groupby(['Date', 'Topic']).size()
sz = df.groupby(['Topic', year]).size()
sz.index.names[1] = 'Year'

# <rawcell>

# sz

# <codecell>

vc = df.Topic.value_counts()
vc /= vc.sum()
vc

# <codecell>

styles = cycle(['-', '--', '-.', ':'])

# <codecell>

def plottable(k, gp, thresh=.15):
    _gp = gp.set_index('Year')[0]
    _gp = (_gp / _gp.sum())
    mx = _gp.max()    
    if mx < thresh:
        return
    return k, _gp

    
    

# <codecell>

yr_grps = filter(bool, starmap(plottable, sz.reset_index().groupby('Topic')))

# <codecell>

plt.figsize(10, 8)
cols = cycle('rgbcmyk')
_rep = int(round(len(yr_grps) / 2))
styles = cycle((['-'] * _rep) + (['--'] * _rep))
tops = []
for k, gp in yr_grps:
    gp.plot(color=cols.next(), ls=styles.next())
    print '{}, {}, {:.1f}'.format(k, gp.idxmax(), gp.max() * 100)
    tops.append(k)
    #gp.set_index('Year')[0].plot()
_ = plt.legend(tops)

# <markdowncell>

# Topics 7, 9 and 0 14 peaked in 1998, while 8 and 4 peaked a year earlier.

# <codecell>

list(itertools.combinations([1, 2, 3], 2))

# <codecell>

lst

# <codecell>

combos = [list(itertools.combinations(lst, 2)) for lst in [map(itemgetter(0), _lst) for _lst in _topic_stats] if len(lst) > 1]
combos[:3]

# <codecell>

from collections import defaultdict

# <codecell>

def _cnt(lst):
    for tup in lst:
        _cnt.dct[tuple(sorted(tup))] += 1
_cnt.dct = defaultdict(int)
_ = map(_cnt, combos)

# <codecell>

_cnt.dct

# <rawcell>

# _df = pd.DataFrame(_cnt.dct.items()).set_index(0).sort(1, ascending=0)
# _df

# <codecell>

_topic_words = [', '.join(w for w in wds) for wds in topic_words]

# <rawcell>

# t98 = 7, 9, 0, 14
# t97 = 8, 4
# for i in t97:
#     print _topic_words[i]
# print
# for i in t98:
#     print _topic_words[i]

# <codecell>

c = Counter([w for subl in topic_words for w in subl])

# <codecell>

c

# <codecell>

vc

# <codecell>

for i, wds in enumerate(_topic_words):
    print '{} ({:.1f}%): {}'.format(i, vc[i] * 100, wds) # reduce num of topics

# <codecell>

search = lambda wrd: sorted(((b['title'], b['date'].year) for b in filter(lambda x: wrd in x['text'].lower(), raw)), key=itemgetter(1))
_s = sorted(((b['title'], b['date'].year) for b in filter(lambda x: 'hastert' in x['text'].lower(), raw)), key=itemgetter(1))
for t, y in _s:
    print '{}: {}'.format(y, t)

# <codecell>

span = lambda seq: range(min(seq), max(seq) + 1)

# <codecell>

plt.figsize(12, 12)
topics = df.Topic.unique()
N = len(topics) // 2
for i, topic in enumerate(sorted(topics)):
    plt.subplot(N * 100 + 20 + i + 1)
    a = df[df.Topic == topic].index.map(lambda x: x.year)
    plt.hist(a, bins=span(a))
    plt.ylabel('Topic #{}'.format(topic))
    plt.title(tformat(topic_words[i]))
plt.tight_layout()
plt.figsize(6, 4)

# <rawcell>

# filter(lambda x: x['title'] == 'Democrat Urges Foley to Resign In Bank Scandal', raw)

# <codecell>

df[df.Title == 'Democrat Urges Foley to Resign In Bank Scandal']

# <codecell>

set(topic_words[3]) & set(topic_words[5])

# <codecell>

sorted((jaccard(topic_words[i], topic_words[j]), i, j) for i, j in itertools.combinations(range(len(topic_words)), 2))[::-1]

# <codecell>

for i, j in itertools.combinations(range(len(topic_words)), 2):
#    N = len(set(topic_words[i]) & set(topic_words[j])) / len(topic_words[i])
    print i, j, jaccard(topic_words[i], topic_words[j])

# <codecell>

jaccard = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))

# <codecell>

for i in range(len(topic_words)):
    for j in range(len(topic_words)):
        if i == j: continue
            

# <codecell>

search('meehan')

# <rawcell>

# Tom Foley speaker until '95, Hastert was there forever
# Marty Meehan was a major sponsor of campaign finance reform bills

# <codecell>

df[df.Topic == 2].Title

# <codecell>

df[-5:]

# <codecell>


# <codecell>

def tformat(l):
    N = len(l) // 3
    f = lambda x: ', '.join(x)
    return '\n'.join([f(l[:N]), f(l[N:2 * N]), f(l[2 * N:])])

topic_words[1]

# <codecell>

print tformat(topic_words[0])

# <codecell>

topic

# <codecell>

a = df[df.Topic == topic].index.map(lambda x: x.year)
plt.hist(a)

# <codecell>

_topic_words

# <codecell>

sorted(map(itemgetter('title', 'date'), filter(lambda x: 'waldholtz' in x['text'].lower(), raw)), key=itemgetter(1))

# <markdowncell>

# [KPMG](http://www.nytimes.com/2004/01/13/business/changes-at-kpmg-after-criticism-of-its-tax-shelters.html) and [Sioux](http://www.nytimes.com/2010/08/02/opinion/02mon3.html), hmo coincides w/ time

# <codecell>

raw[:2]

# <rawcell>

# plt.figsize(10, 8)
# cols = cycle('rgbcmyk')
# styles = cycle(['-'] * , '--'])
# tops = []
# for k, gp in sz.reset_index().groupby('Topic'):
#     _gp = gp.set_index('Year')[0]
#     _gp = (_gp / _gp.sum())
#     mx = _gp.max()
#     if mx < .15:
#         continue
#     _gp.plot(color=cols.next(), ls=styles.next())
#     print '{}, {}, {:.1f}'.format(k, _gp.idxmax(), mx * 100)
#     tops.append(k)
#     #gp.set_index('Year')[0].plot()
# _ = plt.legend(tops)

# <codecell>

df.groupby(year).size().plot()

# <codecell>

df['Year'] = df.index.map(year)

# <codecell>

pd.options.display.max_colwidth = 120

# <codecell>

for top in vc.index[:7]:
    print ', '.join(topic_words[top])

# <codecell>

pd.options.display.max_colwidth
vc = df[df.Year == 1998][['Title', 'Topic']].Topic.value_counts()
vc

# <codecell>

df.head(100)

# <codecell>

k

# <codecell>

gp

# <codecell>


# <codecell>

sz.plot()

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


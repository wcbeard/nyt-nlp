# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Topic detection with Pandas and Gensim
# 
# * Explore gensim
# * Show off some of pandas capabilities for data-wrangling
# 
# I grabbed the ggplot-esque [plot settings](http://matplotlib.org/users/customizing.html) from the [Probabilistic Programming for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/styles)

# <codecell>

import itertools

# <codecell>

from __future__ import division
import json
import pandas as pd
import numpy as np
from time import sleep
from itertools import count, imap, starmap, cycle, izip
import pymongo
import re
from operator import itemgetter
from gensim import corpora, models, similarities
import gensim
from collections import Counter
import datetime as dt
import matplotlib.pyplot as plt

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

raw = list(db.raw_text.find({'text': {'$exists': True}}))

# <rawcell>

# for dct in raw:
#     if 'title' not in dct:
#         dct['title'] = ''

# <markdowncell>

# ##Model generation

# <codecell>

def format(txt):
    tt = re.sub(r"'s\b", '', txt).lower()  #possessives
    tt = re.sub(r'[\.\,\;\:\'\"\(\)\&\%\*\+\[\]\=\?\!/]', '', tt)  #weird stuff
    tt = re.sub(r' *\$[0-9]\S* ?', ' <money> ', tt)  #dollar amounts
    tt = re.sub(r' *[0-9]\S* ?', ' <num> ', tt)    
    tt = re.sub(r'[\-\s]+', ' ', tt)  #hyphen -> space
    tt = re.sub(r' [a-z] ', ' ', tt)  # single letter -> space
    return tt.strip().split()

dmap = lambda dct, a: [dct[e] for e in a]

# <codecell>

#texts = [format(doc.get('text', '')) for doc in raw]
texts = [format(doc['text']) for doc in raw]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
tcorpus = dmap(tfidf, corpus)

# <codecell>

#lda = models.ldamodel.LdaModel(corpus=tcorpus, id2word=dictionary, num_topics=15, update_every=0, passes=20)
np.random.seed(42)
model = models.lsimodel.LsiModel(corpus=tcorpus, id2word=dictionary, num_topics=15)

# <markdowncell>

# ##Analysis

# <markdowncell>

# As far as I know, the only way to get the topic information for each article after fitting the model is looping through and manually grabbing the topic-score list for each article. If you have a slow computer like mine, this python-side looping may take a while.

# <codecell>

_kwargs = dict(formatted=0, num_words=20)
topic_words = [[w for _, w in tups] for tups in model.show_topics(**_kwargs)]

# <codecell>

raw[0]

# <codecell>

itertools.imap(lambda i: i, range(8))

# <codecell>

#imap(itemgetter(u'title', 'text', 'date'), raw):

# <codecell>

it.next()

# <codecell>

_text, (tit, date) = it.next()

# <codecell>

it = itertools.izip(corpus, ((d['title'], d['date']) for d in raw))
topic_data = []  #for each article, collect topic with highest score
_topic_stats = []  # gets all topic-score pairs for each document

for corp_txt, (tit, date) in it:
    _srtd = sorted(model[corp_txt], key=itemgetter(1), reverse=1)
    top, score = _srtd[0]
    topic_data.append((tit, date, top, score))
    _topic_stats.append(_srtd)

topic_stats = [tup for tups in _topic_stats for tup in tups]  #flatten list(tuples) -> list
search = lambda wrd: sorted(((b['title'], b['date'].year) for b in filter(lambda x: wrd in x['text'].lower(), raw)), key=itemgetter(1))  

# <markdowncell>

# The `topic_data` and `_topic_stats` lists keep data on each article and sorted lists of topic-score tuples:

# <codecell>

print topic_data[0]
print _topic_stats[0]

# <codecell>

set(map(len, _topic_stats))

# <codecell>

def searchf(term, fields=['title', 'date', 'url'], sort=None, reverse=True):
    ixs = [i for i, txt in enumerate(texts) if term in txt]
    items = itemgetter(*fields)
    res = [(raw[i], i) for i in ixs]
    if sort:
        res = sorted(res, key=lambda x: x[0][sort], reverse=reverse)
    return [(items(e), i) for e, i in res]

# <codecell>

print raw[1805]['text'].splitlines()[0]

# <rawcell>

# searchf('bradley', fields=['date'], sort='date')

# <codecell>

search(' n.s.a')

# <codecell>

pd.DataFrame(zip(*topic_words))

# <codecell>

him

# <codecell>

ickes: clinton deputy chief of staff http://en.wikipedia.org/wiki/Harold_M._Ickes
Charlie Trie
ickes http://www.nytimes.com/1997/09/21/magazine/bill-clinton-s-garbage-man.html?pagewanted=all&src=pm

# <codecell>

pd.options.display.max_colwidth = 100

# <rawcell>

# df[df.Topic == 3].sort('Date')[['Title', 'Date']]

# <rawcell>

# search('enron')

# <markdowncell>

# Now we can put the topic information into pandas, for faster, easier analysis.

# <codecell>

_df = pd.DataFrame(topic_data, columns=['Title', 'Date', 'Topic', 'Score'])
df = _df#.set_index('Date').sort_index()#.head()

# <codecell>

df.head()

# <codecell>

df.shape

# <markdowncell>

# By plotting the distribution of topic labels for each document, we can see that the detected topics are not very evenly distributed.

# <codecell>

vc = df.Topic.value_counts()
plt.bar(vc.index, vc)

# <rawcell>

# df.Topic.value_counts()

# <codecell>

sdf = pd.DataFrame(topic_stats, columns=['Topic', 'Score'])
#df = _df.set_index('Date').sort_index()#.head()
topic_mean = sdf.groupby('Topic').mean()['Score']
#topic_mean

# <codecell>

topic_stats[:4]

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

# <markdowncell>

# One high level question I had was if certain topics can be seen varying in frequency over time. Using Pandas' `groupby` can be used to aggregate the article counts by year and topic:

# <codecell>

year = lambda x: x.year
sz = df.set_index('Date').groupby(['Topic', year]).size()#.reset_index()
sz.index.names, sz.name = ['Topic', 'Year'], 'Count'
sz = sz.reset_index()
sz.head()

# <markdowncell>

# which can then be reshapen with `pivot`, giving us a Year $\times$ Topic grid:

# <codecell>

top_year = sz.pivot(index='Year', columns='Topic', values='Count').fillna(0)
top_year

# <markdowncell>

# In Pandas land it's easy to find lots of basic information about the distribution--a simple boxplot will give us a good view of the min/median/max number of times a topic was represented over the 21 years.

# <codecell>

plt.figsize(12, 8)
_ = top_year.boxplot()

# <markdowncell>

# We can see topics 8 and 10 hardly show up, while in typical years the first two topics are heavily represented. (And for the curious, viewing the distribution of scandalous articles across topics for a given year is as easy as `top_year.T.boxplot()`.)

# <rawcell>

# _ = top_year.T.boxplot()

# <markdowncell>

# The `plot` method can automatically plot each column as a separate time series, which can give a look at the trend for each scandal-topic:

# <codecell>

_ = top_year.plot()

# <codecell>

gensim.models.lsimodel?

# <codecell>

gensim.utils.random.setstate(

# <markdowncell>

# But because topics like (7 and 13) are

# <codecell>

(top_year / top_year.sum()).plot()

# <markdowncell>

# And averaging for all the topics per year shows spikes in 1998 and 2006 for number of articles including the term *scandal*:

# <codecell>

top_year.mean(axis=1).plot()

# <rawcell>

# vc = df.Topic.value_counts()
# vc /= vc.sum()
# vc

# <codecell>

styles = cycle(['-'])#, '--', '-.', ':'])

# <codecell>

def plottable(k, gp, thresh=None):
    _gp = gp.set_index('Year')[0]
    _gp = (_gp / _gp.sum())
    mx = _gp.max()    
    if mx < thresh:
        return
    return k, _gp

# <codecell>

yr_grps = filter(None, starmap(plottable, sz.reset_index().groupby('Topic')))

# <codecell>

yr_grps

# <codecell>

plt.figsize(10, 8)
cols = cycle('rbcykmg')
_rep = int(round(len(yr_grps) / 2))
#styles = cycle('-')  #cycle((['-'] * _rep) + (['--'] * _rep))
tops = []
for k, gp in yr_grps:
    gp.plot(color=cols.next())
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

# <codecell>

df[df.Topic == 2].head().ix[367][0]

# <codecell>

search(' n.s.a')

# <codecell>

pd.DataFrame(topic_words).T

# <codecell>

_topic_words

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
search('fordham')

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


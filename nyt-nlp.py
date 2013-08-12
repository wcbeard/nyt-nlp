# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


# <markdowncell>

# #Topic detection with Pandas and Gensim
# 
# A few months ago, the [undending](http://en.wikipedia.org/wiki/2013_IRS_scandal
# ) [series](http://thelead.blogs.cnn.com/2013/08/01/exclusive-dozens-of-cia-operatives-on-the-ground-during-benghazi-attack/
# ) of [recent](http://www.usatoday.com/story/news/2013/05/13/justice-department-associated-press-telephone-records/2156521/) [scandals](http://en.wikipedia.org/wiki/2013_mass_surveillance_scandal
# ) inspired me to see whether it would be possible to comb through the text of New York Times articles and automatically detect and identify different scandals that have occurred. I wanted to see if, given articles about the DOJ, IRS, NSA and all the rest, whether the text would be enough for an algorithm to identify them as distinct scandals and distinguish them from one another, in an unsupervised fashion.
# 
# This also gave me an excuse to explore [gensim](http://radimrehurek.com/gensim/) and show off some of [pandas](http://pandas.pydata.org/) capabilities for data-wrangling.
# 
# The IPython notebook for this post is available **here** (and I grabbed the ggplot-esque [plot settings](http://matplotlib.org/users/customizing.html) from [Probabilistic Programming for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/styles)).
# 
# Let's get started by by picking up where we left off, and pull all those articles out of mongo.

# <codecell>

from __future__ import division
import json
import pandas as pd
import numpy as np
from time import sleep
#from itertools import count, imap, starmap, cycle, izip
import itertools
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

connection = pymongo.Connection("localhost", 27017)
db = connection.nyt

# <codecell>

raw = list(db.raw_text.find({'text': {'$exists': True}}))

# <rawcell>

# #Run this the first time, to make sure it doesn't choke on title-less documents
# for dct in raw:
#     if 'title' not in dct:
#         dct['title'] = ''

# <markdowncell>

# ###Some helpful functions
# The `format` function should be pretty self-explanatory, and `search` is to be used later on to verify topic words. 

# <codecell>

def format(txt):
    """Turns a text document to a list of formatted words.
    Get rid of possessives, special characters, multiple spaces, etc.
    """
    tt = re.sub(r"'s\b", '', txt).lower()  #possessives
    tt = re.sub(r'[\.\,\;\:\'\"\(\)\&\%\*\+\[\]\=\?\!/]', '', tt)  #weird stuff
    tt = re.sub(r' *\$[0-9]\S* ?', ' <money> ', tt)  #dollar amounts
    tt = re.sub(r' *[0-9]\S* ?', ' <num> ', tt)    
    tt = re.sub(r'[\-\s]+', ' ', tt)  #hyphen -> space
    tt = re.sub(r' [a-z] ', ' ', tt)  # single letter -> space
    return tt.strip().split()


def search(wrd, df=True): 
    """Searches through `raw` list of documents for term `wrd` (case-insensitive).
    Returns titles and dates of matching articles, sorted by date. Returns
    DataFrame by default.
    """
    wrd = wrd.lower()
    _srch = lambda x: wrd in x['text'].lower()
    title_yr = ((b['title'], b['date'].year) for b in filter(_srch, raw))
    ret = sorted(title_yr, key=itemgetter(1))
    return pd.DataFrame(ret, columns=['Title', 'Year']) if df else ret


dmap = lambda dct, a: [dct[e] for e in a]

# <markdowncell>

# ##Model generation
# Now apply the `format` function to all the text, and convert it to a dictionary of word counts per document form that gensim's models can work with. The `TfidfModel` transformation will [take into account](en.wikipedia.org/wiki/Tf–idf) how common a word is in a certain document compared to how common it is overall (so the algorithm won't just be looking at the most common, but uninformative words like *the* or *and*).

# <codecell>

texts = [format(doc['text']) for doc in raw]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
tcorpus = dmap(tfidf, corpus)

# <codecell>

np.random.seed(42)
model = models.lsimodel.LsiModel(corpus=tcorpus, id2word=dictionary, num_topics=15)

# <markdowncell>

# ##Analysis

# <markdowncell>

# As far as I know, the only way to get the topic information for each article after fitting the model is looping through and manually grabbing the topic-score list for each article.

# <codecell>

_kwargs = dict(formatted=0, num_words=20)
topic_words = [[w for _, w in tups] for tups in model.show_topics(**_kwargs)]

# <codecell>

%%time

it = itertools.izip(corpus, ((d['title'], d['date']) for d in raw))
topic_data = []  #for each article, collect topic with highest score
_topic_stats = []  # gets all topic-score pairs for each document

for corp_txt, (tit, date) in it:
    _srtd = sorted(model[corp_txt], key=itemgetter(1), reverse=1)
    top, score = _srtd[0]
    topic_data.append((tit, date, top, score))
    _topic_stats.append(_srtd)

topic_stats = [tup for tups in _topic_stats for tup in tups]  #flatten list(tuples) -> list

# <markdowncell>

# The `topic_data` and `_topic_stats` lists keep data on each article and sorted lists of topic-score tuples:

# <codecell>

print topic_data[0]
print _topic_stats[0]

# <rawcell>

# def searchf(term, fields=['title', 'date', 'url'], sort=None, reverse=True):
#     ixs = [i for i, txt in enumerate(texts) if term in txt]
#     items = itemgetter(*fields)
#     res = [(raw[i], i) for i in ixs]
#     if sort:
#         res = sorted(res, key=lambda x: x[0][sort], reverse=reverse)
#     return [(items(e), i) for e, i in res]

# <rawcell>

# searchf('bradley', fields=['date'], sort='date')

# <codecell>

print raw[1805]['text'].splitlines()[0]

# <codecell>

#search('n.s.a')
search(' nsa ', df=False)

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

# <rawcell>

# topic_words

# <codecell>

len(topic_words)

# <markdowncell>

# Now we can put the topic information into pandas, for faster, easier analysis.

# <codecell>

df = pd.DataFrame(topic_data, columns=['Title', 'Date', 'Topic', 'Score'])
df.Date = df.Date.map(lambda d: d.date())
print df.shape
df.head()

# <markdowncell>

# By plotting the distribution of topic labels for each document, we can now see that the detected topics are not very evenly distributed.

# <codecell>

vc = df.Topic.value_counts()
plt.bar(vc.index, vc)
_ = plt.ylabel('Topic count')

# <rawcell>

# df.Topic.value_counts()

# <rawcell>

# sdf = pd.DataFrame(topic_stats, columns=['Topic', 'Score'])
# #df = _df.set_index('Date').sort_index()#.head()
# topic_mean = sdf.groupby('Topic').mean()['Score']
# #topic_mean

# <codecell>

topic_stats[:4]

# <rawcell>

# center = lambda top, score: score - topic_mean[top]
# center_list = lambda lst: [(top, center(top, score)) for top, score in lst]
# centered_scores = [max(center_list(lst), key=itemgetter(1)) for lst in _topic_stats]
# second_scores = [sorted(lst, key=itemgetter(1),  reverse=1)[:2][-1] for lst in _topic_stats]

# <markdowncell>

# One high level question I had was if certain topics can be seen varying in frequency over time. Pandas' `groupby` can aggregate the article counts by year and topic:

# <codecell>

d = df.Date[0]
str(d.year) + str(d.month).zfill(2)

# <codecell>

year = lambda x: x.year
    #year = lambda d: d.strftime('%Y%m')
sz = df.set_index('Date').groupby(['Topic', year]).size()#.reset_index()
sz.index.names, sz.name = ['Topic', 'Year'], 'Count'
sz = sz.reset_index()
sz.head()

# <markdowncell>

# which can then be reshapen with `pivot`, giving us a Year $\times$ Topic grid:

# <codecell>

top_year = sz.pivot(index='Year', columns='Topic', values='Count').fillna(0)
top_year #[:20]

# <markdowncell>

# In Pandas land it's easy to find lots of basic information about the distribution--a simple boxplot will give us a good view of the min/median/max number of times a topic was represented over the 21 years.

# <codecell>

#plt.figsize(12, 8)
#_ = top_year.boxplot()
plt.figure(figsize=(12, 8))
top_year.boxplot() and None

# <markdowncell>

# Topics 8, 9, 11 and 12 hardly show up, while in typical years topics like 1 and 2 are heavily represented. The plot also shows that articles most closely associated with topic 2 actually show up 250 times for one year.
# 
# (For the curious, viewing the distribution of scandalous articles across topics for a given year is as easy as `top_year.T.boxplot()`.)

# <markdowncell>

# The `plot` method can automatically plot each column as a separate time series, which can give a look at the trend for each scandal-topic:

# <codecell>

_ = top_year.plot(figsize=(12, 8))

# <rawcell>

# pd.stats.moments.rolling_mean(top_year, 4).plot(figsize=(12, 8))

# <markdowncell>

# The number of times articles with different topics show up in a year varies a lot for most of the topics. It even looks like there are a few years, like 1998 and 2006 where multiple topics spike. This can be verified by plotting the sum of articles for all topics in a given year:

# <codecell>

_ = top_year.sum(axis=1).plot()

# <markdowncell>

# ###Topic words

# <codecell>

top_year

# <codecell>

pd.DataFrame({'Sum': top_year.sum(), 'Max': top_year.max()}).sort('Sum', ascending=0)

# <markdowncell>

# Stacking all the words of the topics and getting the value counts gives an idea of how often certain words show up among the topics:

# <rawcell>

# pd.DataFrame(zip(*topic_words))

# <codecell>

pd.options.display.max_rows = 22
top_wds_df = pd.DataFrame(zip(*topic_words))
vc = top_wds_df.stack(0).value_counts()
print vc
pd.options.display.max_rows = 400

# <markdowncell>

# It looks like the most common topic words are *page*, *enron*, *bush* and *clinton*, with *gore* just behind. It seems these words might be less helpful at finding the meaning of topics since they're closely associated with practically every topic of political scandal in the past two decades. It shouldn't be surprising that presidents show up among the most common topic words, and a cursory look at articles with the word *page* (using `search`, defined above) makes it look like the word shows up both for sexual scandals involving pages, along with a bunch references to *front page scandals* or the *op-ed page*.
# 
# You can find specific headlines from my dataset that include the word *page* (which [duckduckgo](duckduckgo.com) should be able to handle) with `search('page')`.
# 
# In the following, I've given a simple score to the topic words based on how unique they are (from a low score of 0 for the most common, up to 11 for words that only appear for a single topic). All 15 topics are summarized below with the top words scored by how common they are.
# 
# Topics 1 and 6 look to have the most cliched scandal words, while the last few topics are characterized by quite a few unique words.

# <rawcell>

# top_wd_freq = {w: vc.max() - cnt for w, cnt in vc.iteritems()}
# _ = top_wds_df.apply(lambda s: s.map(top_wd_freq)).boxplot()

# <codecell>

pd.options.display.line_width = 130
top_wd_freq = {w: '{}-{}'.format(w, vc.max() - cnt) for w, cnt in vc.iteritems()}
top_wds_df.apply(lambda s: s.map(top_wd_freq))

# <markdowncell>

# ##Story telling

# <markdowncell>

# Now comes the fun part, where we can try to find explanations for the choices of topics generated by Gensim's implementation of [LSI](http://en.wikipedia.org/wiki/Latent_semantic_indexing). While LSI can be very good at finding hidden factors and relationships (i.e., topics) from different documents, there is no way that I'm aware of to easily interpret the algorithm to see why it groups documents with certain topics. The best way I know is to eyeball it, which we can do from the topic-word dataframe above.
# 
# For example, topics 1 and 5 include the words *impeachment, lewinsky, gore, clinton* and *starr*, so it's probably a safe bet to say they're referring to the [Lewinsky scandal](http://en.wikipedia.org/wiki/Lewinsky_scandal). And looking at the topic-year plot from above ******** (`In [36]`), we can see that at least topic 5 has a major spike in the years following the scandal.
# 
# Both also include the rather high-scoring terms *prime* and *minister*, which are probably indicative of the large number of world news summaries included under the topics. For example, 343 of Topic 1's articles have the title *News Summary*, while no other topic has even 40 summaries:

# <codecell>

t1 = df[df.Topic == 1].Title.value_counts()
t1[t1 > 10]

# <markdowncell>

# Topic 3 looks like it's associated with state- and city-level scandals in the New England region. Aside from the cliched terms, we have *rowland* and *rell*, likely in reference to [corruption in Connecticut](http://en.wikipedia.org/wiki/John_G._Rowland#Corruption_as_Governor), and some more pretty specific indicators like *mayor, governor, cuomo, spitzer, city, state* and *albany*.
# 
# Topic 12 looks like it covers New Jersey pretty well. Other than the state's name itself as one of the topic words, you've got [Corzine](en.wikipedia.org/wiki/Jon_Corzine), [Torricelli](http://en.wikipedia.org/wiki/Robert_Torricelli), [Codey](http://en.wikipedia.org/wiki/Richard_Codey), [Schundler](http://en.wikipedia.org/wiki/Bret_Schundler) and [Lautenberg](http://en.wikipedia.org/wiki/Frank_Lautenberg), none of which appear outside of this topic except for *Corzine*.
# 
# Several look international in nature, especially topic 9, which has strong Italian (*berlusconi, italy, italian* and the unique *andreotti* terms) and Japanese (*japan, japanese, [Ozawa](http://en.wikipedia.org/wiki/Ichir%C5%8D_Ozawa), [Hosokawa](http://en.wikipedia.org/wiki/Morihiro_Hosokawa)* and *[Kanemaru](http://en.wikipedia.org/wiki/Shin_Kanemaru)*) showings, and also uniquely identifies German chancellor [Helmut Kohl](http://en.wikipedia.org/wiki/Helmut_Kohl).
# 
# Topic 13 seems to represent public finance scandals, with unique terms *budget, tax, percent, billion* and *plan*, while topic 8 looks like it pertains more to campaign finance, with unique terms *soft, money, raising, earmarks* and *lobbyists*. Topic 7 looks like it has to do with corporate scandals, leading with the pervasive *enron* term, but with largely unique terms *accounting, stock, corporate, attorney, counsel, investigation, companies* and *justice* [as in Department of...?] as well.
# 
# And finally the 2nd topic appears to have a lot of legislative factors in it, with terms unique terms *house, senate, lawmakers, ethics, committee, bill* and *parliament*.
# 
# ##Conclusion
# 
# The results give a much less fine-grained view of scandals than what I was expecting, either because of the sources (not enough articles devoted specifically enough to particular scandals? text not sufficiently preprocessed) or the algorithm (wrong algorithm for the task? wrong settings?). Plus, it turns out there have been a *lot* of American political scandals in the last 20 years. Perhaps more clear patterns could be discerned by expanding the number of topics. 
# 
# The detected topics seem to have a lot of noise (for example, the presidents' names show up as key words in *every* topic), possibly due to the imbalance from some scandals being cited more frequently than others. But when you cut out the noise and try to characterize the topics by the more infrequent key words, I was surprised by the topic clusters it was actually able to detect, from international scandals to corporate scandals to Jersey scandals. I was unfortunately not able to detect the recent set of scandals, but from the experiment, the good scandals seem to require a few years to age before there is enough data to detect them. Hopefully today's events will be easy enough to spot by rerunning this in a few months or years.
# 
# All in all, it was a fun exercise and a good reminder of the strong tradition of corruption we're part of.

# <rawcell>

# top_wds_df[[9]].apply(lambda s: s.map(top_wd_freq))

# <codecell>

tit, yr = zip(*search('kohl'))
print len(tit)
tit

# <codecell>

_df = df[df.Topic.isin([9])]
#_df

# <rawcell>

# tit, yr = zip(*minst)

# <codecell>

_df.Title

# <codecell>

n = 0
_subs = _df[_df.Title.isin(tit)].sort('Date')
#_subs[n:100+n]

# <codecell>

_subs.sort('Score', ascending=0)

# <codecell>

for top, __df in df.groupby('Topic'):
    _vc = __df.Title.value_counts()
    _common_tit = _vc[_vc > 10]
    if len(_common_tit):
#        _common_tit.name = top
        print top
        print _common_tit

# <codecell>

_subs.Title.value_counts()

# <codecell>

len(set(top_wds_df[1]) & set(top_wds_df[5]))

# <codecell>


# <codecell>

top_year.head()

# <codecell>

_ = top_year[[1, 5]].plot(figsize=(12, 8))

# <codecell>

headlines

# <codecell>

Among the topics shown above

# <codecell>

c = 0

# <codecell>

np.random.seed(c)
print c
c += 1
top = df[df.Topic == 13]
ix = np.random.choice(top.index, 15)
top.ix[ix].sort('Date')#[:30]

# <codecell>


# <codecell>

search('billion')

# <codecell>

13, 1, 2, 4

# <codecell>

c

# <rawcell>

# And lastly, 

# <rawcell>

# (top_year / top_year.sum()).sum(axis=1).plot()

# <markdowncell>

# But because topics like (7 and 13) are

# <markdowncell>

# And averaging for all the topics per year shows spikes in 1998 and 2006 for number of articles including the term *scandal*:

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


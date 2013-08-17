# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# {"Title": "Scraping NYT for Scandals",
# "Date": "2013-7-4",
# "Category": "ipython",
# "Tags": "nlp, ipython",
# "slug": "nyt-scraping",
# "Author": "Chris"
# }

# <markdowncell>

# This post uses the [New York Times API](http://developer.nytimes.com/docs/read/article_search_api_v2) to search for articles on US politics that include the word *scandal*, and several python libraries to grab the text of those articles and store them to MongoDB for some natural language processing analytics (in [part 2](|filename|/nyt-nlp.ipynb) of this project).

# <markdowncell>

# These commands should install some of the dependencies for this part of the project:
# 
#     pip install pymongo
#     pip install requests
#     pip install lxml
#     pip install cssselect

# <codecell>

import requests
import json
from time import sleep
import itertools
import functools
from lxml.cssselect import CSSSelector
from lxml.html import fromstring
import pymongo
import datetime as dt
from operator import itemgetter

# <codecell>

%load_ext autosave
%autosave 30

# <markdowncell>

# ###Mongodb

# <markdowncell>

# We need to connect to the database, assuming it's already running (`mongod` from the terminal).

# <codecell>

connection = pymongo.Connection("localhost", 27017 )
db = connection.nyt

# <markdowncell>

# ##Get URLs

# <markdowncell>

# After using your secret API key...

# <codecell>

from key import apikey
apiparams = {'api-key': apikey}

# <markdowncell>

# ...the first thing we need to get is the urls for all the articles that match our search criterion. [Big caveat: I used a v1 api query for the original dataset that I used, and modified it for v2 after discovering it].
# 
# I searched for *scandal* with the `q` parameter, and narrowed it down using *republican OR democrat* with the `f[ilter]q[uery]` parameter. I found out that there are lots of really interesting curated details you can use in the search, such as searching for articles pertaining to certain geographic areas, people or organizations (with a feature called *facets*. I used other parameters to restrict the dates to years 1992-2013, and just return certain fields I thought would be relevant:

# <codecell>

page = 0

q = 'http://api.nytimes.com/svc/search/v2/articlesearch.json?'
params = {'q': 'scandal*',
            'fq': 'republican* OR democrat*',
            'fl': 'web_url,headline,pub_date,type_of_material,document_type,news_desk',
            'begin_date': '19920101',
            'end_date': '20131231',
            'page': page,
            'api-key': apikey,
            'rank': 'newest',
}

# <markdowncell>

# After constructing the query, we grab the search results with [requests](http://docs.python-requests.org/en/latest/). There's no way to tell how many results there will be, so we go as long as we can, shoving everything into MongoDB, incrementing the `offset` query parameter and pausing for a break before the next page of results (the NYT has a limit on how many times you can query them per second).

# <codecell>

def memoize(f):
    "Memoization for args and kwargs"
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        kw_tup = tuple((kargname, tuple(sorted(karg.items()))) for kargname, karg in kwargs.items())
        memo_args = args + kw_tup
        try:
            return wrapper.cache[memo_args]
        except KeyError:
            print '*',
            wrapper.cache[memo_args] = res = f(*args, **kwargs)
            return res
    wrapper.cache = {}
    return wrapper

    
@memoize
def search_nyt(query, params={}):
    r = requests.get(query, params=params)
    res = json.loads(r.content)
    sleep(.1)
    return res

# <codecell>

fdate = lambda d, fmt='%Y%m%d': dt.datetime.strptime(d, fmt)

for page in itertools.count():  #keep looping indefinitely
    params.update({'page': page})  #fetch another ten results from the next page
    res = search_nyt(q, params=params)["response"]
    if res['docs']:
        for dct in res['docs']:
            dct = dct.copy()  #for memoization purposes
            url = dct.pop('web_url')
            dct['pub_date'] = fdate(dct['pub_date'].split('T')[0], '%Y-%m-%d')  #string -> format as datetime object
            dct['headline'] = dct['headline']['main']
            db.raw_text.update({'url': url}, {'$set': dct}, upsert=True)
    else:  #no more results
        break
    if page % 100 == 0:
        print page,

# <markdowncell>

# We can see from the response's metadata that we should expect about 11876 article links to be saved to our database:

# <codecell>

search_nyt(q, params=params)['response']['meta']

# <markdowncell>

# ##Scrape Text

# <markdowncell>

# Here are a few of the resulting URLS that we'll use to get the full text articles:

# <codecell>

[doc['url'] for doc in db.raw_text.find()][:5]

# <markdowncell>

# The scraping wasn't as difficult as I was expecting; over the 20 or so years that I searched for, the body text of the articles could be found by looking at 5 html elements (formatted as CSS selectors in `_sels` below). The following two functions do most of the scraping work-- `get_text`...well...gets the text from the `CSSSelector` parser, and `grab_text` uses this after pulling the html with requests.

# <codecell>

def get_text(e):
    "Function to extract text from CSSSelector results"
    try:
        return ' '.join(e.itertext()).strip().encode('ascii', 'ignore')
    except UnicodeDecodeError:
        return ''


def grab_text(url, verbose=True):
    "Main scraping function--given url, grabs html, looks for and returns article text"
    if verbose and (grab_text.c % verbose == 0):  #page counter
        print grab_text.c,
    grab_text.c += 1
    r = requests.get(url, params=all_pages)
    content = fromstring(r.content)
    for _sel in _sels:
        text_elems = CSSSelector(_sel)(content)
        if text_elems:
            return '\n'.join(map(get_text, text_elems))
    return ''

#Selectors where text of articles can be found; several patterns among NYT articles
_sels =  ['p[itemprop="articleBody"]', "div.blurb-text", 'div#articleBody p', 'div.articleBody p', 'div.mod-nytimesarticletext p']
all_pages = {'pagewanted': 'all'}

# <markdowncell>

# And here is the main loop and counter. Pretty simple.
# 
# On the first run, the output counts up from zero, but since political scandals seem to be popping up by the hour, I've updated the search a few times, but only pulling articles that aren't already in the database (hence the very sparse output below).

# <codecell>

exceptions = []
grab_text.c = 0

for doc in db.raw_text.find(timeout=False):
    if ('url' in doc) and ('text' not in doc):
        # if we don't already have this in mongodb
        try:
            txt = grab_text(doc['url'], verbose=1)        
        except Exception, e:
            exceptions.append((e, doc['url']))
            print ':(',
            continue
        db.raw_text.update({'url': doc['url']}, {'$set': {'text': txt}})
    else:
        pass

db.raw_text.remove({'text': u''})  #there was one weird result that didn't have any text...

# <markdowncell>

# I used the following code to clean up after I downloaded most of the articles and modified the query, which changed some of the output fields. This just renames the original fields and sets the `title` to an empty string where there was none. 

# <codecell>

for doc in db.raw_text.find(timeout=False):
    if ('pub_date' in doc) and ('date' not in doc):
        # cleanup from v1 queries
        updates = {'date': doc['pub_date'],
                   'title': doc['headline']}
        db.raw_text.update({'url': doc['url']}, {'$set': updates})
    if ('title' not in doc) and ('headline' not in doc):
        #cleanup
        db.raw_text.update({'url': doc['url']}, {'$set': {'title': ''}})

# <markdowncell>

# A few of the articles set off exceptions when I tried pulling them; these weren't included in the dataset:

# <codecell>

exceptions

# <codecell>

res = map(itemgetter('date', 'url', 'title'), sorted(db.raw_text.find(), key=itemgetter('date'), reverse=1))

# <codecell>

res[:3]

# <markdowncell>

# And, we got more than 9700 scandalous stories...but still counting!

# <codecell>

len(list(db.raw_text.find()))

# <markdowncell>

# ###Conclusion
# Though it turned out to be pretty brief, I thought this first part of my NYT scandals project deserved its own post.
# Luckily it doesn't take too much effort or space when you're working with a nice, expressive language, though.
# And you can reproduce this for yourself--you can find a copy of this notebook on [Github](https://github.com/d10genes/nyt-nlp).


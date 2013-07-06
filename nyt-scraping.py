# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# {"Title": "NYT nlp",
# "Date": "2013-7-4",
# "Category": "ipython",
# "Tags": "nlp, ipython",
# "slug": "slug-slug-slug",
# "Author": "Chris"
# }

# <markdowncell>

# This document uses the [NYT API](http://developer.nytimes.com/docs/read/article_search_api_v2) to search for articles on US politics that include the word *scandal*, and several python libraries to grab the text of those articles and store them to MongoDB for some natural language processing analytics.

# <markdowncell>

# These commands will install some of the dependencies for this project:

# <codecell>

%%bash
pip install pymongo
pip install requests
pip install lxml
pip install cssselect

# <codecell>

import requests
import json
from time import sleep
from itertools import count
from lxml.cssselect import CSSSelector
from lxml.html import fromstring
import pymongo
import datetime as dt

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

# ...the first thing we need to get is the urls for all the articles that match our search criterion. This is a bit convoluted, since I couldn't find a way to search for *republican OR democrat*, so I ended up just repeating the query both times. I found out that there are lots of really interesting curated details you can use in the search, such as searching for articles pertaining to certain geographic areas, people or organizations. I used some of these features to narrow the results down to the US, restricted the dates to between 1992-2013, and just asked for title, URL and date to use as unique identifiers.

# <codecell>

page = 0
party = 'republican'
#party = 'democrat'

q = 'http://api.nytimes.com/svc/search/v1/article?'
params = {'query': 'body:scandal+{} geo_facet:[UNITED STATES]'.format(party),
            'begin_date': '19920101',
            'end_date': '20131201',
            'fields': 'title,url,date',
            'offset': page,
            'api-key': apikey
}

# <markdowncell>

# After constructing the query, we grab the search results with [requests](http://docs.python-requests.org/en/latest/). There's no way to tell how many results there will be, so we go as long as we can, shoving everything into MongoDB, incrementing the `offset` query parameter and pausing for a break before the next page of results (the NYT has a limit on how many times you can query them per second).

# <codecell>

fdate = lambda d: dt.datetime.strptime(d, '%Y%m%d')

for page in count():  #keep looping indefinitely
    params.update({'offset': page})  #fetch another ten results from the next page
    r = requests.get(q, params=params)
    res = json.loads(r.content)["results"]
    if res:
        for dct in res:
            url = dct.pop('url')
            dct['date'] = fdate(dct['date'])  #string -> format as datetime object
            db.raw_text.update({'url': url}, {'$set': dct}, upsert=True)
    else:  #no more results
        break
    print page,
    sleep(.2)
#urls = {r['url']: r for r in url_lst}    
#del url_lst

# <markdowncell>

# ##Scrape Text

# <markdowncell>

# Here are a few of the resulting URLS that we'll use to get the full text articles:

# <codecell>

[doc['url'] for doc in db.raw_text.find()][:5]

# <markdowncell>

# The scraping wasn't as difficult as I was expecting; over the 20 or so years that I searched for, the body text of the articles could be found by looking at 5 html elements (formatted as CSS selectors in `_sels` below). The following two functions do most of the scraping work-- `get_text`...well...gets the text from the `CSSSelector` parser, and the `grab_text` uses this after pulling the html with requests again.

# <codecell>

def get_text(e):
    "Function to extract text from CSSSelector results"
    try:
        return ' '.join(e.itertext()).strip().encode('ascii', 'ignore')
    except UnicodeDecodeError:
        return ''


def grab_text(url, verbose=True):
    "Main scraping function--given url, grabs html, looks for and returns article text"
    if verbose:  #page counter
        print grab_text.c,
    grab_text.c += 1
    r = requests.get(url, params=all_pages)
    content = fromstring(r.content)
    for _sel in _sels:
        text_elems = CSSSelector(_sel)(content)
        if text_elems:
            return '\n'.join(map(get_text, text_elems))
    return ''

#Selectors where text of articles can be found; quite a few patterns among NYT articles
_sels =  ['p[itemprop="articleBody"]', "div.blurb-text", 'div#articleBody p', 'div.articleBody p', 'div.mod-nytimesarticletext p']
all_pages = {'pagewanted': 'all'}

# <markdowncell>

# And here is the main loop and counter. Pretty simple, huh?
# 
# One the first run, the output counts up from zero, but since political scandals seem to be popping up by the hour, I've updated the search a few times, but only pulling articles that aren't already in the database (hence the mostly underscored output below).

# <codecell>

grab_text.c = 0
for doc in db.raw_text.find():
    if ('url' in doc) and ('text' not in doc):
        # if we don't already have this in mongodb
        txt = grab_text(doc['url'])
        db.raw_text.update({'url': doc['url']}, {'$set': {'text': txt}})
    else:
        print '_',

db.raw_text.remove({'text': u''})  #there was one weird result that didn't have any text...

# <markdowncell>

# And, we got more than 870 scandalous stories...but still counting.

# <codecell>

len(list(db.raw_text.find()))

# <markdowncell>

# ###Conclusion
# Though it turned out to be pretty brief, I thought this first part of my NYT scandals project deserved its own post.
# Luckily it doesn't take too much effort or space when you're working with a nice, expressive language, though.
# And you can reproduce this for yourself--you can find a copy of this notebook on github.


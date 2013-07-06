# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <rawcell>

# pip install pymongo
# pip install requests
# pip install lxml
# pip install cssselect

# <rawcell>

# import redis
# import cPickle as pickle
# import numpy as np
# import pandas as pd
# import myutils as mu
# mu.psettings(pd)

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

# <codecell>

connection = pymongo.Connection( "localhost", 27017 )
db = connection.nyt

# <markdowncell>

# ##Get URLs

# <codecell>

from key import apikey
apiparams = {'api-key': apikey}

# <codecell>

page = 0
party = 'republican'
#party = 'democrat'

q = 'http://api.nytimes.com/svc/search/v1/article?'
params = {'query': 'body:scandal+{} geo_facet:[UNITED STATES]'.format(party),
            #'facets': 'des_facet,per_facet,geo_facet,classifiers_facet,org_facet', #'org_facet',
            'begin_date': '19920101',
            'end_date': '20131201',
            'fields': 'title,url,date',
            'offset': page,
            'api-key': apikey
}

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
    sleep(.2)  #nyt doesn't like it if you ask too often
#urls = {r['url']: r for r in url_lst}    
#del url_lst

# <markdowncell>

# ##Scrape Text

# <codecell>

[doc['url'] for doc in db.raw_text.find()][:5]

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

# <codecell>

len(list(db.raw_text.find()))


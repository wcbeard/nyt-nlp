# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import requests
import json
import pandas as pd
import numpy as np
from time import sleep
from itertools import count
import cPickle as pickle
from lxml.cssselect import CSSSelector
from lxml.html import fromstring
import redis
import pymongo

# <codecell>

import myutils as mu
mu.psettings(pd)

# <codecell>

%load_ext autosave
%autosave 30


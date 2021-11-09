"""
This code converts the datasets from SpERT to the format that fits to the model. 11-04-2021

- SpERT github: https://github.com/lavis-nlp/spert

"""

import os
import sys
import pandas as pd
import json
import re
import pickle


orig_file = 'datasets/scierc/spert/scierc_train.json'

data = json.load(open(orig_file))

for item in data:
	print(item)
	input('enter..')
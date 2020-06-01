import pandas as pd
import re

def extr_words(x):
    return re.findall('"([^"]*)"', x)

df = pd.read_csv('topics.csv')
df['extr_words'] = df.words.apply(lambda x: ' '.join(extr_words(x)))
df.to_csv('topicswithwords.csv')

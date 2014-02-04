import sys
import scipy as sp
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer


def dist(v1, v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

with open('apple-computers.txt') as f:
    content_apple_computers = unicode(f.read(), "ISO-8859-1")
with open('apple-fruit.txt') as f:
    content_apple_fruit = unicode(f.read(), "ISO-8859-1")

input_text = sys.stdin.readlines()
# with open('input.txt') as f:
#     input_text = unicode(f.read(), "ISO-8859-1")
# input_text = input_text.split('\n')
input_text.pop(0)

english_stemmer = nltk.stem.SnowballStemmer('english')
vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words='english', decode_error='ignore')

X = vectorizer.fit_transform([content_apple_computers, content_apple_fruit])

new_post_vec = vectorizer.transform(input_text)

for i in len(input_text):
    d1 = dist(X.getrow(0), new_post_vec.getrow(i))
    d2 = dist(X.getrow(1), new_post_vec.getrow(i))
    if d1 < d2:
        print 'computer-company'
    else:
        print 'fruit'
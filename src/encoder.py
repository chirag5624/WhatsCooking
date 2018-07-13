from sklearn.feature_extraction.text import CountVectorizer

s = [u'this', u'is very', u'test']
tb = CountVectorizer()
dict = tb.fit_transform(s)
print dict.todense()
print tb.get_feature_names()




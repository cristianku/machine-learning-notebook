import graphlab

people = graphlab.SFrame('people_wiki.gl/')

# print people.head()

people['word_count'] = graphlab.text_analytics.count_words(people['text'])

tfidf = graphlab.text_analytics.tf_idf(people['word_count'])


people['tfidf'] = tfidf

print people.head()


elton = people[people['name'] == 'Elton John']

print ' Top word count words for Elton John : '

elton[['word_count']].stack('word_count',new_column_name=['word','count']).sort('count',ascending=False)

print ' Top tfidf count  for Elton John : '
elton[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)

beckam = people[people['name'] == 'Victoria Beckham']


paul   = people[people['name']== 'Paul McCartney' ]

print ' graphlab.distances.cosine elton john and beckam '

print graphlab.distances.cosine(elton['tfidf'][0],beckam['tfidf'][0])

print ' graphlab.distances.cosine elton john and Paul McCartney '

print graphlab.distances.cosine(elton['tfidf'][0],paul['tfidf'][0])

knn_model_cosine_tfdif = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name', distance='cosine')

knn_model_cosine_word_count = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name', distance='cosine')

print ' =========='
print ' = tfdif neighbors elton ========='
print ' =========='

print knn_model_cosine_tfdif.query(elton)
print ' =========='
print ' = word count neighbors elton ========='
print ' =========='
print knn_model_cosine_word_count.query(elton)


print ' =========='
print ' = tfdif neighbors beckam ========='
print ' =========='

print knn_model_cosine_tfdif.query(beckam)
print ' =========='
print ' = word count neighbors beckam ========='
print ' =========='
print knn_model_cosine_word_count.query(beckam)

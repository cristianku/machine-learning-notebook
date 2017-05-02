import graphlab
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)
products = graphlab.SFrame('amazon_baby.gl/')
# graphlab.canvas.set_target('headless')

products = products[products['rating'] != 3]

products['sentiment'] = products['rating'] >=4

products['word_count'] = graphlab.text_analytics.count_words(products['review'])

train_data,test_data = products.random_split(.8, seed=0)

print '==============='
print '= training sentiment_model   '
print '==============='

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target='sentiment',
                                                      features=['word_count'],
                                                      validation_set=test_data)
print '==============='
print '= end of training sentiment_model   '
print '==============='

#
#
#
# giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']
#
# giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')
#
# print giraffe_reviews.head()

def positive_count(word, word_count):
    if word in word_count:
        return word_count[word]
    else:
        return 0


selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow',
                      'hate']

for word in selected_words:
   products[word] = products['word_count'].apply(lambda word_count:positive_count(word, word_count))
   print ' totale ' + word + ' trovate   ' + str(products[word].sum())

print products.head()



train_data,test_data = products.random_split(.8, seed=0)

print '==============='
print '= training selected_words_model   '
print '==============='

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                      target='sentiment',
                                                      features=selected_words,
                                                      validation_set=test_data)
print '==============='
print '= end of training selected_words_model   '
print '==============='

print '==============='
print 'coefficients  '
print '==============='

print selected_words_model['coefficients'].sort('value', ascending=False).print_rows(num_rows=12, num_columns=4)

print '==============='
print '==============='
print '==============='
print 'selected_words_model.evaluate(test_data) '
print '==============='
print '==============='
print '==============='

print selected_words_model.evaluate(test_data)
print ''
print ''
print ''
print ''



print '==============='
print '==============='
print '==============='
print 'sentiment_model.evaluate(test_data) '
print '==============='
print '==============='
print '==============='

print sentiment_model.evaluate(test_data)

# In what range is the accuracy of simply predicting the majority class on the test_data
print '==============='
print 'majority : '
print '==============='

print str ( float(len(test_data[test_data['sentiment'] > 0])) / len(test_data))





diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type='probability')
diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)

print diaper_champ_reviews.head()


predicted_sentiment_for_most_positive_review = selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')


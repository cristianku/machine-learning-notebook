import matplotlib.pyplot as plt

import graphlab

graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)
sales = graphlab.SFrame('home_data.gl/')

train_data,test_data = sales.random_split(.8,seed=0)


sqft_model = graphlab.linear_regression.create(train_data, target='price', features=['sqft_living'],validation_set=None)


sudo -H pip install sklearn
# sudo -H pip install lstm_predictor
sudo -H pip install keras
sudo -H pip install readline
# doesnt work on windows



#Could not find a version that satisfies the requirement lstm_predictor (from versions: )
#No matching distribution found for lstm_predictor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_predictor import generate_data, load_csvdata, lstm_model


LOG_DIR = './ops_logs'
TIMESTEPS = 10
RNN_LAYERS = [{'steps': TIMESTEPS}]
DENSE_LAYERS = [10, 10]
TRAINING_STEPS = 100000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')

rawdata =  pd.read_csv("./TotalLoad.csv",parse_dates={'timeline':['date', '(UTC)']},index_col='timeline', date_parser=dateparse)

rawdata =  pd.read_csv("./TotalLoad.csv",index_col=['date', '(UTC)'])

# parsing errors in pandas


df = pd.read_csv("./TotalLoad.csv")
df

X, y = load_csvdata(rawdata, TIMESTEPS, seperate=False)

regressor = learn.TensorFlowEstimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),n_classes=0,verbose=1,steps=TRAINING_STEPS,optimizer='Adagrad',learning_rate=0.03, batch_size=BATCH_SIZE)

regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),n_classes=0,verbose=1,steps=TRAINING_STEPS,optimizer='Adagrad',learning_rate=0.03, batch_size=BATCH_SIZE)

https://www.tensorflow.org/api_guides/python/contrib.learn

regressor = learn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS))

WARNING:tensorflow:Using temporary folder as model directory: C:\Users\aroug\AppData\Local\Temp\tmp3nwb8sg9
WARNING:tensorflow:Using temporary folder as model directory: C:\Users\aroug\AppData\Local\Temp\tmp3nwb8sg9
INFO:tensorflow:Using default config.
INFO:tensorflow:Using default config.
INFO:tensorflow:Using config: {'_keep_checkpoint_max': 5, '_task_type': None, '_master': '', '_tf_config': gpu_options {
  per_process_gpu_memory_fraction: 1
}
, '_environment': 'local', '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x0000014792A352E8>, '_save_checkpoints_secs': 600, '_task_id': 0, '_keep_checkpoint_ev
ery_n_hours': 10000, '_evaluation_master': '', '_tf_random_seed': None, '_num_ps_replicas': 0, '_save_checkpoints_steps': None, '_save_summary_steps': 100, '_is_chief': True}
INFO:tensorflow:Using config: {'_keep_checkpoint_max': 5, '_task_type': None, '_master': '', '_tf_config': gpu_options {
  per_process_gpu_memory_fraction: 1
}
, '_environment': 'local', '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x0000014792A352E8>, '_save_checkpoints_secs': 600, '_task_id': 0, '_keep_checkpoint_ev
ery_n_hours': 10000, '_evaluation_master': '', '_tf_random_seed': None, '_num_ps_replicas': 0, '_save_checkpoints_steps': None, '_save_summary_steps': 100, '_is_chief': True}
>>>


++++++++++++++++
import numpy
import pandas
import tkinter
import matplotlib.pyplot as plt

dataset = pandas.read_csv('./TotalLoad.csv', usecols=[2], engine='python', skipfooter=3)
dataset 
plt.plot(dataset)
plt.show()

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


dataframe = pandas.read_csv('./TotalLoad.csv', usecols=[2], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
	
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1): \
	dataX, dataY = [], [] \
	for i in range(len(dataset)-look_back-1): \
		a = dataset[i:(i+look_back), 0] \
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0]) \
	return numpy.array(dataX), numpy.array(dataY) 


	#readline fails on Windows
	
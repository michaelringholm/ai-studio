import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import LambdaCallback, LearningRateScheduler
import keras
import csv
from datetime import datetime, timedelta
import random 
import json
from keras import backend as KerasBackend
from keras import layers
import modules.om_logging as oml
import datetime as dt
import modules.om_model_callback as ommc
import modules.om_observer as omo
import modules.om_data_loader as omdl
import modules.om_hyper_params as omhp

#region functions
def save_model_as_hd5_and_json(model):
    # save architecture 
    model_json = model.to_json()
    with open("model-architecture.json", "w") as json_file:
        json_file.write(model_json)

    # save weights
    model.save_weights("model-weights.h5")

    # load architecture and weights back
    from keras.models import model_from_json
    with open("model-architecture.json") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("model-weights.h5")

def save_model_as_hd5(model):
    # Save and load model
    model.save('model.h5')
    model = tf.keras.models.load_model('model.h5')    

def generate_synthetic_data(num_rows, data_file):
  with open(data_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['date', 'open', 'high', 'low', 'close', 'volume', 'future_quote'])
    current_date = datetime(2022, 1, 1)
    for i in range(num_rows):
      open_price = random.randint(1, 5) #random.uniform(1, 10)
      high_price = open_price + random.randint(0, 3) #open_price + random.uniform(0, 10)  
      low_price = open_price - random.randint(0, 3) #open_price - random.uniform(0, 10)
      close_price = open_price + random.randint(-3, 3) # open_price + random.uniform(-3, 3)
      volume=random.randint(1, 5)#random.randint(500, 9999999)
      future_quote=random.randint(1, 10) #round(random.uniform(1.00, 15.99), 2)
      row = [current_date.strftime('%Y-%m-%d'), 
             open_price, high_price, low_price, close_price,
             volume, future_quote]
      writer.writerow(row)  
      current_date += timedelta(days=random.randint(-180, 180))

def load_data_new():
    # Keep only relevant columns
    data = data[['date', 'open', 'high', 'low', 'close', 'future_quote']]
    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])
    # Set date as index
    data.set_index('date', inplace=True)
    # Scale input features 
    scaler = MinMaxScaler()
    data[['open', 'high', 'low', 'close']] = scaler.fit_transform(data[['open', 'high', 'low', 'close']])
    # Split data into train/test sets
    train = data[:-30]
    test = data[-30:] 
    # Reshape features for LSTM
    train_X = train[['open', 'high', 'low', 'close']].values.reshape(-1,1,4)
    test_X = test[['open', 'high', 'low', 'close']].values.reshape(-1,1,4)
    # Define model
    model = Sequential()
    model.add(LSTM(64, input_shape=(1,4)))
    model.add(Dense(1))
    # Compile and fit
    model.compile(loss='mae', optimizer='adam') 
    model.fit(train_X, train['future_quote'], epochs=100, batch_size=16, verbose=0)

def load_data_old(data_file):
    # Load data   
    data = pd.read_csv(data_file) 
    # Preprocess data
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    # Split data
    train_data, eval_data = train_test_split(data, test_size=0.2, shuffle=False)
    # Split features and target
    target_col="future_quote" # remove the column that we will predict from the input data set
    #X_train -> feature_training_data 
    #y_train -> target_evaluation_data
    #X_test -> feature_testing_data
    #y_test -> target_evaluation_data
    #feature_training_data is used to train the model
    #feature_testing_data is used to test the model performance
    #target_evaluation_data is used to evaluate the model accuracy on the training and testing sets

    train_data_features = train_data.drop(target_col, axis=1) 
    train_data_target = train_data[target_col]
    eval_data_features = eval_data.drop(target_col, axis=1)
    eval_data_target = eval_data[target_col]
    return train_data_features, train_data_target, eval_data_features, eval_data_target

def on_epoch_end(epoch, logs):
  if (epoch + 1) % 10 == 0:
    print(f"Epoch {epoch}: loss={logs['loss']/1000}K, learning rate={keras.backend.get_value(model.optimizer.lr)}")
    #print(logs)
    #print(f"model={model.optimizer.learning_rate.numpy()}")
    #print(model.optimizer.param_groups[0]['lr'])

def my_mse_loss(y_pred, y_true):
    print(f"y_pred={y_pred}, y_true={y_true}")
    squared_error = (y_pred - y_true) ** 2
    #sum_squared_error = np.sum(squared_error)
    #loss = sum_squared_error / y_true.size
    #return 1
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)

def create_model_new(num_features, num_neurons_per_layer):
    #inputs = keras.Input(shape=(num_features,), name="digits")
    inputs = keras.Input(shape=(num_features,))
    x = layers.Dense(num_neurons_per_layer, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(num_neurons_per_layer, activation="relu", name="dense_2")(x)
    #x = layers.Dense(32, activation="relu")(inputs)
    #x = layers.Dropout(0.2)(x)
    #x = layers.Dense(32, activation="relu")(x)
    #outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    x = layers.Dense(32, activation="relu")(inputs) 
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_model_old():
   # Define, compile and train model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(1)
    ])
    return model
   
def getTimePrefix():
    now = dt.datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3] 

def create_sequential_model(train_data_features,first_layer_neurons:int=30,hidden_layers:int=2,output_nodes:int=2):
    model = keras.models.Sequential([
        keras.layers.Dense(first_layer_neurons, activation="relu", input_shape=train_data_features.shape[1:]), # train_data_features.shape[1:] slices off the first element of the shape tuple, which is typically the number of samples.
        keras.layers.Dense(1) # The second layer has 1 node for output.
    ])
    return model

#endregion functions

def train_model(hyper_parameters:omhp.OMHyperParameters,observer:omo.OMObserver,modelCallback:ommc.OMModelCallback):
    oml.debug("train_model called!")
    #train_data_features, train_data_target, eval_data_features, eval_data_target=load_data_old(synthetic_data_file)
    data_loader=omdl.OMDataLoader()
    #train_data_features, train_data_target, eval_data_features, eval_data_target=data_loader.load_trade_data()
    train_data_features, train_data_target, eval_data_features, eval_data_target=data_loader.load_mnist_housing_data()
    observer.observe(observer.DATA_LOADED_EVENT, args=(train_data_features, train_data_target, eval_data_features, eval_data_target))
    #train_data_features, train_data_target, eval_data_features, eval_data_target=load_mnist_housing_data()
    #exit(1)

    # Initial learning rate
    #initial_lr = KerasBackend.eval(model.optimizer.lr)
    #print(f"Initial Learning Rate: {initial_lr}")

    # Initial learning rate
    initial_lr = hyper_parameters.learning_rate #1e-3 #0.00001
    # Set up a LearningRateScheduler using the cosine annealing function
    # Set up a cosine decay learning rate schedule
    #lr_scheduler = keras.optimizers.schedules.CosineDecay(initial_lr, decay_steps=50) # 50
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=10, decay_rate=0.95)
    #lr_scheduler = LearningRateScheduler(lambda epoch, lr: cosine_annealing(epoch, initial_lr))
    #optimizer = keras.optimizers.Adam(lr=initial_lr)
    optimizer = keras.optimizers.Adam(lr_scheduler)
    #model.compile(loss='mse', optimizer=optimizer)
    #model.compile(loss=my_mse_loss, optimizer=optimizer)
    #model.compile(loss='mean_squared_error', optimizer='adam')

    #model=create_model_new(num_features=8, num_neurons_per_layer=16)
    model=create_sequential_model(train_data_features=train_data_features,first_layer_neurons=hyper_parameters.first_layer_neurons,hidden_layers=hyper_parameters.hidden_layers,output_nodes=hyper_parameters.output_nodes)
    model.compile(loss=hyper_parameters.loss_function, optimizer=hyper_parameters.optimizer)
    observer.observe(observer.MODEL_COMPILE_DONE_EVENT,args=(model))
    #model.compile(loss='mean_squared_error', optimizer=keras.optimizers.RMSprop())

    #model.compile(
    #   optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
    #  loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
    # metrics=[keras.metrics.SparseCategoricalAccuracy()],
    #)
    KerasBackend.set_value(model.optimizer.learning_rate, initial_lr)
    batch_size=hyper_parameters.batch_size #64 # org:32
    num_epochs=hyper_parameters.num_epochs  # Adjust this based on your requirements
    #observer.observe(observer.HYPER_PARAMS_SET_EVENT, args=(batch_size,num_epochs))
    # Calculate the number of steps per epoch
    steps_per_epoch = len(train_data_features) // batch_size
    oml.debug(f"steps_per_epoch={steps_per_epoch}")
    #print(model.fit)
    #model.fit(train_data_features, train_data_target, epochs=num_epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch, verbose=0,callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
    #oml.debug(f"features={train_data_features}")
    #oml.debug(f"target={train_data_target}")
    history = model.fit(
        train_data_features,
        train_data_target,
        #batch_size=64,
        epochs=num_epochs,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(eval_data_features, eval_data_target),
        verbose=False,
        callbacks=[modelCallback]
    )
    observer.observe("model_fit_done", args=(history,model))
    # Evaluate model 
    #loss, accuracy = model.evaluate(feature_testing_data, target_evaluation_data)
    loss = model.evaluate(eval_data_features, eval_data_target)
    observer.observe("model_eval_done", args=(loss,model))
    #oml.debug(f'Test loss: {loss}')
    #observer("test_loss", loss)
    #print('last_lr:', lr_scheduler.get_last_lr())
    #model.summary()

    # Train model
    #model.fit(train_data, test_data, epochs=100, verbose=0) 

    #save_model_as_hd5(model)
    save_model_as_hd5_and_json(model)
    #predictions = model.predict(model, X_test)
    #print(f"predictions={predictions}")



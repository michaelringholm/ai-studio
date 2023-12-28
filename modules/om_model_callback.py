import keras
import modules.om_logging as oml
import modules.om_observer as omo

class OMModelCallback(keras.callbacks.Callback):
    def __init__(s,observer:omo.OMObserver):
        s.observer=observer
        return
    
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("Starting training; got log keys: {}".format(keys))
        oml.debug(f"on_test_batch_end. Keys={keys}. Values={values}")

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("Stop training; got log keys: {}".format(keys))
        oml.debug(f"on_test_batch_end. Keys={keys}. Values={values}")

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
        oml.debug(f"on_epoch_begin. Epoch={epoch} ended. Keys={keys}. Values={values}")

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        oml.debug(f"on_epoch_end. Epoch={epoch} ended. Keys={keys}. Values={values}")
        self.observer.observe("on_epoch_end", args=(epoch,logs))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("Start testing; got log keys: {}".format(keys))
        oml.debug(f"on_test_begin. Keys={keys}. Values={values}")

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("Stop testing; got log keys: {}".format(keys))
        oml.debug(f"on_test_batch_end. Keys={keys}. Values={values}")

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("Start predicting; got log keys: {}".format(keys))
        oml.debug(f"on_predict_begin. Keys={keys}. Values={values}")

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("Stop predicting; got log keys: {}".format(keys))
        oml.debug(f"on_predict_end. Keys={keys}. Values={values}")

    def on_train_batch_begin(self, batch, logs=None):
        #keys = list(logs.keys())
        #print("...Training: start of batch {}; got log keys: {}".format(batch, keys))
        keys = list(logs.keys())
        values=list(logs.values())
        #print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        oml.debug(f"on_train_batch_begin. batch={batch}. Keys={keys}. Values={values}")

    def on_train_batch_end(self, batch, logs=None):
        #keys = list(logs.keys())
        #print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
        keys = list(logs.keys())
        values=list(logs.values())
        #print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        oml.debug(f"on_train_batch_end. batch={batch}. Keys={keys}. Values={values}")

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))
        oml.debug(f"on_test_batch_begin. batch={batch}. Keys={keys}. Values={values}")

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))
        oml.debug(f"on_test_batch_end. batch={batch}. Keys={keys}. Values={values}")

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))
        oml.debug(f"on_predict_batch_begin. batch={batch}. Keys={keys}. Values={values}")

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        values=list(logs.values())
        #print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
        oml.debug(f"on_predict_batch_end. batch={batch}. Keys={keys}. Values={values}")
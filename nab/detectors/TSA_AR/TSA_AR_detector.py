from nab.detectors.base import AnomalyDetector
import numpy as np
from nab.detectors.TSA_AR.AR import AR
import sys
import time

class TSA_ARDetector(AnomalyDetector):

  def __init__(self, *args, **kwargs):

    super(TSA_ARDetector, self).__init__(*args, **kwargs)
    self.detector_name = "TSA_AR"
    self.parameter_training_epochs = 40
    self.retrain_parapeters_interval = 20
    self.count = 0
    self.values = []
    self.predictions = []
    self.ar_model = None

  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore).
    The anomalyScore is simply a random value from 0 to 1
    """
    self.values.append(inputData['value'])
    self.ar_model.set_training_data_set(self.values)

    if self.count % self.retrain_parapeters_interval:
      mse = np.zeros(self.parameter_training_epochs-1)
      for i in range(1, self.parameter_training_epochs):
          self.ar_model.set_p(i)
          self.ar_model.calculate_weights()
          
          predictions = self.ar_model.get_prediction(self.ar_model.training_data_set)
          mse[i-1] = self.ar_model.get_mse(self.ar_model.training_data_set, predictions)

      best_p = int(np.argmin(mse))+1
      self.log("New MSE is minimum at P at count "+str(self.count)+" = "+str(best_p))
      sys.stdout.write("\033[F")
      self.ar_model.set_p(best_p)
  
    self.ar_model.calculate_weights()
    prediction = self.ar_model.get_single_prediction(self.ar_model.training_data_set,self.count)
    self.predictions.append(prediction)
    anomalyScore = self.ar_model.get_anomaly_scores(self.ar_model.training_data_set, self.predictions)[-1]
    
    self.count += 1
    self.log(str(self.count))
    sys.stdout.write("\033[F")
    return (anomalyScore, )

  def initialize(self):
    self.log("Initialzing started")
    values = self.dataSet.data["value"]
    self.log("probationaryPeriod(Training Set) : "+str(int(self.probationaryPeriod)))
    training_data_set = np.array(values[0:int(self.probationaryPeriod)])
    validation_data_set = np.array(values)

    ar_model = AR(1)
    ar_model.set_training_data_set(training_data_set)
    ar_model.set_validation_data_set(validation_data_set)
    
    mse = np.zeros(self.parameter_training_epochs-1)
    for i in range(1, self.parameter_training_epochs):
        ar_model.set_p(i)
        ar_model.calculate_weights()
        
        predictions = ar_model.get_prediction(ar_model.training_data_set)
        mse[i-1] = ar_model.get_mse(ar_model.training_data_set, predictions)

    best_p = int(np.argmin(mse))+1
    self.log("MSE is minimum at P = "+str(best_p))
    ar_model.set_p(best_p)

    self.ar_model = ar_model
    self.count = 0
    self.values = []
    self.predictions = []
    self.log("Initialzing ended")
  
  def log(self, message):
    print("%s:[%s]:%s: %s" % (self.threadId, self.detector_name, time.ctime(),message))
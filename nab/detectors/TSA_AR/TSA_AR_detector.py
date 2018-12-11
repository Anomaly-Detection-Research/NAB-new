from nab.detectors.base import AnomalyDetector
import numpy as np
from nab.detectors.TSA_AR.AR import AR

class TSA_ARDetector(AnomalyDetector):

  def __init__(self, *args, **kwargs):

    super(TSA_ARDetector, self).__init__(*args, **kwargs)
    self.count = 0
    self.anomaly_scores = None
    self.ar_model = None
    self.input_data = []
    self.predictions = []

  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore).
    The anomalyScore is simply a random value from 0 to 1
    """

    self.input_data.append(inputData['value'])

    self.ar_model.set_training_data_set(self.input_data)
    self.ar_model.calculate_weights()
    
    prediction = self.ar_model.get_single_prediction(self.input_data,self.count)
    self.predictions.append(prediction)
    
    # if self.count == 0 :
    #   print(self.input_data)
    #   print(self.predictions)
    
    anomalyScore = self.ar_model.get_anomaly_scores(self.input_data, self.predictions)[-1]
    
    # # anomalyScore = self.anomaly_scores[self.count]
    self.count += 1
    # anomalyScore = 0.0
    print(self.count)
    return (anomalyScore, )

  def initialize(self):
    values = self.dataSet.data["value"]
    
    print("probationaryPeriod : "+str(int(self.probationaryPeriod)))
    training_set_values = np.array(values[0:int(self.probationaryPeriod)])
    validation_set_values = np.array(values)
    
    ar_model = AR(1)
    ar_model.set_training_data_set(training_set_values)
    ar_model.set_validation_data_set(validation_set_values)

    epochs = 30
    mse = np.zeros(epochs-1)
    for i in range(1, epochs):
        ar_model.set_p(i)
        ar_model.calculate_weights()
        prediction = ar_model.get_prediction(ar_model.training_data)
        mse[i-1] = ar_model.get_mse(ar_model.training_data, prediction)
    
    best_p = int(np.argmin(mse))+1
    # print(mse)
    print("MSE is minimum at P = "+str(best_p))
    ar_model.set_p(best_p)
    # ar_model.calculate_weights()
    # print("Making AR prediction ...")
    # prediction = ar_model.get_prediction(ar_model.validation_data_set)
    # print("Calculating AR Anomaly Score ...")
    # anomaly_scores = ar_model.get_anomaly_scores(ar_model.validation_data_set, prediction)

    self.ar_model = ar_model
    # self.anomaly_scores = anomaly_scores
    self.count = 0
    self.input_data = []
    self.predictions = []
    print("Done")
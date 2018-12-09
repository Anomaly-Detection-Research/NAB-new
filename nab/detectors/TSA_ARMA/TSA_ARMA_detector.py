from nab.detectors.base import AnomalyDetector
import numpy as np
from nab.detectors.TSA_ARMA.ARMA import ARMA

class TSA_ARMADetector(AnomalyDetector):

  def __init__(self, *args, **kwargs):

    super(TSA_ARMADetector, self).__init__(*args, **kwargs)
    self.count = 0
    self.anomaly_scores = None


  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore).
    The anomalyScore is simply a random value from 0 to 1
    """
    anomalyScore = self.anomaly_scores[self.count]
    self.count += 1
    # print("handle Record : "+str(self.count) )
    # print(inputData)
    return (anomalyScore, )


  def initialize(self):
    # values = self.dataSet.data[list(self.dataSet.data)[1]]
    values = self.dataSet.data["value"]
    training_set_values = np.array(values[0:int(self.probationaryPeriod)])
    # validation_set_values = np.array(values[int(self.probationaryPeriod):-1])
    validation_set_values = np.array(values)
    
    arma_model = ARMA(1,1)
    arma_model.set_training_data_set(training_set_values)
    arma_model.set_validation_data_set(validation_set_values)


    epochs = 10
    mse = np.zeros((epochs-1,epochs-1))
    mse_x = np.arange(1, epochs)
    mse_y = np.arange(1, epochs)


    for i in range(1, epochs):
      arma_model.set_p(i)
      for j in range(1,epochs):
          arma_model.set_q(j)
          
          prediction = arma_model.get_prediction(arma_model.training_data)
          mse[i-1][j-1] = arma_model.get_mse(arma_model.training_data, prediction)

          print("running training data set on arma("+str(i)+","+str(j)+")")

    min_i = 0
    min_j = 0
    minimum = mse[0][0]
    for i in range(0,mse.shape[0]):
      for j in range(0, mse.shape[1]):
        if mse[i][j] < minimum:
                min_i = i
                min_j = j
                minimum = mse[i][j]

    print(mse)
    print("MSE is minimum at P = "+str(min_i+1)+" and Q = "+str(min_j+1))
    min_p = min_i+1
    min_q = min_j+1
    arma_model.set_p(min_p)
    arma_model.set_q(min_q)

    prediction = arma_model.get_prediction(arma_model.validation_data_set)
    anomaly_scores = arma_model.get_anomaly_scores(arma_model.validation_data_set, prediction)

    self.anomaly_scores = anomaly_scores
    self.count = 0
    print("Done")
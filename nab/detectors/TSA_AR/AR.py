import warnings
import itertools
import pandas
import math
import sys
import numpy as np


class AR:
    def __init__(self, p):
        self.p = p
    
    # Setters
    def set_p(self,p):
        self.p=p 
        return 0
    
    def set_training_data_time(self, time):
        self.training_data_time = time
    
    def set_validation_data_time(self, time):
        self.validation_data_time = time
        
    def set_testing_data_time(self, time):
        self.testing_data_time = time
    
    def set_validation_data_set(self,data):
        self.validation_data_set = data
        
    def set_testing_data_set(self,data):
        self.testing_data_set = data
    
    def set_training_data_set(self,data):
        self.training_data = data
        self.training_data_mean = np.mean(data)
        self.training_data_std = np.std(data, ddof=1)
        self.Z = data - self.training_data_mean
        self.Z.shape = (len(data),1)
        self.Z_mean = np.mean(self.Z)
        self.Z_std = np.std(self.Z, ddof=1)
        return 0
    
    # Model
    def shock(self):
#         return np.random.normal(self.Z_mean, self.Z_std, 1)
        return 1
    
    def calculate_normal_matrix_x_row(self,data,t):
        row = np.zeros((1,self.p+1))
        j = 0
        for i in range(t-self.p,t):
            if i < 0:
                row[0][j] = 0
            else:
                row[0][j] = data[i]
            j+=1
        row[0][-1] = self.shock()
        return row
    
    def calculate_weights(self):
        normal_matrix = np.zeros((len(self.training_data),self.p+1))
        
        for i in range(0,len(self.training_data)):
            normal_matrix[i] = self.calculate_normal_matrix_x_row(self.Z,i)
        
        normal_matrix_tanspose = normal_matrix.transpose()
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(normal_matrix_tanspose,normal_matrix)),normal_matrix_tanspose),self.Z)
        return 0
        
    def get_prediction(self,data_set):
        self.prediction = np.zeros((np.max(data_set.shape),1))
        Z = data_set - np.mean(data_set)
        Z.shape = (np.max(data_set.shape),1)
        for i in range(0,np.max(data_set.shape)):
            self.prediction[i] = np.dot(self.calculate_normal_matrix_x_row(Z, i), self.weights)
    
        self.prediction = self.prediction.transpose()[0] + np.mean(data_set)
        return self.prediction
    
    # Diagnostics and identification messures
    def mse(self,values,pridicted):
        error = 0.0
        for i in range(0,len(values)):
            error += (values[i] - pridicted[i])**2
        return error/len(values)
    
    def get_mse(self, data, prediction):
        return self.mse(data,prediction)
    
    def plot_autocorrelation(self, data_set, lag):
        autocorrelations = np.zeros(lag)
        autocorrelations_x = np.arange(lag)
        autocorrelations[0] = 1.0
        for i in range(1,lag):
            autocorrelations[i] = np.corrcoef(data_set[i:],data_set[:-i])[0,1]
        
        trace = {"x": autocorrelations_x,
                 "y": autocorrelations,
                 'type': 'bar',
                 "name": 'Autocorrelation',         
                }
        
        traces = [trace]
        layout = dict(title = "Autocorrelation",
                  xaxis = dict(title = 'Lag'),
                  yaxis = dict(title = 'Autocorrelation')
                 )
        fig = dict(data=traces, layout=layout)
        # iplot(fig)
    
    def plot_partial_autocorrelation(self, data_set, lag):
        pac = np.zeros(lag)
        pac_x = np.arange(lag)
        
        residualts = data_set
        slope, intercept = np.polyfit(data_set,residualts,1)
        estimate = intercept + slope*data_set
        residualts = residualts - estimate
        pac[0] = 1
        for i in range(1,lag):
            pac[i] = np.corrcoef(data_set[:-i],residualts[i:])[0,1]
            
            slope, intercept = np.polyfit(data_set[:-i],residualts[i:],1)
            estimate = intercept + slope*data_set[:-i]
            
            residualts[i:] = residualts[i:] - estimate
        
        trace = {"x": pac_x,
                 "y": pac,
                 'type': 'bar',
                 "name": 'Partial Autocorrelation',         
                }

        traces = [trace]
        layout = dict(title = "Partial Autocorrelation",
                  xaxis = dict(title = 'Lag'),
                  yaxis = dict(title = 'Partial Autocorrelation')
                 )
        fig = dict(data=traces, layout=layout)
        # iplot(fig)
    
    def plot_residuals(self, data_set, prediction):
        x = np.arange(len(data_set))
        residual = data_set - prediction
        mean = np.ones(len(data_set))*np.mean(residual)
        
        trace = {"x": x,
                 "y": residual,
                 "mode": 'markers',
                 "name": 'Residual'}

        trace_mean = {"x": x,
                     "y": mean,
                     "mode": 'lines',
                     "name": 'Mean'}
        traces = [trace,trace_mean]
        layout = dict(title = "Residual",
                      xaxis = dict(title = 'X'),
                      yaxis = dict(title = 'Residual')
                     )
        fig = dict(data=traces, layout=layout)
        # iplot(fig)
        print("Standard Deviation of Residuals : " + str(np.std(residual, ddof=1)))
        print("Mean of Residuals : " + str(np.mean(residual)))
    
    def plot_data(self, data_set, time):
        mean = np.mean(data_set)
        means = np.ones(len(data_set))*mean
        trace_value = {"x": time,
                     "y": data_set,
                     "mode": 'lines',
                     "name": 'value'}

        trace_mean = {"x": time,
                         "y": means,
                         "mode": 'lines',
                         "name": 'mean'}
        traces = [trace_value,trace_mean]
        layout = dict(title = "Values with mean",
                      xaxis = dict(title = 'Time'),
                      yaxis = dict(title = 'Value')
                     )
        fig = dict(data=traces, layout=layout)
        iplot(fig)
        
        normalized_data = data_set - mean
        trace_value = {"x": time,
                     "y": normalized_data,
                     "mode": 'lines',
                     "name": 'value'}
        traces = [trace_value]
        layout = dict(title = "After removing mean",
                      xaxis = dict(title = 'Time'),
                      yaxis = dict(title = 'Value')
                     )
        fig = dict(data=traces, layout=layout)
        # iplot(fig)
    
    def print_stats(self,data,prediction):
        print("Mean Square Error : " + str(self.mse(data,prediction)))
        print("Mean of real values : " + str(np.mean(data)))
        print("Standard Deviation of real values : " + str(np.std(data, ddof=1)))
        print("Mean of predicted values : " + str(np.mean(prediction)))
        print("Standard Deviation of predicted values : " + str(np.std(prediction, ddof=1)))
        print("Number of data points : " + str(len(data)))
    
    def plot_result(self, time, data, prediction):
        trace_real = {"x": time,
                     "y": data,
                     "mode": 'lines',
                     "name": 'Real value'}

        trace_predicted = {"x": time,
                         "y": prediction,
                         "mode": 'lines',
                         "name": 'Predicted value'}
        traces = [trace_real,trace_predicted]
        layout = dict(title = "Training Data Set with AR("+str(self.p)+")",
                      xaxis = dict(title = 'Time'),
                      yaxis = dict(title = 'Value')
                     )
        fig = dict(data=traces, layout=layout)
        # iplot(fig)
        self.print_stats(data,prediction)
        self.plot_residuals(data,prediction)
    
    def get_anomaly_scores(self, data, prediction):
        mse = np.zeros(len(prediction))
        for i in range(0, len(prediction)):
            mse_value = data[i]-prediction[i]
            if mse_value < 0:
                mse_value = -1*mse_value
            mse[i] = mse_value
        max_mse = np.max(mse)
        return mse/max_mse
    
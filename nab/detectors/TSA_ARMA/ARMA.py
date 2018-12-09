#ref: https://www.youtube.com/watch?v=IcxMywGiWUc
import warnings
import itertools
import pandas
import math
import sys
import numpy as np

class ARMA:
    def __init__(self, p,q):
        self.p = p
        self.q = q
    
    # Setters
    def set_p(self,p):
        self.p=p 
    
    def set_q(self,q):
        self.q=q
    
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
        self.Z = data - np.mean(data)
        self.Z.shape = (len(data),1)
    
    # Model
    def shock(self,mean,std):
        return np.random.normal(mean, std, 1)
#         return 0
    
    def calculate_AR_normal_matrix_x_row(self,data,t,mean,std):
        row = np.zeros((1,self.p+1))
        j = 0
        for i in range(t-self.p,t):
            if i < 0:
                row[0][j] = 0
            else:
                row[0][j] = data[i]
            j+=1
        row[0][-1] = self.shock(mean,std)
        return row
    
    def calculate_AR_weights(self):
        normal_matrix = np.zeros((len(self.training_data),self.p+1))
        mean = np.mean(self.Z)
        std = np.std(self.Z, ddof=1)
        for i in range(0,len(self.training_data)):
            normal_matrix[i] = self.calculate_AR_normal_matrix_x_row(self.Z,i,mean,std)
        
        normal_matrix_tanspose = normal_matrix.transpose()
        self.AR_weights = np.dot(np.dot(np.linalg.pinv(np.dot(normal_matrix_tanspose,normal_matrix)),normal_matrix_tanspose),self.Z)

        
    def get_AR_prediction(self,data_set):
        self.calculate_AR_weights()
        self.AR_prediction = np.zeros((np.max(data_set.shape),1))
        mean = np.mean(data_set)
        std = np.std(data_set, ddof=1)
        Z = np.array(data_set)
        Z.shape = (np.max(Z.shape),1)
        Z = Z - mean
        for i in range(0,np.max(Z.shape)):
            self.AR_prediction[i] = np.dot(self.calculate_AR_normal_matrix_x_row(Z, i, mean, std), self.AR_weights)
        
        self.AR_prediction = self.AR_prediction.transpose()[0] + mean
        return self.AR_prediction
    
    def get_previous_q_values(self,data,t):
        previous_q = np.zeros(self.q)
        j = 0
        for i in range(t-self.q,t):
            if i < 0:
                previous_q[j] = 0
            else:
                previous_q[j] = data[i]
            j+=1
        return previous_q
    
    def get_MA_prediction(self,data_set):
        self.MA_prediction = np.zeros(np.max(data_set.shape))
        Z = np.array(data_set)
        Z.shape = (np.max(Z.shape),1)
        for i in range(0,np.max(Z.shape)):
            self.MA_prediction[i] = np.average(self.get_previous_q_values(Z, i))
        
        return self.MA_prediction
    
    def calculate_AR_MA_normal_matrix_x_row(self,t):
        row = np.zeros((1,2))
        row[0][0] = self.MA_prediction[t]
        row[0][1] = self.AR_prediction[t]
        return row
    
    def calculate_AR_MA_weights(self):
        self.get_MA_prediction(self.training_data)
        self.get_AR_prediction(self.training_data)
        normal_matrix = np.zeros((len(self.training_data),2))
        
        for i in range(0,len(self.training_data)):
            normal_matrix[i] = self.calculate_AR_MA_normal_matrix_x_row(i)
        
        normal_matrix_tanspose = normal_matrix.transpose()
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(normal_matrix_tanspose,normal_matrix)),normal_matrix_tanspose),self.training_data)
        
#         print(self.weights)
#         #normalizing weigts
#         total = self.weights[0] + self.weights[1]
#         self.weights[0] = self.weights[0]/total
#         self.weights[1] = self.weights[1]/total
#         print(self.weights)
        
    def get_prediction(self, data_set):
        self.calculate_AR_MA_weights()
        
        self.get_MA_prediction(data_set)
        self.get_AR_prediction(data_set)
        Z = np.array(data_set)
        Z.shape = (np.max(Z.shape),1)
        self.prediction = np.zeros((np.max(Z.shape),1))
        for i in range(0,np.max(Z.shape)):
            self.prediction[i] = np.dot(self.calculate_AR_MA_normal_matrix_x_row(i), self.weights)
        
        self.prediction = self.prediction.transpose()[0]
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
        iplot(fig)
    
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
        iplot(fig)
    
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
        iplot(fig)
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
        iplot(fig)
    
    
    def print_stats(self,data,prediction):
        print("Mean Square Error : " + str(self.mse(data,prediction)))
        print("Mean of real values : " + str(np.mean(data)))
        print("Standard Deviation of real values : " + str(np.std(data, ddof=1)))
        print("Mean of predicted values : " + str(np.mean(prediction)))
        print("Standard Deviation of predicted values : " + str(np.std(prediction, ddof=1)))
        print("Number of data points : " + str(len(data)))
    
    def plot_result(self, time, data, prediction):
        data.shape = (1,np.max(data.shape))
        data = data[0]
        trace_real = {"x": time,
                     "y": data,
                     "mode": 'lines',
                     "name": 'Real value'}
        trace_predicted = {"x": time,
                         "y": prediction,
                         "mode": 'lines',
                         "name": 'Predicted value'}
        traces = [trace_real,trace_predicted]
        layout = dict(title = "Training Data Set with ARMA("+str(self.p)+","+str(self.q)+")",
                      xaxis = dict(title = 'Time'),
                      yaxis = dict(title = 'Value')
                     )
        fig = dict(data=traces, layout=layout)
        iplot(fig)
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
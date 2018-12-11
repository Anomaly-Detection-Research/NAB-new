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
    
    def get_single_prediction(self,data_set,time):
        mean = np.mean(data_set)
        Z = data_set - mean
        prediction = np.dot(self.calculate_normal_matrix_x_row(Z, time), self.weights)
        prediction = prediction[0][0]
        prediction = prediction + mean
        return prediction

    # Diagnostics and identification messures
    def get_mse(self, values, prediction):
        error = 0.0
        for i in range(0,len(values)):
            error += (values[i] - prediction[i])**2
        return error/len(values)

    def get_anomaly_scores(self, data, prediction):
        mse = np.zeros(len(prediction))
        for i in range(0, len(prediction)):
            mse_value = data[i]-prediction[i]
            if mse_value < 0:
                mse_value = -1*mse_value
            mse[i] = mse_value
        max_mse = np.max(mse)
        if max_mse == 0:
            return mse
        return mse/max_mse

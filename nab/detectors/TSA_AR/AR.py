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
    
    def set_validation_data_set(self,data):
        if hasattr(data,'shape'):
            size = np.max(data.shape)
        else:
            size = len(data)
        self.validation_data_set = np.array(data)
        self.validation_data_set.shape = (size,1)
        self.normalized_valitaion_data_set = self.validation_data_set - np.mean(self.validation_data_set)
        
    def set_testing_data_set(self,data):
        if hasattr(data,'shape'):
            size = np.max(data.shape)
        else:
            size = len(data)
        self.testing_data_set = np.array(data)
        self.testing_data_set.shape = (size,1)
        self.normalized_testing_data_set = self.testing_data_set - np.mean(self.testing_data_set) 
    
    def set_training_data_set(self,data):
        if hasattr(data,'shape'):
            size = np.max(data.shape)
        else:
            size = len(data)
        self.training_data_set = np.array(data)
        self.training_data_set.shape = (size,1)
        self.normalized_training_data_set = self.training_data_set - np.mean(self.training_data_set)
    
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
        normal_matrix = np.zeros((np.max(self.training_data_set.shape),self.p+1))
        
        for i in range(0,np.max(self.training_data_set.shape)):
            normal_matrix[i] = self.calculate_normal_matrix_x_row(self.normalized_training_data_set,i)
        
        normal_matrix_tanspose = normal_matrix.transpose()
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(normal_matrix_tanspose,normal_matrix)),normal_matrix_tanspose),self.normalized_training_data_set)
        
    def get_prediction(self,data):
        if hasattr(data,'shape'):
            size = np.max(data.shape)
        else:
            size = len(data)
            data = np.array(data)
        self.predictions = np.zeros((size,1))
        data.shape = (size,1)
        data_mean = np.mean(data)
        
        normalized_data = data - data_mean
        
        for i in range(0,size):
            self.predictions[i] = np.dot(self.calculate_normal_matrix_x_row(normalized_data, i), self.weights)
        
        self.predictions = self.predictions.transpose()[0] + data_mean
        return self.predictions
    
    def get_single_prediction(self,data,time):
        if hasattr(data,'shape'):
            size = np.max(data.shape)
        else:
            size = len(data)
            data = np.array(data)
        data.shape = (size,1)
        data_mean = np.mean(data)
        normalized_data = data - data_mean
        self.prediction = np.dot(self.calculate_normal_matrix_x_row(normalized_data, time), self.weights)
        
        self.prediction = self.prediction[0][0]
        self.prediction = self.prediction + data_mean
        return self.prediction

    # Diagnostics and identification messures
    def get_mse(self, data, predictions):
        if hasattr(data,'shape'):
            size = np.max(data.shape)
        else:
            size = len(data)
            data = np.array(data)
        data.shape = (size,1)
        error = 0.0
        for i in range(0,size):
            error += (data[i] - predictions[i])**2
        return error/size

    def get_anomaly_scores(self, data, predictions):
        if hasattr(predictions,'shape'):
            size = np.max(predictions.shape)
        else:
            size = len(predictions)
            predictions = np.array(predictions)
        predictions.shape = (size,1)
        anomaly_score = np.zeros(size)
        for i in range(0, size):
            anomaly_score_tmp = data[i]-predictions[i]
            if anomaly_score_tmp < 0:
                anomaly_score_tmp = -1*anomaly_score_tmp
            anomaly_score[i] = anomaly_score_tmp
        max_anomaly_score = np.max(anomaly_score)
        if max_anomaly_score == 0:
            return anomaly_score
        return anomaly_score/max_anomaly_score

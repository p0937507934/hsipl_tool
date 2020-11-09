import numpy as np


class PreProcessing:
    def __init__(self, HIM):
        self._HIM = HIM

    def data_normalize(self):
        input_data = np.array(self._HIM)*1.0
        maximum = np.max(np.max(input_data))
        minimum = np.min(np.min(input_data))
        
        normal_data = (input_data-minimum)/(maximum-minimum)*1.0
        return normal_data

    def msc(self, reference = None):
        ''' 
        Multiplicative Scatter Correction
        
        param input_data: signal, type is 2d-array, size is [signal num, band num]
        param reference: reference of msc, type is 2d-array, size is [1, band num], if reference is None, it will be calculated in the function
        '''
        for i in range(self._HIM.shape[0]):
            input_data[i,:] -= self._HIM[i,:].mean()
        if reference is None:
            ref = np.mean(input_data, axis = 0)
        else:
            ref = reference
        data_msc = np.zeros_like(input_data)
        for i in range(input_data.shape[0]):
            fit = np.polyfit(ref, input_data[i,:], 1, full=True)
            data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
        return (data_msc, ref)
    
    def snv(self):
        '''
        Standard Normal Variate
        
        param input_data: signal, type is 2d-array, size is [signal num, band num]
        '''
        data_snv = np.zeros_like(self._HIM)
        for i in range(self._HIM.shape[0]):
            data_snv[i,:] = (self._HIM[i,:] - np.mean(self._HIM[i,:])) / np.std(self._HIM[i,:])
        return data_snv

    
    
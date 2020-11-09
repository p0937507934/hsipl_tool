from .preprocessing import PreProcessing
from .target_dectection import TargetDetection
import matplotlib.pyplot as plt
import numpy as np


class HyperSpectral:

    def __init__(self, HIM, d, preprocessing=None, algo=None):
        self._HIM = HIM
        self._x, self._y, self._z = HIM.shape
        self._d = d
        self._dvalue = self._HIM[self._d[0], self._d[1], :]
        self._preprocess = []
        self._algo = []
        self._result = {}
        self._PHI = None

    @property
    def getX(self):
        return self._x

    @property
    def getY(self):
        return self._y

    @property
    def getZ(self):
        return self._z

    @property
    def getD(self):
        return self._d

    @property
    def getDvalue(self):
        return self._dvalue

    @property
    def preprocessing(self):
        return self._preprocess

    @preprocessing.setter
    def preprocessing(self, args):
        if type(args) == list:
            self._preprocess = self._preprocess + [x for x in args]
        elif type(args) == str:
            self._preprocess.append(args)

    @property
    def algo(self):
        return self._algo

    @algo.setter
    def algo(self, args):
        if type(args) == list:
            self._algo = self._algo + [x for x in args]
        elif type(args) == str:
            self._algo.append(args)

    def plot(self, strategy="origin", deafult_bands=100):

        if strategy == "origin":
            plt.figure('origin hsi')
            plt.imshow(self._HIM[:, :, deafult_bands])
       
        elif strategy == "PHI":
            plt.figure('After preprocessing hsi')
            plt.imshow(self._PHI[:, :, deafult_bands])
        
        elif strategy == "result":
            for key in self._result:

                plt.figure(str(key))
                plt.imshow(self._result[key])
        plt.show()

    def getShape(self):
        print((self._x,self._y,self._z))

    def setD(self,args):
        self._d = args
        if(len(self._preprocess != 0 and self._PHI != None )):
            self._dvalue = self._PHI[self._d[0],self._d[1],:]
        else:
            self._dvalue = self._HIM[self_d[0],self._d[1],:]
            

    def compile(self):

        if(len(self._preprocess) > 0):
            p = PreProcessing(self._HIM)
            for i in self._preprocess:
                self._PHI = getattr(p, i)()
                self._dvalue = self._PHI[self._d[0], self._d[1], :]
                p = PreProcessing(self._PHI)
            if(len(self._algo) > 0):
                t = TargetDetection(self._PHI, self._dvalue)
                for j in self._algo:
                    self._result[j] = getattr(t, j)()

            else:
                print('there is no any algorithem.')
        else:
            if(len(self._algo) > 0):
                t = TargetDetection(self._HIM, self._dvalue)
                for j in self._algo:
                    self._result[j] = getattr(t, j)()
            else:
                print('there is no any algorithem.')

from unittest import TestCase
from src.preprocessing import PreProcessing
from src.target_dectection import TargetDetection
import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi




class TestPreProcessing(TestCase):
    
    def test_datanormalize(self):  
        p = PreProcessing(im)   
        d = p.data_normalize()
        assert(im.shape == d.shape)
    
    def test_snv(self):
        a = np.linspace(1,100,224).reshape(1,224)
        p = PreProcessing(a)
        d = p.snv()
        assert(a.shape == d.shape)
    
   

class TestTargetDetection(TestCase):

    def setup(self):

        self.img = np.linspace(1,100,100*100*224).reshape(100,100,224)
        self.d =  self.img[50,50,:]

    def test_d(self):
        self.setup()
        t =TargetDetection(self.img,self.d)
        self.assertEqual(self.d.all() , t._d.all())

    def test_img(self):
        self.setup()
        t =TargetDetection(self.img,self.d)
        self.assertEqual(self.img.all() , t._HIM.all())

    def test_cem(self):
        
        self.setup()
        t =TargetDetection(self.img,self.d)
        r = t.cem()
        assert(r.shape == (100,100))
    
    def test_sam_img(self):
        self.setup()
        t =TargetDetection(self.img,self.d)
        r = t.sam_img()
        assert(r.shape == (100,100))
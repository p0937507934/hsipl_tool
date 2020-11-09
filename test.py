from hyperspectral import HyperSpectral
import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi

#only test everything work

file_path = r'D:\apple0820\apple0820\1-1_RT'
im = envi.open(file_path + '.hdr', file_path + '.raw').asarray()

hsi = HyperSpectral(im,[50,50])
s = "data_normalize"
hsi.preprocessing = ["data_normalize"]
hsi.algo = ["sam_img","cem"]
hsi.compile()
hsi.plot()
hsi.plot(strategy="PHI")
hsi.plot(strategy="result")
# print(hsi.__dict__)

# unit test

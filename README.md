# Hyperspectral image tool
A python library for simple hyperspectral processing.
And this library only for target_detection nowaday.
Ideas were from **HSIPL Lab National YunLin Techlogy University**.



# Installation

Using git clone to this repo.



# Usage

```python

from hsi_tool.src import HyperSpectral
import numpy as np
import spectral.io.envi as envi

#claim a instance,give two parameters(hyperspectral image(3darray[x,y,z]),targetD(1,bands))
h = HyperSpectral(HSI,targetD)
#add preprocess algorithem
h.preprocessing = ["data_normalize"]
#add target_detection algorithem
h.algo = ["sam_img","cem"]
#using compile to do all algorithem in one process.
h.compile()
#show result image.
h.plot()
h.plot(strategy="PHI")
h.plot(strategy="result")
```



# Contributors

Especially thank [ek2061](https://github.com/ek2061) for implement most algorithem in this library.  



-----
###  **For research use only**
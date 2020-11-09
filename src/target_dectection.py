import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi
import warnings


class TargetDetection:

    def __init__(self, img, d):
        self._HIM = img
        self._d = d

    def cem(self, R=None):
        '''
        Constrained Energy Minimization for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        d = np.reshape(self._d, [self._HIM.shape[2], 1])
        if R is None:
            R = self.calc_R()
        try:
            Rinv = np.linalg.inv(R)
        except:
            Rinv = np.linalg.pinv(R)
            warnings.warn(
                'The pseudo-inverse matrix is used instead of the inverse matrix in cem_img(), please check the input data')
        result = np.dot(np.transpose(r), np.dot(Rinv, d)) / \
            np.dot(np.transpose(d), np.dot(Rinv, d))
        result = np.reshape(result, [self._HIM.shape[0], self._HIM.shape[1]])
        return result

    def subset_cem(self, win_height=None, win_width=None):
        '''
        Subset Constrained Energy Minimization

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        param win_height: window height for subset cem, type is int
        param win_width: window width for subset cem, type is int
        '''
        if win_height is None:
            win_height = np.ceil(self._HIM.shape[0] / 3)

        if win_width is None:
            win_width = np.ceil(self._HIM.shape[1] / 3)

        if win_height > self._HIM.shape[0] or win_width > self._HIM.shape[1] or win_height < 2 or win_width < 2:
            raise ValueError('Wrong window size for subset_cem()')

        d = np.reshape(self._d, [self._HIM.shape[2], 1])
        result = np.zeros([self._HIM.shape[0], self._HIM.shape[1]])

        for i in range(0, self._HIM.shape[0], win_height):
            for j in range(0, self._HIM.shape[1], win_width):
                result[i: i + win_height, j: j +
                       win_width] = self.cem(self._HIM[i: i + win_height, j: j + win_width, :], d)
        return result

    def hcem(self, max_it=100, λ=200, e=1e-6):
        '''
        Hierarchical Constrained Energy Minimization

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        param max_it: maximum number of iterations, type is int
        param λ: coefficients for constructing a new CEM detector, type is int
        param e: stop iterating until the error is less than e, type is int
        '''
        imgH = self._HIM.shape[0]  # image height
        imgW = self._HIM.shape[1]  # image width
        N = imgH*imgW  # pixel number
        D = self._HIM.shape[2]  # band number
        Weight = np.ones([1, N])
        y_old = np.ones([1, N])

        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))

        for T in range(max_it):
            for pxlID in range(N):
                r[:, pxlID] = r[:, pxlID]*Weight[:, pxlID]
            R = r@r.T / N

            w = np.linalg.inv(R + 0.0001*np.eye(D))@self._d / \
                (self._d.T@np.linalg.inv(R + 0.0001*np.eye(D))@self._d)

            y = w.T@r
            Weight = 1 - np.exp(-λ*y)
            Weight[Weight < 0] = 0

            res = np.linalg.norm(y_old)**2/N - np.linalg.norm(y)**2/N
            print(f'iteration {T + 1}: ε = {res}')
            y_old = y.copy()

            # stop criterion:
            if np.abs(res) < e:
                break

            # display the detection results of each layer
            hCEMMap = np.reshape(y, [imgH, imgW])
        return hCEMMap

    def sam_img(self):
        '''
        Spectral Angle Match for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        
        d = np.reshape(self._d, [self._HIM.shape[2], 1])
        rr = np.sum(r**2, 0)**0.5
        dd = np.sum(d**2, 0)**0.5
        rd = np.sum(r*d, 0)
        result = np.arccos(rd/(rr*dd))

        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def ed_img(self):
        '''
        Euclidean Distance for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        d = np.reshape(self._d, [self._HIM.shape[2], 1])

        result = np.sum((r-d)**2, 0)**0.5
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def sid_img(self):
        '''
        Spectral Information Divergence for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        d = np.reshape(self._d, [self._HIM.shape[2], 1])

        m = r/np.sum(r, 0)
        n = d/np.sum(d, 0)
        drd = np.sum(m*np.log(m/n), 0)
        ddr = np.sum(n*np.log(n/m), 0)
        result = drd+ddr

        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def ace(self, K=None, u=None):
        '''
        Adaptive Cosin/Coherent Estimator for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]    
        param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
        param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        d = np.reshape(self._d, [self._HIM.shape[2], 1])
        if K is None or u is None:
            K, u = self.calc_K_u()
        rt = np.transpose(r)
        dt = np.transpose(d)
        try:
            Kinv = np.linalg.inv(K)
        except:
            Kinv = np.linalg.pinv(K)

        result = (dt@Kinv@r)**2 / ((dt@Kinv@d) * np.sum((rt@Kinv)*rt, 1)
                                   ).reshape(1, self._HIM.shape[0]*self._HIM.shape[1])
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def mf(self,  K=None, u=None):
        '''
        Matched Filter for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
        param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        if K is None or u is None:
            K, u = self.calc_K_u()
        du = d-u
        try:
            Kinv = np.linalg.inv(K)
        except:
            Kinv = np.linalg.pinv(K)

        k = 1 / (du.T@Kinv@du)
        w = k*(Kinv@du)
        result = w.T@r
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def kmd_img(self,  K=None, u=None):
        '''
        Covariance Mahalanobis Distance for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
        param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        d = np.reshape(self._d, [self._HIM.shape[2], 1])
        if K is None or u is None:
            K, u = self.calc_K_u()
        try:
            Kinv = np.linalg.inv(K)
        except:
            Kinv = np.linalg.pinv(K)
            warnings.warn(
                'The pseudo-inverse matrix is used instead of the inverse matrix in kmd_img(), please check the input data')

        result = np.sum(Kinv@(r-d)*(r-d), 0)**0.5
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def rmd_img(self,  R=None):
        '''
        Correlation Mahalanobis Distance for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        d = np.reshape(self._d, [self._HIM.shape[2], 1])
        if R is None:
            R = self.calc_R()
        try:
            Rinv = np.linalg.inv(R)
        except:
            Rinv = np.linalg.pinv(R)
            warnings.warn(
                'The pseudo-inverse matrix is used instead of the inverse matrix in rmd_img(), please check the input data')

        result = np.sum(Rinv@(r-d)*(r-d), 0)**0.5
        result = 1-(result/np.max(result))
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def kmfd(self, K=None, u=None):
        '''
        Covariance Matched Filter based Distance for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
        param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        if K is None or u is None:
            K, u = self.calc_K_u()
        try:
            Kinv = np.linalg.inv(K)
        except:
            Kinv = np.linalg.pinv(K)

        result = (r-u).T@Kinv@(d-u)
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def rmfd(self, R=None):
        '''
        Correlation Matched Filter based Distance for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
        param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        if R is None:
            R = self.calc_R()
        try:
            Rinv = np.linalg.inv(R)
        except:
            Rinv = np.linalg.pinv(R)

        result = r.T@Rinv@d
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def K_rxd(self, axis=''):
        '''
        Reed–Xiaoli Detector for image to point use Covariance Matrix and mean value µ

        param HIM: hyperspectral imaging, type is 3d-array
        param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
        param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
        param axis: 'N' is normalized RXD, 'M' is modified RXD, other inputs represent the original RXD
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        if K is None or u is None:
            K, u = self.calc_K_u()
        ru = r-u
        rut = np.transpose(ru)
        try:
            Kinv = np.linalg.inv(K)
        except:
            Kinv = np.linalg.pinv(K)
            warnings.warn(
                'The pseudo-inverse matrix is used instead of the inverse matrix in K_rxd(), please check the input data')

        if axis == 'N':
            n = np.sum((rut*rut), 1)
            result = np.sum(((np.dot(rut, Kinv))*rut), 1)
            result = result/n
        elif axis == 'M':
            n = np.power(np.sum((rut*rut), 1), 0.5)
            result = np.sum(((np.dot(rut, Kinv))*rut), 1)
            result = result/n
        else:
            result = np.sum((np.dot(rut, Kinv))*rut, 1)
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def R_rxd(self, axis="None"):
        '''
        Reed–Xiaoli Detector for image to point use Correlation Matrix

        param HIM: hyperspectral imaging, type is 3d-array
        param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
        param axis: 'N' is normalized RXD, 'M' is modified RXD, other inputs represent the original RXD
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        rt = np.transpose(r)

        if R is None:
            R = self.calc_R()
        try:
            Rinv = np.linalg.inv(R)
        except:
            Rinv = np.linalg.pinv(R)
            warnings.warn(
                'The pseudo-inverse matrix is used instead of the inverse matrix in R_rxd(), please check the input data')

        if axis == 'N':
            n = np.sum((rt*rt), 1)
            result = np.sum(((np.dot(rt, Rinv))*rt), 1)
            result = result/n
        elif axis == 'M':
            n = np.power(np.sum((rt*rt), 1), 0.5)
            result = np.sum(((np.dot(rt, Rinv))*rt), 1)
            result = result/n
        else:
            result = np.sum(((np.dot(rt, Rinv))*rt), 1)
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def lptd(self):
        '''
        Low Probability Target Detector

        param HIM: hyperspectral imaging, type is 3d-array
        param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        oneL = np.ones([1, self._HIM.shape[2]])
        if R is None:
            R = self.calc_R()
        try:
            Rinv = np.linalg.inv(R)
        except:
            Rinv = np.linalg.pinv(R)
            warnings.warn(
                'The pseudo-inverse matrix is used instead of the inverse matrix in lptd(), please check the input data')

        result = np.dot(np.dot(oneL, Rinv), r)
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def utd(self, K=None, u=None):
        '''
        Uniform Target Detector for image to point

        param HIM: hyperspectral imaging, type is 3d-array
        param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
        param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
        '''
        r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
        oneL = np.ones([1, self._HIM.shape[2]])
        if K is None or u is None:
            K, u = self.calc_K_u()
        ru = r-u
        try:
            Kinv = np.linalg.inv(K)
        except:
            Kinv = np.linalg.pinv(K)
            warnings.warn(
                'The pseudo-inverse matrix is used instead of the inverse matrix in utd(), please check the input data')

        result = (oneL-np.transpose(u))@Kinv@ru
        result = result.reshape(self._HIM.shape[0], self._HIM.shape[1])
        return result

    def rxd_utd(self, K=None, u=None):
        '''


        param HIM: hyperspectral imaging, type is 3d-array
        param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
        param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
        '''
        N = self._HIM.shape[0]*self._HIM.shape[1]
        r = np.transpose(self._HIM.reshape([N, self._HIM.shape[2]]))
        if K is None or u is None:
            K, u = self.calc_K_u()
        ru = r-u
        try:
            Kinv = np.linalg.inv(K)
        except:
            Kinv = np.linalg.pinv(K)
            warnings.warn(
                'The pseudo-inverse matrix is used instead of the inverse matrix in rxd_utd(), please check the input data')

        result = np.sum((np.transpose(r-1)@Kinv)*np.transpose(ru), 1)
        result = result.reshape([self._HIM.shape[0], self._HIM.shape[1]])
        return result

    def calc_R(self):
        '''
        Calculate the Correlation Matrix R

        param HIM: hyperspectral imaging, type is 3d-array
        '''
        try:
            N = self._HIM.shape[0]*self._HIM.shape[1]
            r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
            R = 1/N*(r@r.T)
            return R
        except:
            print('An error occurred in calc_R()')

    def calc_K_u(self):
        '''
        Calculate the Covariance Matrix K and mean value µ
        mean value µ was named u just because it looks like u :P

        param HIM: hyperspectral imaging, type is 3d-array
        '''
        try:
            N = self._HIM.shape[0]*self._HIM.shape[1]
            r = np.transpose(np.reshape(self._HIM, [-1, self._HIM.shape[2]]))
            u = (np.mean(r, 1)).reshape(self._HIM.shape[2], 1)
            K = 1/N*np.dot(r-u, np.transpose(r-u))
            return K, u
        except:
            print('An error occurred in calc_K_u()')

import numpy as np
from scipy.spatial.distance import cdist
from .abstract_kernel import AbstractKernel


class MaternKernel(AbstractKernel):
    """Matern-5/2 Kernel Class"""
    def cov(self, model_X, model_Y):
        # Compute the squared Euclidean distance between points.
        nX = model_X / self.length_scales
        nY = model_Y / self.length_scales
        dist_sq = cdist(nX, nY, "sqeuclidean")
        dist = np.sqrt(dist_sq)
        K = (1. + np.sqrt(5.)*dist + 5./3.*dist_sq) * np.exp(-np.sqrt(5.)*dist)
        return self.amplitude * K

    def grad_input(self, x, Y):
        """Implementation of abstract base class method.

        This code was taken from the implementation in scikit-optimize [1]. Per
        the New BSD License:

        Copyright (c) 2016 - scikit-optimize developers. All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are
        met:

        a. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

        b. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

        c. Neither the name of the scikit-optimize developers nor the names of
        its contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
        IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
        TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
        PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
        CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
        PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
        PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
        LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
        NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        [1] https://github.com/scikit-optimize
        """
        # diff = (x - Y) / length_scales
        # size = (n_train_samples, n_dimensions)
        diff = x - Y
        diff /= self.length_scales

        # dist_sq = \sum_{i=1}^d (diff ^ 2)
        # dist = sqrt(dist_sq)
        # size = (n_train_samples,)
        dist_sq = np.sum(diff**2, axis=1)
        dist = np.sqrt(dist_sq)

        # grad(fg) = f'g + fg'
        # where f = (1 + sqrt(5) * euclidean((X - Y) / length_scale) +
        #            5 / 3 * sqeuclidean((X - Y) / length_scale))
        # where g = exp(-sqrt(5) * euclidean((X - Y) / length_scale))
        sqrt_5_dist = np.sqrt(5.) * dist
        f2 = (5. / 3.) * dist_sq
        f2 += sqrt_5_dist
        f2 += 1
        f = np.expand_dims(f2, axis=1)

        # For i in [0, D) if x_i equals y_i
        # f = 1 and g = 1
        # Grad = f'g + fg' = f' + g'
        # f' = f_1' + f_2'
        # Also g' = -g * f1'
        # Grad = f'g - g * f1' * f
        # Grad = g * (f' - f1' * f)
        # Grad = f' - f1'
        # Grad = f2' which equals zero when x = y
        # Since for this corner case, diff equals zero,
        # dist can be set to anything.
        nzd_mask = dist != 0.0
        nzd = dist[nzd_mask]
        dist[nzd_mask] = np.reciprocal(nzd, nzd)

        dist *= np.sqrt(5.)
        dist = np.expand_dims(dist, axis=1)
        diff /= self.length_scales
        f1_grad = dist * diff
        f2_grad = (10. / 3.) * diff
        f_grad = f1_grad + f2_grad

        sqrt_5_dist *= -1.
        g = np.exp(sqrt_5_dist, sqrt_5_dist)
        g = np.expand_dims(g, axis=1)
        g_grad = -g * f1_grad
        return f * g_grad + g * f_grad

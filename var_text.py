# import theano
# import theano.tensor as T
# import numpy as np
#
# from theano import function
# class log_reg:
#
#     def __init__(self, input, n_feature,rng):
#
#
#
#         self.feature = n_feature#784
#         self.rng = rng
#
#
#         self.w = theano.shared(rng.randn(self.feature))
#         self.b = theano.shared(0.)
#
#         x = T.matrix()
#         y = T.vector()
#         m = 1 / (1 + T.exp(-T.dot(x, self.w) - self.b))
#         self.y_prime = m > .5
#
#         xent = -y * T.log(m) - (1-y) * T.log(1 - m)
#         cost = T.mean(xent) + 0.1*(self.w**2).sum()
#         gw, gb = T.grad(cost, [self.w, self.b])
#         up ={self.w : self.w-0.1*gw, self.b : self.b-0.1*gb}
#         self.train = theano.function(inputs = [x, y],outputs = [self.y_prime, xent], updates = up)
#         self. predict = theano.function(inputs = [x], outputs = self.y_prime)
# def train_model():
#
#     training_steps =1000
#     N = 400
#     rng = np.random
#     n_feature = 784
#     D = (rng.randn(N, n_feature), rng.randint(low=0, high=2, size=N))
#
#     model  = log_reg(D[0], n_feature,rng)
#     pred = theano.function(inputs = [model.x], outputs = model.y_prime)
#     for i in range(training_steps):
#         pred, err = model.train(D[0], D[1] )
#     #    print pred
#
#
#
#
#     prediction = model.predict(D[0])
#     for i in range(len(D[1])):
#         print prediction[i]- D[1][i]
#     #print "predictions on D: ", predict(D[0])
#
# train_model()
#

from progressbar import Percentage, ProgressBar, Bar
import time
widgets = ['Test: ', Percentage(), ' ',Bar(marker='=', left='[', right=']')]
pbar = ProgressBar(widgets=widgets, max_value=500)
pbar.start()
for i in range(1, 100, 1):
    time.sleep(0.1)
    pbar.update(i/2)
pbar.finish()
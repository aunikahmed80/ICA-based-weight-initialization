
 def init_model_from_file(self, fileName='mydA_param_custom_noise'):
        f = open(fileName)
        self.W = theano.shared(numpy.asarray(cPickle.load(f), dtype=theano.model_config.floatX), borrow = True)
        self.b = theano.shared(numpy.asarray(cPickle.load(f), dtype=theano.model_config.floatX), borrow = True)
        self.b_prime = theano.shared(numpy.asarray(cPickle.load(f), dtype=theano.model_config.floatX), borrow = True)
        self.W_prime = self.W.T

        self.params = [self.W, self.b, self.b_prime]
        f.close()

    def save_model_params(self):
        f = open('mydA_param_custom_noise','wb')
        cPickle.dump(self.W.get_value(borrow= True),f , -1)
        cPickle.dump(self.b.get_value(borrow= True),f, -1)
        cPickle.dump(self.b_prime.get_value(borrow= True),f, -1)
        f.close()


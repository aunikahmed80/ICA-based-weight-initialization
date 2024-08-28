import gzip
from heapq import nlargest

import cPickle

import numpy
import theano
from ica_prepare_dataset_cnn import   show_img
import csv
def get_prediction(classifier , input,out,dataset='test'):

    pred_model =  theano.function(
    inputs=[],
    outputs=[classifier.pred,classifier.p_y_given_x],
    givens={
        classifier.input: input
    }
    )

    y_pred, p_y =pred_model()
    print len(y_pred)
    count = 0
    error_idx = []
    numpy.set_printoptions(precision=2,linewidth=150)
    y_actual = out.eval()
    #show_std(y_pred,p_y,y_actual)

    for i, y in enumerate(y_actual):
        if(y_pred[i] != y ):
            count+=1
            error_idx.append(i)
            print 'Actual ', y ,' Predicted ',y_pred[i],nlargest(2, enumerate(p_y[i]), key= lambda  x:x[1])
    print count

    f = gzip.open('../../book_code/data/mnist.pkl.gz', 'rb')
    train_set, test_set, valid_set = cPickle.load(f)

    if dataset == 'test':
        test_set_x, test_set_y = test_set
        test_set_x = test_set_x[error_idx]
    elif dataset == 'valid':
        test_set_x, test_set_y = valid_set
        test_set_x = test_set_x[error_idx]
    print numpy.shape(test_set_x)
    show_img(numpy.asmatrix(test_set_x))

    save_error_idx(error_idx, y_actual, y_pred)


def save_error_idx(idx, y_actual, y_pred):


    with open('error_index.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in idx:
            writer.writerow([i,y_actual[i],y_pred[i]])

def show_std(y_pred,p_y,y_actual):
    for i, y in enumerate(y_actual):
        print 'Actual ', y ,' Predicted ',y_pred[i],p_y[i],'\t',numpy.var(p_y[i])
        if i >10:
            break
    count = 0
    for i, y in enumerate(y_actual):
        if(y_pred[i] != y ):
           count +=1
           print 'Actual ', y ,' Predicted ',y_pred[i],p_y[i],'\t',numpy.var(p_y[i])
        if count > 10 :
            break




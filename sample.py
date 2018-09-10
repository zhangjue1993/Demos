import numpy

# without theano
def calculateprob(x, mean, cov):
    x = numpy.asarray(x, dtype = numpy.float32 )
    mean = numpy.asarray(mean, dtype = numpy.float32)
    cov = numpy.asarray(cov, dtype = numpy.float32)
    cov_det = numpy.linalg.det(cov)
    cov_inv = numpy.linalg.inv(cov)
    temp = numpy.dot((x-mean),cov_inv)
    temp = numpy.dot(temp,(x-mean).T)
    dim = x.shape[1] # only if x is a matrix
    prob = (1./numpy.sqrt(((2*numpy.pi)**dim)*cov_det))*numpy.exp(-0.5*temp)
    prob = numpy.diag(prob)
    return prob

# validate
x = [[0,0],[1,1],[0,1]]
mean = [1,2]
cov =[[1,0],[0,1]]

print calculateprob(x,mean,cov)

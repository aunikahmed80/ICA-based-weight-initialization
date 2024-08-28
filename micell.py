import math
def H(p):
    h =0
    for x in p:
        h+=  -x * math.log(x,2)
    return h

#print H([.5, .25, .125, .125])


def kullback_leibler(p,q):
    D = 0
    for r,s in zip(p,q):
        D+= r * math.log(r/s,2)

        #print r,s
    return  D

print kullback_leibler([.5,.5],[.25,.75])
print kullback_leibler([.25,.75],[.5,.5])
print 7.0/4

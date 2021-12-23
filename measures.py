import numpy as np 
import math


def chebyshev(rd,pd):
    temp = np.abs(rd-pd)
    temp = np.max(temp,1)
    distance = np.mean(temp)
    return distance

def clark(rd,pd):
    temp1 = (pd - rd)**2
    temp2 = (pd + rd)**2
    temp = np.sqrt(np.sum(temp1 / temp2, 1))
    distance = np.mean(temp)
    return distance

def canberra(rd,pd):
    temp1 = np.abs(rd-pd)
    temp2 = rd + pd
    temp = np.sum(temp1 / temp2,1)
    distance = np.mean(temp)
    return distance

def kl_dist(rd,pd):
    eps = 1e-12
    temp = rd * np.log(rd / pd + eps)
    temp = np.sum(temp,1)
    distance = np.mean(temp)
    return distance

def cosine(rd,pd):
    inner = np.sum(pd*rd,1)
    temp1 = np.sqrt(np.sum(pd**2,1))
    temp2 = np.sqrt(np.sum(rd**2,1))
    temp = inner / (temp1*temp2)
    distance = np.mean(temp)
    return distance


def intersection(rd,pd):
    (rows,cols) = np.shape(rd)
    dist = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            dist[i] = dist[i] + min(rd[i,j],pd[i,j])
    distance = np.mean(dist)
    return distance

import numpy as np
import urllib.request

def read_digits(url):
    data = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            data.append(line.strip())
    datastring = []
    for line in data:
        datastring.append(line.decode("utf-8"))
    datastring = ''.join(datastring)
    datastring = list(map(int, list(datastring)))
    return (np.array(datastring))
class PRNGs:
    def __init__(self):
        pass
    def LCG(self, n, seed, M, a, c):
        result = np.zeros(n+1, dtype='int')
        result[0] = seed % M
        for i in range(1,n+1):
            result[i] = (a * result[i-1]+c) % M
        return result[1:]
    def GLCG(self, n, seed, M, a):
        k = len(seed)
        result = np.zeros(n+k,dtype='int')
        result[0:k] = seed % M
        for i in range(1,n+1):
            result[k+i-1] = (a * result[i:(k+i)]).sum() % M
        return result[k:]
    def Excel(self,n, seed):
        result = np.zeros(n+1,dtype='float')
        result[0] = seed
        for i in range(1,n+1):
            result[i] = (0.9821 * result[i-1] + 0.211327) % 1
        return result[1:]
    def BlumBlumShub(self, n, seed, M):
        result = np.zeros(n + 1, dtype='int')
        result[0] = seed % M
        for i in range(1, n + 1):
            result[i] = result[i - 1]**2 % M
        return result[1:]
    def RC4(self, n, M, key):
        result = np.zeros(n, dtype='int')
        keylen = len(key)
        S = np.arange(0,M)
        j=0
        for i in range(M):
            j = (j+ S[i] + key[i % keylen]) % M
            S[i], S[j] = S[j], S[i]
        i=0
        j=0
        while i<n:
            i = (i + 1)
            j = (j + S[i % M])
            S[i % M], S[j % M] = S[j % M], S[i % M]
            result[i-1] =S[(S[i % M]+S[j % M]) % M]
        return result

    def irrational_expansion(self,n):
        pi=read_digits('http://www.math.uni.wroc.pl/~rolski/Zajecia/data.pi')
        euler=read_digits('http://www.math.uni.wroc.pl/~rolski/Zajecia/data.e')
        sqrt2=read_digits('http://www.math.uni.wroc.pl/~rolski/Zajecia/data.sqrt2')
        result=np.concatenate((pi,euler,sqrt2))
        return result[:n]

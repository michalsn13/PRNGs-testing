import numpy as np
import scipy.special as special
import math
import scipy.stats as stats
class tests:
    def __init__(self):
        pass
    def chi_squared(self,seq,k=10,p='uniform'): #for all
        n = len(seq)
        if p == 'uniform':
            p = 1 / k * np.ones(k)
        if abs(p.sum()-1)>1e-5:
            raise Exception('Probabilities should sum up to 1...')
        chi_squared=0
        cdf = 0
        for i in range(k):
            y = -cdf
            cdf = (seq <= ((i+1)/k)).sum()
            y += cdf
            chi_squared += ((y-n*p[i])**2)/(n*p[i])
        p_value = 1-stats.chi2(k-1).cdf(chi_squared)
        return p_value
    def KS(self,seq): #for all
        n = len(seq)
        a = 0
        for el in list(set(seq)):
            F_emp = (seq <= el).sum()/n
            b = abs(F_emp-el)
            if b >= a:
                a = b
        D = np.sqrt(n)*a
        p_value = 2*sum([(-1)**(i-1)*np.exp(-2*i*i*D*D) for i in range(1,10**3+1)])
        return p_value
    def monobit(self,seq): #for bits
        n = len(seq)
        seq_mapped = 2*seq - 1
        test = seq_mapped.sum()/np.sqrt(n)
        p_value = 2*(1-stats.norm().cdf(abs(test)))
        return p_value
    def serial(self,seq,k=10): #for uniform discrete
        n = len(seq)
        pairs = np.array([[seq[2*i],seq[2*i+1]] for i in range(n//2)])
        chi_squared = 0
        for i in range(k):
            for j in range(k):
                y = ((pairs[:,0] <(i+1)/k)*(pairs[:,0]>=i/k)*(pairs[:,1] <(j+1)/k)*(pairs[:,1]>=j/k)).sum()
                chi_squared += ((y-len(pairs)/(k**2))**2)/(len(pairs)/(k**2))
        p_value = 1-stats.chi2(k**2-1).cdf(chi_squared)
        return p_value
    def poker(self,seq): #for uniform discrete
        M=6
        seq=(seq*M).astype(int)
        n = len(seq)
        chi_squared = 0
        r_nums = [len(set(seq[5*i:5*(i+1)])) for i in range(n//5)]
        factor = 1/M**5
        stirlings=[1,15,25,10,1]
        for i in range(1,6):
            factor *= (M-i+1)
            chi_squared += (r_nums.count(i)-len(r_nums)*stirlings[i-1]*factor)**2/(len(r_nums)*stirlings[i-1]*factor)
        p_value = 1-stats.chi2(5-1).cdf(chi_squared)
        return p_value

    def approx_entropy(self,seq): #for bits
        m = 5
        n = len(seq)
        seq_calc = list(seq)
        phis=[]
        for l in [m,m+1]:
            seq_calc = seq_calc + seq_calc[-(l - 1):]
            bits = [tuple(seq_calc[i:(i + l)]) for i in range(n)]
            possibilities = list(set(bits))
            probs = [bits.count(i) / n for i in possibilities]
            entropy = sum([i * np.log(i) for i in probs])
            phis.append(entropy)
        ApEn=phis[0]-phis[1]
        chi2=2*n*(np.log(2)-ApEn)
        p_value=1-stats.chi2(2**m).cdf(chi2)
        return p_value


























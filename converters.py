import numpy as np
def int_binary(seq,m):
    binary_len=int(np.ceil(np.log2(m)))
    result=[]
    for num in seq:
        result+=[int(i) for i in format(num, f'0{binary_len}b')]
    return np.array(result)

def binary_int(seq,m):
    binary_len=int(np.ceil(np.log2(m)))
    result=[]
    a=0
    power=binary_len-1
    for num in seq:
        a+=2**power * num
        if power==0:
            result.append(int(a))
            a=0
            power=binary_len-1
        else:
            power -= 1
    return np.array(result)

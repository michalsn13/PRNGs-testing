from PRNGs import PRNGs
from tests import tests
from converters import int_binary,binary_int
from time import time
import numpy as np
import matplotlib.pyplot as plt

prngs=PRNGs()
tests=tests()
R=100
n=200
params=[(13, np.array([3,7,68])),
         (2**10,np.array([3,7,68]))
         ]

all_tests=[{'name':'chi_squared','fun':(lambda seq,k: tests.chi_squared(seq)),'form':'float'},
           {'name':'KS','fun':(lambda seq,k: tests.KS(seq)),'form':'float'},
           {'name':'poker','fun':(lambda seq,k: tests.poker(seq)),'form':'float'},
           {'name':'serial','fun':(lambda seq,k: tests.serial(seq,k)),'form':'float'},
           {'name':'monobit','fun':(lambda seq,k: tests.monobit(seq)),'form':'binary'},
           {'name':'approx_entropy','fun':(lambda seq,k: tests.approx_entropy(seq)),'form':'binary'}
           ]
plot_num=-2
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
for par in params:
    x0 = np.array([round(time() * (10**i)) % par[0] for i in range(len(par[1]))])
    k = 5 if par[0]<=32 else 10
    print(f'Testing for GLCG{par} pseudo-random sequence with (x0,x1,x2)={x0}...')
    sequence=prngs.GLCG(n*R,x0,*par)
    sequence_float=sequence/par[0]
    sequence_binary=int_binary(sequence,par[0])
    plot_num+=2
    axs[0].plot(sequence_float[:n])
    axs[1].hist(sequence_float[:n])
    fig.suptitle(f'GLCG{par} pseudo-random sequence with (x0,x1,x2)={x0}\n{n} first iterations')
    plt.savefig(f'GLCG/GLCG{plot_num}.png')
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    for test in all_tests:
        print(f'    {test["name"]} test calculations...')
        if test['form']=='float':
            sequence_scaled = sequence_float
        else:
            sequence_scaled = sequence_binary
        p_values = []
        for iter in range(R):
            seq_cropped = sequence_scaled[(iter * n):((iter + 1) * n)]
            p_values.append(test["fun"](seq_cropped,k))
        p_values = np.array(p_values)
        try:
            p_value_2nd = tests.chi_squared(p_values)
        except Exception as e:
            print(e)
        else:
            axs[plot_num % 2].hist(p_values)
            axs[plot_num % 2].set_title(
                f'GLCG{par} with (x0,x1,x2)={x0}\n{R} p-values from {test["name"]} test with 2-nd level testing p-value={round(p_value_2nd, 4)}')
            plot_num += 1
            if plot_num % 2 == 0:
                plt.savefig(f'GLCG/GLCG{plot_num}.png')
                fig, axs = plt.subplots(1, 2, figsize=(16, 5))

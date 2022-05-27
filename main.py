import argparse
# import imp
from loguru import logger
import multiprocessing as mp
import sys
import time
import tracemalloc
import numpy as np
import pandas
import matplotlib.pyplot as plt
from consist_h import Consist_h

# import recorder
# from evaluator import Evaluator
PL = [1307,34,2660,332,125,159,152,3513,80,164,71,177,1702,
1769,246,502,411,1551,1719,73,86,493,289,164,157,212,46,376,
416,65,101,341,55,159,47,119,376,76,3347,101,37,99,163,144,
3366,41,341,478,79,718,1726,60,143,64,53,272,131,45,376,119,
1722,512,3347,188,1247,104,112,335,702,878,37,443,762,247,
826,878,1177,411,3422,108,83,120,87,1204,135,289,56,61,95,
118,163,61,36,42,341,103,723,69,1734,86 ]

PR = [1769,2556,3408,466,188,762,2660,4293,124,2298,314,463,
2556,2026,463,1442,3513,1721,3513,126,105,3422,492,1707,197,
1875,82,1725,1243,508,4180,3403,163,197,433,1725,1204,160,
3422,458,502,1110,1110,439,3408,1307,1726,538,502,723,2552,
163,678,215,3218,2026,137,313,458,224,1734,1091,3408,723,2591,
1110,2298,522,4569,2026,111,826,2556,262,1779,1726,3403,629,
3513,328,272,177,109,2015,177,4180,1442,72,479,139,250,362,
384,118,723,627,1875,136,2273,178 ]

AL = [57,57,19,28,55,42,50,58,33,23,60,29,34,20,21,33,60,38,
47,52,32,81,45,71,75,38,71,66,32,26,32,30,51,32,26,28,65,36,
44,46,29,75,78,41,29,72,23,15,56,27,55,78,59,29,15,70,77,32,
70,42,19,31,20,69,17,35,43,72,18,25,29,48,51,78,45,45,45,58,
77,56,77,32,75,47,70,40,81,52,74,62,24,68,33,52,38,68,63,39,
47,41]

AR = [63,64,74,67,74,75,65,73,84,49,61,45,64,31,36,34,67,56,
77,62,73,84,48,73,77,53,72,75,73,61,36,64,59,59,57,71,84,63,
62,71,71,81,81,46,39,84,34,21,74,70,57,81,62,50,65,74,84,38,
78,63,37,56,67,70,36,40,78,74,63,51,33,54,75,81,71,59,68,72,
78,65,84,37,84,68,77,46,84,58,75,84,47,77,70,56,60,70,71,63,
59,73]

NL = [62,100,73,90,62,111,73,52,97,57,44,124,90,113,92,57,93,
150,114,51,128,147,108,43,21,112,119,82,30,66,39,51,108,63,142,
98,96,141,100,71,56,93,82,64,69,130,104,107,76,108,65,83,147,
81,137,150,145,166,107,106,89,59,140,153,144,69,102,61,112,149,
122,125,111,80,88,70,43,169,121,73,133,43,110,38,148,80,113,64,
161,21,81,68,65,125,111,52,128,75,73,35]

NR = [131,148,153,161,65,169,140,150,105,95,150,159,118,142,133,
133,141,171,166,80,155,154,126,123,77,147,129,127,97,116,154,52,
117,132,145,142,122,147,110,132,113,150,94,123,104,171,156,155,
93,144,156,127,149,101,141,166,161,171,131,124,90,75,159,182,151,
105,147,154,150,150,129,141,141,146,96,79,107,171,147,78,140,72,
132,148,166,150,126,124,169,136,147,108,100,148,151,134,159,103,
111,135]
# accidential_drug_deaths  normal_data_5000r 49733rows_link_ping
parser = argparse.ArgumentParser(description='exp of Consist_h method queries') 
parser.add_argument('--dataset',type=str, default= 'normal_data_200000r', help='specify the name of data in result file')
parser.add_argument('--s', type=int, default=0, help='specify the starting value of the bottom bin edge')
parser.add_argument('--step', type=int, default=1, help='specify the step value in creating the bottom bin')
parser.add_argument('--n', type=int, default=240000, help='specify the num of bottom level bins')
parser.add_argument('--num',type=int,default=100,help='specify the number of range queries')
parser.add_argument('--range_epsilon', type=float, default=1, help='specify the epsilon value for range queries')
parser.add_argument('--fanout', type=int, default=16, help='specify the fanout value in establishing the hierarchy')
parser.add_argument('--type', type=str, default='uniform', help='specify the type of range queries')
parser.add_argument('--x', type=str, default='x', help='specify the name of x value')

args = parser.parse_args()
res_name = f'{args.dataset}_eps_{args.range_epsilon}f_{args.fanout}.txt'
sys.stdout = open(res_name, 'w')

df = pandas.read_csv(f"../data/{args.dataset}.csv", usecols=[args.x],nrows=5000)

data_o = df.to_numpy()
n = len(data_o)
print('the num of samples is ', n)
data = []
for i in range(n):
    data.append(data_o[i][0])

# print(data)
# and we want to translate it into unit bin
sorted_data = np.sort(data)
unique_data = np.unique(data)
min_value = sorted_data[0]
max_value = sorted_data[-1]
print('min', min_value)
print('max', max_value)

# in t-digest settings, we can test range query like [value1, value2]
# actually we can create a mapping between bin indexes and true values
# [v1,v2] means:  if unit bin, then [index1, index2]
# if larger bin, then (r1-v1)/bin*[bin1],[bin2],(v2-l3)/bin*[bin3]

# create unit bins and remember the mappings
# int value, firstly create fine bins, i.e., the unit bin values
def bottom_creator(s, t, step):
    unit_bins = range(s,t,step)
    hist, bins = np.histogram(sorted_data,bins=unit_bins)
  
    return hist, bins

# s = 0
# note that 5002 serves for create unit_bins as [0...5001], then the bin count can be computed on [0,1)...[5000,5001)]
# t = 5002

s = args.s
t = args.n + 1
step = args.step

print("now we create the bottom level bins, with start value %d, ending value %d, step %d" % (s,t,step))
bottom_bin, bins = bottom_creator(s, t, step) 
print(bottom_bin)

print('now we display the bottom bins histogram')
fig, ax = plt.subplots()
bin_array, bins, patches = ax.hist(sorted_data, bins=range(5002))
plt.close()
# in fact, we find most values fall into 0-1000, and we can consider whether to cut the tail

def range_query_creator(pre=200):
    type = args.type
    num = args.num
    l_array = np.zeros(pre)
    r_array = np.zeros(pre,int)

    if type == "uniform":
        l_array = np.random.choice(unique_data,pre,replace=True)
        for i in range(pre):
            # make sure r is > l, the problem is when l==r, always ans 0, but that's wrong
            # choose from unique, cannot be the same
            while(r_array[i] < l_array[i]):
                r_array[i]=np.random.choice(unique_data)
            # print('create a range query [%d,%d]' % (l_array[i],r_array[i]))
        
        tot = 0 
        L = []
        R = []
        for i in range(pre):
            if r_array[i] == l_array[i]:
                continue
            L.append(l_array[i])
            R.append(r_array[i])
            tot += 1
            if(tot == 100):
                break
    print('now we create', args.num, 'range queries:')
    print('L:', L)
    print('R:', R)
    return L, R


def run_range_queries(l_array, r_array):
    num = args.num
    err = []
    
    print('now we run the experiment with range_epsilon:', args.range_epsilon)
    for i in range(num):
        l = l_array[i]
        r = r_array[i]
        print('now we run the query [%d,%d)' % (l,r))
     
        sp = np.where(sorted_data == l)
        s = sp[0][0]
        ep = np.where(sorted_data == r)
        e = ep[0][0]
        true_count = e-s
        print('the true count is %d' % true_count)

        est_count = H.hierarchy_range_count(consist_h,l,r)
        print('the hierarchical estimated count is %f' % est_count)
        err.append(np.abs(true_count-est_count))
        # if i ==0:
        #     break
    mean_err = np.mean(err)
    mse = np.var(err)
    print("in %d range queries, the mean absolute err is %f, and the var of err is %f" % (num, mean_err, mse))

np.random.seed(42)
l_array, r_array = range_query_creator()

print('now we create the hierarchy')
st = time.time()
tracemalloc.start()
H = Consist_h(bottom_bin, args)
# the GS =1, establish and consist
raw_h = H.est_hierarchy(1)
consist_h = H.consist(raw_h)
print('now we have established the consisted hierarchy with fanout %d' % args.fanout)

run_range_queries(l_array, r_array)
ed = time.time()
elapse = (ed-st)*1000
print('the elpased time in hierarchical range-exp is',elapse, "milliseconds")

print(tracemalloc.get_traced_memory())
tracemalloc.stop()

sys.stdout.close()



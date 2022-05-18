import argparse
import imp
from loguru import logger
import multiprocessing as mp
import sys
import numpy as np
import pandas
import matplotlib.pyplot as plt
from consist_h import Consist_h

# import recorder
# from evaluator import Evaluator

df = pandas.read_csv("./data/49733rows_link_ping.csv", nrows=5000)
data_o = df.to_numpy()
n = len(data_o)
print('the num of samples is ', n)
data = []
for i in range(n):
    data.append(data_o[i][1])

# now we have data in an array
# and we want to translate it into unit bin
sorted_data = np.sort(data)
unique_data = np.unique(data)
min_value = sorted_data[0]
max_value = sorted_data[-1]


parser = argparse.ArgumentParser(description='exp of Consist_h method queries') 
parser.add_argument('--s', type=int, default=0, help='specify the starting value of the bottom bin edge')
parser.add_argument('--step', type=int, default=1, help='specify the step value in creating the bottom bin')
parser.add_argument('--n', type=int, default=5001, help='specify the num of bottom level bins')
parser.add_argument('--fanout', type=int, default=16, help='specify the fanout value in establishing the hierarchy')
parser.add_argument('--num',type=int,default=20,help='specify the number of range queries')
parser.add_argument('--range_epsilon', type=float, default=10, help='specify the epsilon value for range queries')
parser.add_argument('--type', type=str, default='uniform', help='specify the type of range queries')

args = parser.parse_args()
# print(sorted_data)
# in t-digest settings, we can test range query like [value1, value2]
# here, we can test range query like the same?
# actually we can create a mapping between bin indexes and true values
# [v1,v2] means:
# if unit bin, then [index1, index2]
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

print('now we display the bottom bins histogram')
fig, ax = plt.subplots()
bin_array, bins, patches = ax.hist(sorted_data, bins=range(5002))
plt.close()
# in fact, we find most values fall into 0-1000, and we can consider whether to cut the tail


print('now we create the hierarchy')
H = Consist_h(bottom_bin, args)
# the GS =1, establish and consist
raw_h = H.est_hierarchy(1)
consist_h= H.consist(raw_h)
print('now we have established the consisted hierarchy with fanout %d' % args.fanout)

# randomly select range queries or specify all the queries with ml, mr
np.random.seed()
ml = [ 42, 54,  55,  82,  41,  67,  38,  62,  60, 137,  65,  54,  67,  88,  68,  50,  65,  39,
  72,  69]
mr = [ 67, 105,  69, 402,  69,  70,  77,  82,  67, 402,  72,  69,  82, 114,
 137, 228,  75,  75, 105,  77]

def range_query_creator():
    type = args.type
    num = args.num
    l_array = np.zeros(num)
    r_array = np.zeros(num)

    if type == "uniform":
        l_array = np.random.choice(unique_data,num,replace=True)
        for i in range(num):
            # make sure r is > l, the problem is when l==r, always ans 0, but that's wrong
            # choose from unique, cannot be the same
            while(r_array[i] <= l_array[i]):
                r_array[i]=np.random.choice(unique_data)
            print('create a range query [%d,%d]' % (l_array[i],r_array[i]))
    print("now we create the range queries",l_array,r_array)
    return l_array, r_array


def run_range_queries(l_array, r_array):
    num = args.num
    err = []
    for i in range(num):
        l = l_array[i]
        r = r_array[i]
        print('now we run the query [%d,%d]' % (l,r))
     
        sp = np.where(sorted_data == l)
        s = sp[0][0]
        ep = np.where(sorted_data == r)
        e = ep[0][-1]
        true_count = e-s+1
        print('the true count is %d' % true_count)

        est_count = H.hierarchy_range_count(consist_h,l,r)
        print('the hierarchical estimated count is %f' % est_count)
        err.append(np.abs(true_count-est_count))

    mean_err = np.mean(err)
    mse = np.var(err)
    print("in %d range queries, the mean absolute err is %f, and the var of err is %f" % (num, mean_err, mse))


print('now we run the experiment with range_epsilon:', args.range_epsilon)
run_range_queries(ml,mr)






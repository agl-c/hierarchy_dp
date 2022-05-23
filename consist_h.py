import abc
from ast import arg
from distutils.command.build_scripts import first_line_re
import math
import numpy as np
import primitive

class Consist_h(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, bottom_bins, args):
        self.bottom_bins = bottom_bins
        self.args = args
        self.fanout = args.fanout
        # n is the num of bins in level 0
        self.n = args.n
        # exclude the root level
        self.num_levels = int(math.log(self.n, self.fanout))
        print('num_levels', self.num_levels)
        self.epsilon = args.range_epsilon / self.num_levels
        self.granularities = [self.fanout**h for h in range(self.num_levels)]
        print('granularities', self.granularities)
    
   
    def update_fanout(self):
        self.num_levels = int(math.log(self.n, self.fanout))
        self.epsilon = self.args.range_epsilon / self.num_levels
        self.granularities = [self.fanout ** h for h in range(self.num_levels)]


    # establish the original tree and add noises to each level
    def est_hierarchy(self, ell):
       
        # cur_data = np.copy(self.users.data[self.args.m:])
        # cur_data[cur_data > ell] = ell
        cur_data = self.bottom_bins

        count = []
        # from leaf to root
        for granularity in self.granularities:
            # on higher level, the bigger node use some empty nodes with 0 count, only valid count contributes
            num_slots = np.ceil(self.n / granularity).astype(int)
            #print('granularity is ', granularity)
            #print('num of slots is ', num_slots)
            count_l = np.zeros(num_slots)
            for slot in range(num_slots):
                # if slot == num_slots -1:
                #      print('debug:', cur_data[slot* granularity: (slot+1)*granularity])
                # the turth is ,when slice stop value is further than array area, it will use array area
                # so don't worry on sum more values
                count_l[slot] = sum(cur_data[slot * granularity: (slot + 1) * granularity])
            count.append(count_l)
        # print(count)

        # add noise to each level, count[h] means a array which hold the h-th level counts
        for h in range(self.num_levels):
            ell = 1
            count[h] = primitive.laplace(ell / self.epsilon, count[h])
        # then we have a raw noisy tree, i.e., count   
        # print('noisy:', count)

        return count

    def consist(self, count):
        # requires a complete tree

        fanout = self.fanout

        # leaf to root, this step get weighted value in the whole tree
        for h in range(1, len(count)):
            coeff = fanout ** (h + 1)
            coeff2 = fanout ** h

            for est_i in range(len(count[h])):
                children_est = sum(count[h - 1][est_i * fanout: (est_i + 1) * fanout])
                count[h][est_i] = (coeff - coeff2) / (coeff - 1) * count[h][est_i] + (coeff2 - 1) / (
                        coeff - 1) * children_est

        # root to leaf, now we consist the err by add it equally between all the children of one parent
        for h in range(len(count) - 1, 0, -1):
            for est_i in range(len(count[h])):
                children_est = sum(count[h - 1][est_i * fanout: (est_i + 1) * fanout])

                diff = (count[h][est_i] - children_est) / fanout
                count[h - 1][est_i * fanout: (est_i + 1) * fanout] += diff      
        # print(count)
        return count


    # def my_range_sum(self, h, l, r):
    #     if np.isscalar(h[0]):
    #         return sum(h[l:r])
    #     else:
    #         return self.hierarchy_range(h, l, r)

    def hierarchy_range_count(self, h, l, r):
        # print(h)
        result = 0
        # since the index is from 0
        layer = self.num_levels - 1
        # only one level to invest
        if layer == 0:
            bottom = h[0]
            return sum(bottom[l: r])
        # nodes_to_invest = range(self.fanout)  ccccccc bug qaqqqqqqq
        # h-1-th level holds slots: 
        # we should invest the highest level's all slots:
        granularity_max = self.fanout ** layer
        first_invest_slots = np.ceil(self.n / granularity_max).astype(int)
        nodes_to_invest = range(first_invest_slots)

        # from top to bottom
        while nodes_to_invest:
            granularity = self.fanout ** layer
            new_nodes = []
            for node in nodes_to_invest:
                
                # print(layer,node)
                if granularity * (node + 1) <= l or granularity * node >= r:
                    # print('unrelated')
                    continue

                if granularity * node >= l and granularity * (node + 1) <= r:
                    # print('add in')
                    result += h[layer][node]
                # should invest the corresponding nodes in lower level, compute the indexes    
                else:
                    # print('dig fur')
                    new_nodes += range(self.fanout * node, self.fanout * (node + 1))

            nodes_to_invest = new_nodes
            # print(layer)
            layer -= 1
            # print(layer)
            if layer < 0:
                # print('break')
                break

        return result




  




    
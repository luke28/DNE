#!/usr/bin/python

import sys, os

def statistic_group(ifile, ofile):
    idict = {}
    cnt = 0
    with open(ifile, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            cnt += 1
            if cnt > 56300:
                break
            line = line.strip()
            items = line.split()
            if len(items) < 2:
                continue
            node_id = items[0]
            for it in items[1:]:
                if it not in idict:
                    idict[it] = 1
                else:
                    idict[it] += 1

    with open(ofile, 'w') as outfile:
        for k in idict:
            outfile.write(k + "\t" + str(idict[k]) + "\n")

if __name__ == '__main__':
    ifile = sys.argv[1]
    ofile = sys.argv[2]
    statistic_group(ifile, ofile)

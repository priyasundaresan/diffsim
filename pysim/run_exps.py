#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    for i in range(1,6):
        os.system('python exp_debug_triangle_vid.py')
        os.system('mv default_out ./runs_full_vid/run%d'%(i))

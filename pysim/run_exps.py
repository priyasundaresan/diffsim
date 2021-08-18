#!/usr/bin/env python
import os
import sys

if __name__ == '__main__':
    for i in range(7):
        os.system('python exp_triangle_video.py')
        os.system('mv default_out ./runs/run%d'%(i+4))

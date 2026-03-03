#!/usr/bin/env python
"""Check if improvement meets threshold"""
import sys

try:
    improvement = float(sys.argv[1])
    threshold = float(sys.argv[2])
    print('true' if improvement >= threshold else 'false')
except Exception as e:
    print('false')
    sys.exit(0)

#!/usr/bin/env python
"""Extract improvement percentage from metrics JSON file"""
import json
import sys

try:
    with open('improvement_metrics.json', 'r') as f:
        data = json.load(f)
    improvement = data.get('improvement_percentage', 0)
    print(improvement)
except Exception as e:
    print(0)
    sys.exit(0)

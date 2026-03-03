#!/usr/bin/env python
"""Verify key dependencies are installed and print versions"""
import sys

try:
    import torch
    print(f'torch: {torch.__version__}')
except ImportError as e:
    print(f'ERROR: torch not installed - {e}')
    sys.exit(1)

try:
    import numpy
    print(f'numpy: {numpy.__version__}')
except ImportError as e:
    print(f'ERROR: numpy not installed - {e}')
    sys.exit(1)

try:
    import sklearn
    print(f'sklearn: {sklearn.__version__}')
except ImportError as e:
    print(f'ERROR: sklearn not installed - {e}')
    sys.exit(1)

print('\n✅ All key dependencies verified')

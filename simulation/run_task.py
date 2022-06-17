import json
import os

from simulation.comparison import repeat, simulation_compare

import ctypes

openblas_lib = ctypes.cdll.LoadLibrary('/usr/local/lib/python3.7/dist-packages/numpy/.libs/libopenblasp-r0-2ecf47d5.3.7.dev.so')

openblas_lib.openblas_set_num_threads(12)

with open('task.json') as f:
    tasks = json.load(f)

for task in tasks:
    if os.path.exists('output/' + task['name'] + '.pkl'):
        continue
    print(f"running {task['name']}")
    repeat(simulation_compare, ** task)
# ***********************************************
# Import libraries
# ***********************************************
import time
import os
from MCTS_NEW import *
import pickle

from tensorflow.python.client import device_lib


code_begins = time.time()

# print(device_lib.list_local_devices())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
args = {
    "Total_time": 100,          # Time window available for scheduling in seconds
    "NumTasks": 15,             # Number of tasks
    "NumFeatures": 8,           # start, interval, tardiness
    "ValidationSize": 20,     # Number of problems per iteration
    "TestSize": 100,
    "max_cost": 10**10,            # Maximum cost used to normalize all costs
    "NumMCTSRollouts": 256,
}

nmcts = MCTS(args)

# mcts_cost, optimal_cost, est_cost, edt_cost, optimal_nodes = nmcts.compare()
optimal_cost, est_cost, edt_cost, optimal_nodes, optimal_dropped, est_dropped, edt_dropped\
    = nmcts.compare()

print('      Optimal AvgCost: %.2f' % np.mean(optimal_cost))
print('      EST AvgCost: %.2f' % np.mean(est_cost))
print('      EDT AvgCost: %.2f' % np.mean(edt_cost))
print('      Optimal Dropped: %.2f' % np.mean(optimal_dropped))
print('      EST Dropped: %.2f' % np.mean(est_dropped))
print('      EDT Dropped: %.2f' % np.mean(edt_dropped))
# print('      MCTS AvgCost: %.2f' % mcts_avg_cost)

folder = 'results'
if not os.path.exists(folder):
    os.mkdir(folder)

filename = 'optimal_cost'
filepath = os.path.join(folder, filename)
np.save(filepath, optimal_cost)

filename = 'est_cost'
filepath = os.path.join(folder, filename)
np.save(filepath,est_cost)

filename = 'edt_cost'
filepath = os.path.join(folder, filename)
np.save(filepath,edt_cost)

filename = 'optimal_dropped'
filepath = os.path.join(folder, filename)
np.save(filepath,optimal_dropped)

filename = 'est_dropped'
filepath = os.path.join(folder, filename)
np.save(filepath,est_dropped)

filename = 'edt_dropped'
filepath = os.path.join(folder, filename)
np.save(filepath,edt_dropped)

# filename = 'mcts_cost'
# filepath = os.path.join(folder, filename)
# np.save(filepath, mcts_cost)

filename = 'optimal_nodes'
filepath = os.path.join(folder, filename)
f = open(filepath, 'wb')
pickle.dump(optimal_nodes, f)

# filename = 'mcts_nodes'
# filepath = os.path.join(folder, filename)
# f = open(filepath, 'wb')
# pickle.dump(mcts_nodes, f)

# Print Runtime
time=time.time() - code_begins
hours=int(time/3600)
minutes=int((time - 3600*hours)/60)
seconds=int(time - 3600*hours - 60*minutes)
print("--- %s hrs, %s mins, %s secs ---" % (hours, minutes, seconds))




from test import output
from statistic import statics
num_node=40
for i in range(num_node):
    print("node_id=",i)
    output(i)
    statics()
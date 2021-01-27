import numpy as np
import exarl
from exarl.simple_comm import ExaSimple
from exarl.learner_trace import learner_trace

tr = learner_trace(ExaSimple())
for i in range(10):
    tr.update(10-i)
    if i%3==0:
        tr.snapshot()
    
data = tr.write()
import numpy as np
import exarl
from exarl.simple_comm import ExaSimple
from exarl.learner_trace import learner_trace

tr = learner_trace(ExaSimple())
for i in range(10):
    tr.update()
    tr.snapshot(999)
data = tr.write()
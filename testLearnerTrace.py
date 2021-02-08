import numpy as np
import exarl
from exarl.simple_comm import ExaSimple
from utils.trace_win import Trace_Win
from utils.trace_win import Trace_Win_Up
from utils.trace_win import Trace_Win_Snap

comm = ExaSimple()
test = Trace_Win(name="blah", comm=comm, arrayType=np.int64)
test = Trace_Win(name="blah")
test = Trace_Win(name="blah")

@Trace_Win_Snap("Test1.txt", comm, arrayType=np.float64)
@Trace_Win_Up("Test1.txt", comm, arrayType=np.float64, position=0)
def test1(value, other):
    print(value)
    return value


for i in range(10):
    test1(i, 9.99)
#     # tr.update(10-i)
#     # if i%3==0:
#     #     tr.snapshot()

# Trace_Win_Repo("Test1.txt").write()

Trace_Win.write()
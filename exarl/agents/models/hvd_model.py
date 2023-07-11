# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
import functools
import tensorflow as tf
from exarl.utils.globals import ExaGlobals
from exarl.base.comm_base import ExaComm
_use_hvd = False

try:
    if ExaComm.learner_comm is not None:
        if ExaComm.learner_comm.num_learners >= 2:
            import horovod
            import horovod.tensorflow as hvd
            _use_hvd = True
except Exception as e:
    print("Failed to load Horovod.")
    print(e)

def if_hvd_loaded(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if Horovod_Model.use_hvd and not Horovod_Model.init:
            Horovod_Model.init_horovod_learners()
        # JS: Init could reset so check again.
        # Should now be initialized...
        if Horovod_Model.use_hvd:
            return func(*args, **kwargs)
        return args
    return wrapper

class Horovod_Model:
    use_hvd = _use_hvd
    init = False
    _first = True

    def init_horovod_learners():
        try:
            if ExaComm.learner_comm.num_learners < 2:
                Horovod_Model.use_hvd = False
            else:
                hvd.init(comm=ExaComm.learner_comm.raw())
                Horovod_Model.init = True
        except:
            Horovod_Model.use_hvd = False

    @if_hvd_loaded
    def optimizer(opt):
        opt = hvd.DistributedOptimizer(opt)
        hvd.BroadcastGlobalVariablesHook(0)
        return opt

    @if_hvd_loaded
    def gradient_tape(tape):
        return hvd.DistributedGradientTape(tape)
    
    @if_hvd_loaded
    def first(model, opt):
        if Horovod_Model._first:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)
            Horovod_Model._first = False
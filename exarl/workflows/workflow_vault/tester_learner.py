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
import exarl as erl
import pickle
from exarl.base.comm_base import ExaComm
import tensorflow as tf
import exarl.utils.log as log
import exarl.utils.candleDriver as cd
from exarl.utils.profile import *
from exarl.utils.introspect import *
from exarl.network.simple_comm import ExaSimple
MPI = ExaSimple.MPI

logger = log.setup_logger(__name__, cd.run_params['log_level'])


class TESTER(erl.ExaWorkflow):
    def __init__(self):
        print('Class TESTER learner')
        self.data_file = cd.run_params['tester_data_file']
        self.epocs = cd.run_params['tester_epocs']
        print(self.data_file, self.epocs)

    @PROFILE
    def run(self, workflow):
        @TIMERET
        @introspectTrace(position=1)
        def train_time(batch, epochs=1):
            with tf.device(workflow.agent.device):
                print(batch[0].shape, batch[1].shape)
                workflow.agent.model.fit(batch[0], batch[1], epochs=epochs, verbose=0)

        # This is the stuff we care about
        data = []
        commSize = MPI.COMM_WORLD.Get_size()
        if self.data_file is not None:
            if commSize == 1:
                with open(self.data_file, "rb") as input_file:
                    data = pickle.load(input_file)
                print(len(data))

                workflow.agent.set_learner()

                times=[]
                for i in range(10):
                    times.append(train_time(data[0], self.epocs))
                print("Average Time:", sum(times)/len(times))

            elif commSize == 2:
                comm = ExaComm.agent_comm
                target_weights = None
                if ExaComm.is_learner():
                    workflow.agent.set_learner()
                    target_weights = workflow.agent.get_weights()

                current_weights = comm.bcast(target_weights, 0)
                workflow.agent.set_weights(current_weights)
                if ExaComm.is_agent():
                    done = True
                    for e in range(workflow.agent.batch_size):
                        if done:
                            current_state = workflow.env.reset()
                            total_reward = 0
                            done = False

                        action, _ = workflow.agent.action(current_state)
                        next_state, reward, done, _ = workflow.env.step(action)
                        total_reward += reward
                        memory = (current_state, action, reward, next_state, done, total_reward)
                        workflow.agent.remember(memory[0], memory[1], memory[2], memory[3], memory[4])
                        current_state = next_state

                    data.append(next(workflow.agent.generate_data()))
                    print(workflow.agent.batch_size, len(data), flush=True)
                        
                if len(data) > 0:
                    with open(self.data_file, 'wb') as outfile:
                        pickle.dump(data, outfile)
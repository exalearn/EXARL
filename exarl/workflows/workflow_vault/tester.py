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
import time
import csv
from mpi4py import MPI
import exarl as erl
import exarl.mpi_settings as mpi_settings
import exarl.utils.log as log
import exarl.utils.candleDriver as cd
from exarl.utils.profile import *
logger = log.setup_logger(__name__, cd.run_params['log_level'])


class DEMO(erl.ExaWorkflow):
    def __init__(self):
        print('Class DEMO tester')

    @PROFILE
    def run(self, workflow):
        comm = MPI.COMM_WORLD

        if comm.rank == 0:

            # Set target model the sample for all
            filename = "sync_vtrace.h5"


            workflow.agent.load(filename)



        # Loop over episodes
            for e in range(workflow.nepisodes):
                # Reset variables each episode
                workflow.env.monitor.start('cartpole-demo-ep{}'.format(e),force=True)
                current_state = workflow.env.reset()
                total_reward  = 0
                steps         = 0
                done          = False

                start_time_episode = time.time()

                while done != True and steps != workflow.nsteps:

                    action, policy_type = workflow.agent.action(current_state)
                    next_state, reward, done, _ = workflow.env.step(action)
                    steps += 1

                workflow.env.monitor.close()

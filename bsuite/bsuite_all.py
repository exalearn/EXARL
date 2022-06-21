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
import sys
from bsuite import sweep

subsets = {"all": ["bandit", "bandit_noise", "bandit_scale",
                   "cartpole", "cartpole_noise", "cartpole_scale", "cartpole_swingup",
                   "catch", "catch_noise", "catch_scale",
                   "deep_sea", "deep_sea_stochastic",
                   "discounting_chain",
                   "memory_len", "memory_size",
                   "mnist", "mnist_noise", "mnist_scale",
                   "mountain_car", "mountain_car_noise", "mountain_car_scale",
                   "umbrella_distract", "umbrella_length"],
           "working": ["bandit", "bandit_noise", "bandit_scale",
                       "cartpole", "cartpole_noise", "cartpole_scale",
                       "catch", "catch_noise", "catch_scale",
                       "deep_sea", "deep_sea_stochastic",
                       "discounting_chain",
                       "memory_len", "memory_size",
                       "mnist", "mnist_noise", "mnist_scale",
                       "umbrella_distract", "umbrella_length"],
           "developer": ["cartpole", "cartpole_noise", "bandit"],
           "basic": ["bandit", "mnist", "catch", "mountain_car", "cartpole"],
           "noise": ["bandit_noise", "mnist_noise", "catch_noise", "mountain_car_noise", "cartpole_noise"],
           "scale": ["bandit_scale", "mnist_scale", "catch_scale", "mountain_car_scale", "cartpole_scale"],
           "exploration": ["deep_sea", "deep_sea_stochastic", "cartpole_swingup"],
           "credit_assignment": ["umbrella_length", "umbrella_distract", "discounting_chain"],
           "memory": ["memory_len", "memory_size"],
           "quick_basic": ["bandit", "catch", "discounting_chain"],
           "dynamics_learning": ["cartpole_swingup", "deep_sea", "discounting_chain", "memory_len", "memory_size", "umbrella_length", "umbrella_distract"],
           "cartpole": ["cartpole", "cartpole_noise", "cartpole_scale"],
           "cartpole_only": ["cartpole"],
           "cartpole_noise": ["cartpole_noise"],
           "cartpole_scale": ["cartpole_scale"],
           "cartpole_swingup": ["cartpole_swingup"],
           "catch": ["catch", "catch_noise", "catch_scale"],
           "catch_only": ["catch"],
           "catch_noise": ["catch_noise"],
           "catch_scale": ["catch_scale"],
           "mountain_car_only": ["mountain_car"],
           "mountain_car_noise": ["mountain_car_noise"],
           "mountain_car_scale": ["mountain_car_scale"],
           "deep": ["deep_sea", "deep_sea_stochastic"],
           "umbrella": ["umbrella_length"],
           "umb_dist": ["umbrella_distract"],
           "discount": ["discounting_chain"],
           "empty": []}


def parse_entry(entry):
    temp = entry.split("/")
    return temp[0], int(temp[1]) + 1

def get_all(filter):
    ret = {}
    for entry in sweep.SWEEP:
        name, seed = parse_entry(entry)
        reps = sweep.EPISODES[entry]
        if filter is None or name in subsets[filter]:
            if name in ret:
                seed = max([ret[name][0], seed])
                reps = max([ret[name][1], reps])
            ret[name] = (seed, reps)
    return ret


if __name__ == "__main__":
    filter = "all"
    if len(sys.argv) == 2:
        filter = str(sys.argv[1])

    if filter == "display":
        for sub in subsets:
            if sub != "empty":
                print(sub)
    else:
        if filter not in subsets:
            filter = "empty"
        envs = get_all(filter)
        for i in envs:
            print(i, envs[i][0], envs[i][1])

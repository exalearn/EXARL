import json

class ExaEnv():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'env_cfg':
                self.env_cfg = value
            else:
                self.env_cfg = 'envs/env_vault/env_cfg/env_setup.json'

        with open(self.env_cfg) as json_file:
            env_data = json.load(json_file)

        ## TODO: We need to define defaults for MPI, GPU, OMP, etc. and exit code if invalid parameters ##
        self.num_child_per_parent = int(env_data['child_spawn_per_parent']) if 'child_spawn_per_parent' in env_data.keys() else 0
        if(self.num_child_per_parent > 0):
            # defaults to running toy example of computing PI
            self.worker = (env_data['worker_app']).lower() if 'worker_app' in env_data.keys() else "envs/env_vault/cpi.py"
        else:
            self.worker = None

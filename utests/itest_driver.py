import subprocess
import pytest

class TestClass:
    def test_driver_default_env(self) -> None:
        assert subprocess.run(['mpirun', '--version']).returncode == 0, 'mpirun must be on path!'

        p = subprocess.run([
            'mpirun', '-np', '2',
            'python', './driver/driver.py',
            '--n_episodes', '50', '--n_steps', '5',
            '--workflow', 'async', '--agent', 'DQN-v0',
        ], capture_output=True)
        assert p.returncode == 0

# conftest.py

def pytest_addoption(parser):
    """
    These are the options for all pytests.
    See individual tests for meanings.
    """
    parser.addoption("--test_env_name", action="store", default=None)
    parser.addoption("--test_env_class", action="store", default=None)
    parser.addoption("--test_env_file", action="store", default=None)
    parser.addoption("--test_agent_name", action="store", default="DQN-v0")
    parser.addoption("--test_agent_class", action="store", default="DQN")
    parser.addoption("--test_agent_file", action="store", default="dqn")
    parser.addoption("--on-policy", action="store", default=-1)
    parser.addoption("--behind", action="store", default=-1)
    parser.addoption("--rank_sleep", action="store_true", default=False)
    parser.addoption("--random_sleep", action="store_true", default=False)
    parser.addoption("--test_save_load_dir", action="store", default='./save_load_dir')
    parser.addoption("--mpi4py_rc", action="store_false", default=True)

# conftest.py

def pytest_addoption(parser):
    # parser.addoption("--test_env_name", action="store", default="ExaCartPoleStatic-v0")
    # parser.addoption("--test_env_class", action="store", default="ExaCartpoleStatic")
    # parser.addoption("--test_env_file", action="store", default="ExaCartpoleStatic")
    parser.addoption("--test_env_name", action="store", default=None)
    parser.addoption("--test_env_class", action="store", default=None)
    parser.addoption("--test_env_file", action="store", default=None)
    parser.addoption("--test_agent_name", action="store", default="DQN-v0")
    parser.addoption("--test_agent_class", action="store", default="DQN")
    parser.addoption("--test_agent_file", action="store", default="dqn")
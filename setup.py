from setuptools import setup

setup(name='exarl', 
      version='0.0.1',
      description='A scalable software framework for reinforcement learning environments and agents/policies used for the Design and Control applications',
      url='https://github.com/exalearn/ExaRL/',
      license='BSD-3',
      packages=['exarl', 'agents', 'envs'],
      zip_safe=False)
      
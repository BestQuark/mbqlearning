# Using RL in MBQC

Project by Luis Mantilla, Dmytro Bondarenko, Polina Feldmann, and Robert Raussendorf.

This code implements two RL agents: Q-learning agent and a PPO agent, to learn measurement patterns in an MBQC simulator.

### Q-learning 
The functions that implement this agent and its environment are in [MBQsimul.jl](MBQsimul.jl). 
We use these functions and solve some examples in [qlearning-MBQC.ipynb](qlearning-MBQC.ipynb).

### PPO
The functions that implement this agent and its environment are in [env_mbqc.py](env_mbqc.py).
We use these functions and solve some examples in [ppo-MBQC.ipynb](ppo-MBQC.ipynb).


### TODO:
- [ ] Currently, the flow function should be defined in the output qubits to be something, but this is never used. Task: fix this such that a user can specify a function with the correct domain f: measured_qubits --> qubits. 


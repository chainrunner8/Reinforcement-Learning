Please see requirements.txt for the required packages to run DQN_TN_ER.py

Open a terminal, navigate to the folder where DQN_TN_ER.py is and run: 

```python DQN_TN_ER.py```

This will perform the very last experiment presented in the paper: naive model, TN-only model, ER-only model and TN+ER model with 5 trials for each.

Note: this file runs on 8 vectorised environments by default because my Ryzen CPU has 8 cores, so you need a CPU with 8 cores. If your CPU has a different amount of cores, you can pass that number as a sys argument after the file name, for example with 16 cores:

```python DQN_TN_ER.py 16```

This will run the file on 16 vectorised environments to speed up training. You can also put in a smaller number of cores.
All the models were trained on the CPU because my GPU (RTX 3060 140w) did not speed up training, in fact it slowed it, perhaps because CartPole isn't computationally complex enough for the GPU to make a difference.
PPO.py runs the experiments from the report.
bufferClass.py contains the class for the buffer object used in PPO.py to log and manage data collected from the environment.

The required packages are contained in requirements.txt.

In order to replicate Figure 1 from the report:

```python PPO.py --experiment "hyper tuning" ```

In order to replicate the first grid search from the report:

```python PPO.py --experiment "first grid" ```

In order to replicate the second grid search from the report:

```python PPO.py --experiment "second grid" ```

In order to replicate Figure 2 from the report:

```python PPO.py --experiment "ablation" ```

In order to replicate Figure 3 from the report:

```python PPO.py --experiment "engineering" ```

Please be aware that despite the fact that the programme uses multiprocessing to run the different algorithms in parallel, these experiments take some time to complete since they carry out 8 trials over a million steps for each configuration.
The programme can detect the number of cpu cores of the machine it's being run on.
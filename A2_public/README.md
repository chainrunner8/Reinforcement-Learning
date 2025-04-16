REINFORCE.py contains the functions needed to run the REINFORCE algorithm.
AC.py contains the functions needed to run the AC algorithm.
A2C.py contains the functions needed to run the A2C algorithm.
main.py needs to be run in order to run the experiments of our report.

The required packages are contained in requirements.txt.

In order to replicate the REINFORCE vs AC vs A2C experiment from question 2.4 (Figure 1 in the report), run the following line in a terminal:

```python main.py q2.4```

In order to replicate the sensitivity analysis experiment on the n parameter (Figures 2 & 3 in the report), run the following line in a terminal:

```python main.py sensitivity```

In order to train and render A2C with the optimal parameter n = 250 found in the report, run the following line in a terminal:

```python A2C.py```

Please be aware that despite the fact that our code uses multiprocessing to run the different algorithms in parallel, these experiments take a long time to run since they carry out 20 trials over a million steps for each configuration.
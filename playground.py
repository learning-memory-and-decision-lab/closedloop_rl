import sys, os

import jax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import rl_model
from resources.model_evaluation import plot_traces


if __name__=='__main__':
    var = jax.numpy.array(False, dtype=float)
    
    print(var)
    if var:
        print('Passed')
    else:
        print('NONONON')
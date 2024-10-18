import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

rnn_main.main(
    checkpoint=True,
    epochs=0,
    sigma=0.1,
    # n_sessions=4096*4,
    # n_trials_per_session=16,
    batch_size=128,
    n_steps_per_call=16,
    
    n_submodels=1,
    dropout=0.1,
    bagging=False,
    learning_rate=0.01,
    # weight_decay=0.001,
    
    alpha=0.25,
    forget_rate=0.,
    regret=False,
    confirmation_bias=False,
    perseveration_bias=0.25,
    beta=5,
    # directed_exploration_bias=1.,
    undirected_exploration_bias=5.,
    
    analysis=True,
)
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

_, loss = rnn_main.main(
    checkpoint=False,
    epochs=256,
    
    # data='data/study_recovery_s02/data_rldm_16p_0.csv',
    # model='params/study_recovery_s02/params_rldm_16p_0.pkl',
    
    # model=f'params/benchmarking/rnn_eckstein.pkl',
    # data = 'data/2arm/eckstein2022_291_processed.csv',
    
    # model = f'params/benchmarking/rnn_sugawara.pkl',
    # data = 'data/2arm/sugawara2021_143_processed.csv',
    
    n_actions=2,
    
    dropout=0.25,
    participant_emb=True,
    bagging=True,

    learning_rate=1e-2,
    batch_size=-1,
    sequence_length=-1,#64,
    train_test_ratio=0,
    n_steps=16,
    
    n_sessions=32,
    n_trials=256,
    sigma=0.2,
    beta_reward=3.,#3.,
    alpha_reward=0.25,#0.25,
    alpha_penalty=0.5,#0.5,
    forget_rate=0.,#0.2,
    confirmation_bias=0.,#0.5,
    beta_choice=1.,
    alpha_choice=0.,
    counterfactual=False,#True,
    alpha_counterfactual=0.,#0.5,
    # parameter_variance=0.,
    
    analysis=True,
    session_id=3,
)
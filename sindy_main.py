import sys
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

sys.path.append('resources')
from resources.rnn import RLRNN
from resources.bandits import AgentQ, AgentSindy, AgentNetwork, BanditsDrift, BanditsSwitch, plot_session, create_dataset as create_dataset_bandits
from resources.sindy_utils import create_dataset, check_library_setup
from resources.rnn_utils import parameter_file_naming
from resources.sindy_training import fit_model
from utils.convert_dataset import convert_dataset
from utils.plotting import session as plot_session

warnings.filterwarnings("ignore")

def main(
    model: str = None,
    data: str = None,
    
    # generated training dataset parameters
    n_trials = 256,
    participant_id: int = None,
    
    # sindy parameters
    threshold = 0.03,
    polynomial_degree = 1,
    regularization = 1e-2,
    verbose = True,
    
    # ground truth parameters
    beta_reward = 3.,
    alpha = 0.25,
    alpha_penalty = -1.,
    forget_rate = 0.,
    confirmation_bias = 0.,
    beta_choice = 0.,
    alpha_choice = 0.,
    alpha_counterfactual = 0.,
    parameter_variance = 0.,
    reward_prediction_error: Callable = None,
    
    # environment parameters
    n_actions = 2,
    sigma = .1,
    counterfactual = False,
    
    analysis: bool = False, 
    ):

    # ---------------------------------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------------------------------
    
    # tracked variables and control signals in the RNN
    module_list = ['x_V_LR', 'x_V_nc', 'x_C', 'x_C_nc']
    control_list = ['c_a', 'c_r', 'c_V']
    sindy_feature_list = module_list + control_list

    # library setup: 
    # which terms are allowed as control inputs in each SINDy model
    # key is the SINDy model name, value is a list of allowed control inputs from the list of control signals 
    library_setup = {
        'x_V_LR': ['c_V', 'c_r'],
        'x_V_nc': ['c_r'],
        'x_C': [],
        'x_C_nc': [],
    }

    # data-filter setup: 
    # which samples are allowed as training samples in each SINDy model based on the given filter condition (conditions are always equality conditions)
    # key is the SINDy model name, value is a list with a triplet of values:
    #   1. str: feature name to be used as a filter
    #   2. numeric: the numeric filter condition
    #   3. bool: remove feature from control inputs --> TODO: check if this is necessary or makes things just more complicated
    # Multiple conditions can also be given as a list of triplets.
    # Example:
    #   'x_V_nc': ['c_a', 0, True] means that for the SINDy model 'x_V_nc', only samples where the feature 'c_a' == 0 are used for training the SINDy model. 
    #   The control parameter 'c_a' is removed afterwards from the list of control signals for training of the model
    datafilter_setup = {
        'x_V_LR': ['c_a', 1, True],
        'x_V_nc': ['c_a', 0, True],
        'x_C': ['c_a', 1, True],
        'x_C_nc': ['c_a', 0, True],
    }

    # data pre-processing setup:
    # define the processing steps for each variable and control signal.
    # possible processing steps are: 
    #   Trimming: Remove the first 25% of the samples along the time-axis. This is useful if the RNN begins with a variable at 0 but then accumulates first first to a specific default value, i.e. the range changes from (0, p) to (q, q+p). That way the data is cleared of the accumulation process. Trimming will be active for all variables, if it is active for one. 
    #   Offset-Clearing: Clearup any offset by determining the minimal value q of a variable and move the value range from (q, q+p) -> (0, p). This step makes SINDy equations less complex and aligns them more with RL-Theory
    #   Normalization: Scale the value range of a variable to x_max - x_min = 1. Offset-Clearing is recommended to achieve a value range of (0, 1) 
    # The processing steps are passed in the form of a binary triplet in this order: (Trimming, Offset-Clearing, Normalization) 
    dataprocessing_setup = {
        'x_V_LR': [1, 0, 0],
        'x_V_nc': [1, 0, 0],
        'x_C': [1, 1, 0],
        'x_C_nc': [1, 1, 0],
        # 'c_a': [0, 0, 0],
        # 'c_r': [0, 0, 0],
        'c_V': [1, 0, 0],
    }
    
    if not check_library_setup(library_setup, sindy_feature_list, verbose=True):
        raise ValueError('Library setup does not match feature list.')
    
    # ---------------------------------------------------------------------------------------------------
    # Data setup
    # ---------------------------------------------------------------------------------------------------
    
    agent = None
    participant_ids = None
    if data is None:
        # set up ground truth agent and environment
        environment = BanditsDrift(sigma=sigma, n_actions=n_actions, counterfactual=counterfactual)
        # environment = EnvironmentBanditsSwitch(sigma, n_actions=n_actions, counterfactual=counterfactual)
        agent = AgentQ(
            n_actions=n_actions, 
            beta_reward=beta_reward, 
            alpha_reward=alpha, 
            alpha_penalty=alpha_penalty, 
            beta_choice=beta_choice, 
            alpha_choice=alpha_choice, 
            forget_rate=forget_rate, 
            confirmation_bias=confirmation_bias, 
            alpha_counterfactual=alpha_counterfactual,
            )
        if reward_prediction_error is not None:
            agent.set_reward_prediction_error(reward_prediction_error)
        _, experiment_list_test, _ = create_dataset_bandits(agent, environment, 100, 1)
        _, experiment_list_train, _ = create_dataset_bandits(agent, environment, n_trials, 1)
        participant_ids = np.arange(len(experiment_list_train))
    else:
        # get data from experiments for later evaluation
        _, experiment_list_test, df, _ = convert_dataset(data)
        participant_ids = np.unique(np.array([experiment_list_test[i].session[0] for i in range(len(experiment_list_test))])).tolist()
        
        # set up environment to run with trained RNN to collect data
        environment = BanditsDrift(sigma=sigma, n_actions=n_actions, counterfactual=counterfactual) # TODO: compute counterfactual from data based on rewards       
        agent_dummy = AgentQ(n_actions=n_actions, alpha_reward=0.5, beta_reward=1.0)
        experiment_list_train = create_dataset_bandits(agent=agent_dummy, environment=environment, n_trials=n_trials, n_sessions=1)[1]
    
    # ---------------------------------------------------------------------------------------------------
    # RNN Setup
    # ---------------------------------------------------------------------------------------------------
    
    # set up rnn agent and expose q-values to train sindy
    if model is None:
        params_path = parameter_file_naming('params/params', beta_reward=beta_reward, alpha_reward=alpha, alpha_penalty=alpha_penalty, beta_choice=beta_choice, alpha_choice=alpha_choice, forget_rate=forget_rate, confirmation_bias=confirmation_bias, alpha_counterfactual=alpha_counterfactual, variance=parameter_variance, verbose=True)
    else:
        params_path = model
    state_dict = torch.load(params_path, map_location=torch.device('cpu'))['model']
    # participant_embedding_index = [i for i, s in enumerate(list(state_dict.keys())) if 'participant_embedding' in s]
    # participant_embedding_bool = True if len(participant_embedding_index) > 0 else False
    # n_participants = 0 if not participant_embedding_bool else state_dict[list(state_dict.keys())[participant_embedding_index[0]]].shape[0]
    n_participants = len(participant_ids)
    key_hidden_size = [key for key in state_dict if 'x' in key.lower()][0]  # first key that contains the hidden_size
    hidden_size = state_dict[key_hidden_size].shape[0]
    rnn = RLRNN(
        n_actions=n_actions, 
        hidden_size=hidden_size,
        n_participants=n_participants, 
        list_sindy_signals=sindy_feature_list, 
        )
    print('Loaded model ' + params_path)
    rnn.load_state_dict(state_dict)
    agent_rnn = AgentNetwork(rnn, n_actions, deterministic=True)
    
    # ---------------------------------------------------------------------------------------------------
    # SINDy training
    # ---------------------------------------------------------------------------------------------------
    
    sindy_models = {module: {} for module in module_list}
    range_participant_id = [participant_id] if participant_id is not None else participant_ids
    for pid in range_participant_id:
        # get SINDy-formatted data with exposed latent variables computed by RNN-Agent
        variables, control, feature_names, beta_scaling = create_dataset(
            agent=agent_rnn,
            data=experiment_list_train,
            # data=environment,
            n_trials=n_trials,
            n_sessions=1,
            participant_id=pid,
            shuffle=True,
            dataprocessing=dataprocessing_setup,
            )
        
        # fit SINDy models -> One model per module
        sindy_models_pid = fit_model(
            variables=variables, 
            control=control, 
            feature_names=feature_names, 
            polynomial_degree=polynomial_degree, 
            library_setup=library_setup, 
            filter_setup=datafilter_setup, 
            verbose=verbose, 
            get_loss=False, 
            optimizer_threshold=threshold, 
            optimizer_alpha=regularization,
            )
        
        for model in module_list:
            sindy_models[model][pid] = sindy_models_pid[model]
        
    # set up SINDy-Agent -> One SINDy-Agent per session if participant embedding is activated
    agent_sindy = AgentSindy(
        model=agent_rnn._model,
        sindy_modules=sindy_models,
        n_actions=n_actions,
        deterministic=True, 
        )
    
    # if verbose:
    #     print(f'SINDy Beta: {agent_rnn._model._beta_reward.item():.2f} and {agent_rnn._model._beta_choice.item():.2f}')
    #     print('Calculating RNN and SINDy loss in X (predicting behavior; Target: Subject)...', end='\r')
    #     test_loss_rnn_x = bandit_loss(agent_rnn, experiment_list_test, coordinates="x")
    #     test_loss_sindy_x = bandit_loss(agent_sindy, experiment_list_test, coordinates="x")
    #     print(f'RNN Loss in X: {test_loss_rnn_x}')
    #     print(f'SINDy Loss in X: {test_loss_sindy_x}')

    # ---------------------------------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------------------------------
    
    if analysis:
        
        participant_id_test = [participant_id] if participant_id is not None else [participant_ids[0]]
        
        if experiment_list_test is None:
            experiment_list_test = [experiment_list_train[participant_id_test[0]]]
        
        agent_rnn.new_sess(participant_id=participant_id_test[0])
        agent_sindy.new_sess(participant_id=participant_id_test[0])
        
        # print sindy equations from tested sindy agent
        print('\nDiscovered SINDy models:')
        for model in module_list:
            agent_sindy._model.submodules_sindy[model][participant_id_test[0]].print()
        print(f'(x_beta_r) = {agent_sindy._beta_reward:.3f}')
        print(f'(x_beta_c) = {agent_sindy._beta_choice:.3f}')
        print('\n')
        
        # set up ground truth agent by getting parameters from dataset if specified
        if data is not None and agent is None and analysis and 'mean_beta_reward' in df.columns:
            agent = AgentQ(
                beta_reward = df['beta_reward'].values[(df['session']==participant_id_test[0]).values][0],
                alpha_reward = df['alpha_reward'].values[(df['session']==participant_id_test[0]).values][0],
                alpha_penalty = df['alpha_penalty'].values[(df['session']==participant_id_test[0]).values][0],
                confirmation_bias = df['confirmation_bias'].values[(df['session']==participant_id_test[0]).values][0],
                forget_rate = df['forget_rate'].values[(df['session']==participant_id_test[0]).values][0],
                beta_choice = df['beta_choice'].values[(df['session']==participant_id_test[0]).values][0],
                alpha_choice = df['alpha_choice'].values[(df['session']==participant_id_test[0]).values][0],
            )
        
        # get analysis plot
        if agent is not None:
            agents = {'groundtruth': agent, 'rnn': agent_rnn, 'sindy': agent_sindy}
            plt_title = r'$GT:\beta_{reward}=$'+str(np.round(agent._beta_reward, 2)) + r'; $\beta_{choice}=$'+str(np.round(agent._beta_choice, 2))+'\n'
        else:
            agents = {'rnn': agent_rnn, 'sindy': agent_sindy}
            plt_title = ''
            
        experiment_analysis = experiment_list_test[0]
        fig, axs = plot_session(agents, experiment_analysis)
        plt_title += r'$SINDy:\beta_{reward}=$'+str(np.round(agent_sindy._beta_reward, 2)) + r'; $\beta_{choice}=$'+str(np.round(agent_sindy._beta_choice, 2))
        
        fig.suptitle(plt_title)
        plt.show()
        
    features = {}
    for model in sindy_models:
        features[model] = {}
        for pid in participant_ids:
            features_i = sindy_models[model][pid].get_feature_names()
            coeffs_i = [c for c in sindy_models[model][pid].coefficients()[0]]
            index_u = []
            for i, f in enumerate(features_i):
                if 'dummy' in f:
                    index_u.append(i)
            features_i = [item for idx, item in enumerate(features_i) if idx not in index_u]
            coeffs_i = [item for idx, item in enumerate(coeffs_i) if idx not in index_u]
            features[model][pid] = tuple(features_i)
            features[model][pid] = tuple(coeffs_i)
    
    features['beta_reward'] = {}
    features['beta_choice'] = {}
    for pid in participant_ids:
        agent_sindy.new_sess(participant_id=pid)
        features['beta_reward'][pid] = agent_sindy._beta_reward
        features['beta_choice'][pid] = agent_sindy._beta_choice
        
    return agent_sindy, sindy_models, features


if __name__=='__main__':
    main(
        model = 'params/benchmarking/rnn_sugawara.pkl',
        data = 'data/2arm/sugawara2021_143_processed.csv',
        n_trials=None,
        n_sessions=None,
        verbose=False,
        
        # sindy parameters
        polynomial_degree=2,
        threshold=0.05,
        regularization=0,

        # generated training dataset parameters
        # n_trials_per_session = 200,
        # n_sessions = 100,
        
        # ground truth parameters
        # alpha = 0.25,
        # beta = 3,
        # forget_rate = 0.,
        # perseverance_bias = 0.25,
        # alpha_penalty = 0.5,
        # confirmation_bias = 0.5,
        # reward_update_rule = lambda q, reward: reward-q,
        
        # environment parameters
        # sigma = 0.1,
        
        analysis=True,
    )
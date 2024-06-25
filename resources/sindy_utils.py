import numpy as np
from typing import Iterable, List, Dict, Tuple

import pysindy as ps

from bandits import *
from resources.rnn import EnsembleRNN, BaseRNN

def make_sindy_data(
    dataset,
    agent,
    sessions=-1,
    ):

  # Get training data for SINDy
  # put all relevant signals in x_train

  if not isinstance(sessions, Iterable) and sessions == -1:
    # use all sessions
    sessions = np.arange(len(dataset))
  else:
    # use only the specified sessions
    sessions = np.array(sessions)
    
  n_control = 2
  
  choices = np.stack([dataset[i].choices for i in sessions], axis=0)
  rewards = np.stack([dataset[i].rewards for i in sessions], axis=0)
  qs = np.stack([dataset[i].q for i in sessions], axis=0)
  
  choices_oh = np.zeros((len(sessions), choices.shape[1], agent._n_actions))
  for sess in sessions:
    # one-hot encode choices
    choices_oh[sess] = np.eye(agent._n_actions)[choices[sess]]
    
  # concatenate all qs values of one sessions along the trial dimension
  qs_all = np.concatenate([np.stack([np.expand_dims(qs_sess[:, i], axis=-1) for i in range(agent._n_actions)], axis=0) for qs_sess in qs], axis=0)
  c_all = np.concatenate([np.stack([c_sess[:, i] for i in range(agent._n_actions)], axis=0) for c_sess in choices_oh], axis=0)
  r_all = np.concatenate([np.stack([r_sess for _ in range(agent._n_actions)], axis=0) for r_sess in rewards], axis=0)
  
  # get observed dynamics
  x_train = qs_all
  feature_names = ['q']

  # get control
  control_names = []
  control = np.zeros((*x_train.shape[:-1], n_control))
  control[:, :, 0] = c_all
  control_names += ['c']
  control[:, :, n_control-1] = r_all
  control_names += ['r']
  
  feature_names += control_names
  
  print(f'Shape of Q-Values is: {x_train.shape}')
  print(f'Shape of control parameters is: {control.shape}')
  print(f'Feature names are: {feature_names}')
  
  # make x_train and control sequences instead of arrays
  x_train = [x_train_sess for x_train_sess in x_train]
  control = [control_sess for control_sess in control]
 
  return x_train, control, feature_names


def create_dataset(
  agent: AgentNetwork,
  environment: Environment,
  n_trials_per_session: int,
  n_sessions: int,
  normalize: bool = False,
  shuffle: bool = False,
  ):
  
  keys_x = [key for key in agent._model.history.keys() if key.startswith('x')]
  keys_c = [key for key in agent._model.history.keys() if key.startswith('c')]
  
  # x_train = np.zeros((n_sessions*agent._n_actions*(n_trials_per_session-1), 2, len(keys_x)))
  # control = np.zeros((n_sessions*agent._n_actions*(n_trials_per_session-1), 2, len(keys_c)))
  x_train = {key: [] for key in keys_x}
  control = {key: [] for key in keys_c}
  
  for session in range(n_sessions):
    agent.new_sess()
    
    for trial in range(n_trials_per_session):
      # generate trial data
      choice = agent.get_choice()
      reward = environment.step(choice)
      agent.update(choice, reward)
    
    # sort the data of one session into the corresponding signals
    for key in agent._model.history.keys():
      if len(agent._model.history[key]) > 1:
        # TODO: resolve ugly workaround with class distinction
        history = agent._model.history[key]
        if isinstance(agent._model, EnsembleRNN):
          history = history[-1]
        values = np.concatenate(history)
        if key in keys_x:
          # add values of interest of one session as trajectory
          if normalize:
            value_min = np.min(values)
            value_max = np.max(values)
            values = (values - value_min) / (value_max - value_min)
          for i_action in range(agent._n_actions):
            x_train[key] += [v for v in values[:, :, i_action]]
        if key in keys_c:
          # add control signals of one session as corresponding trajectory
          if values.shape[-1] == 1:
            values = np.repeat(values, 2, -1)
          for i_action in range(agent._n_actions):
            control[key] += [v for v in values[:, :, i_action]]
              
  # get all keys of x_train and control that have no values and remove them
  keys_x = [key for key in keys_x if len(x_train[key]) > 0]
  keys_c = [key for key in keys_c if len(control[key]) > 0]
  x_train = {key: x_train[key] for key in keys_x}
  control = {key: control[key] for key in keys_c}
  feature_names = keys_x + keys_c
  
  # make x_train and control List[np.ndarray] with shape (n_trials_per_session-1, len(keys)) instead of dictionaries
  x_train_list = []
  control_list = []
  for i in range(len(control[keys_c[0]])):
    x_train_list.append(np.stack([x_train[key][i] for key in keys_x], axis=-1))
    control_list.append(np.stack([control[key][i] for key in keys_c], axis=-1))
  
  if shuffle:
    shuffle_idx = np.random.permutation(len(x_train_list))
    x_train_list = [x_train_list[i] for i in shuffle_idx]
    control_list = [control_list[i] for i in shuffle_idx]
  
  return x_train_list, control_list, feature_names


def optimize_beta(experiment, agent: AgentNetwork, agent_sindy: AgentSindy, plot=False):
  # fit beta parameter of softmax by fitting on choice probability of the RNN by simple grid search

  # number of observed points
  n_points = 100

  # get choice probabilities of the RNN
  _, choice_probs_rnn = get_update_dynamics(experiment, agent)

  # set prior for beta parameter; x_max seems to be a good starting point
  # beta_range = np.linspace(x_max-1, x_max+1, n_points)
  beta_range = np.linspace(1, 10, n_points)

  # get choice probabilities of the SINDy agent for each beta in beta_range
  choice_probs_sindy = np.zeros((len(beta_range), len(choice_probs_rnn), agent._n_actions))
  for i, beta in enumerate(beta_range):
      agent_sindy._beta = beta
      _, choice_probs_sindy_beta = get_update_dynamics(experiment, agent_sindy)
      
      # add choice probabilities to choice_probs_sindy
      choice_probs_sindy[i, :, :] = choice_probs_sindy_beta
      
  # get best beta value by minimizing the error between choice probabilities of the RNN and the SINDy agent
  errors = np.zeros(len(beta_range))
  for i in range(len(beta_range)):
      errors[i] = np.sum(np.abs(choice_probs_rnn - choice_probs_sindy[i]))

  # get right beta value
  beta = beta_range[np.argmin(errors)]

  if plot:
    # plot error plot with best beta value in title
    plt.plot(beta_range, errors)
    plt.title(f'Error plot with best beta={beta}')
    plt.xlabel('Beta')
    plt.ylabel('MAE')
    plt.show()

  return beta


def check_library_setup(library_setup: Dict[str, List[str]], feature_names: List[str], verbose=False) -> bool:
  msg = '\n'
  for key in library_setup.keys():
    if key not in feature_names:
      msg += f'Key {key} not in feature_names.\n'
    else:
      for feature in library_setup[key]:
        if feature not in feature_names:
          msg += f'Key {key}: Feature {feature} not in feature_names.\n'
  if msg != '\n':
    msg += f'Valid feature names are {feature_names}.\n'
    print(msg)
    return False
  else:
    if verbose:
      print('Library setup is valid. All keys and features appear in the provided list of features.')
    return True
  

def remove_control_features(control_variables: List[np.ndarray], feature_names: List[str], target_feature_names: List[str]) -> List[np.ndarray]:
  control_new = []
  for control in control_variables:
    remaining_control_variables = [control[:, feature_names.index(feature)] for feature in target_feature_names]
    if len(remaining_control_variables) > 0:
      control_new.append(np.stack(remaining_control_variables, axis=-1))
    else:
      control_new = None
      break
  return control_new


def conditional_filtering(x_train: List[np.ndarray], control: List[np.ndarray], feature_names: List[str], relevant_feature: str, condition: float, remove_relevant_feature=True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  x_train_relevant = []
  control_relevant = []
  x_features = [feature for feature in feature_names if feature.startswith('x')]
  control_features = [feature for feature in feature_names if feature.startswith('c')]
  for i, x, c in zip(range(len(x_train)), x_train, control):
    if relevant_feature in feature_names:
      i_relevant = control_features.index(relevant_feature)
      if c[0, i_relevant] == condition:
        x_train_relevant.append(x)
        control_relevant.append(c)
        if remove_relevant_feature:
          control_relevant[-1] = np.delete(control_relevant[-1], i_relevant, axis=-1)
  
  if remove_relevant_feature:
    control_features.remove(relevant_feature)
    
  return x_train_relevant, control_relevant, x_features+control_features


def setup_library(library_setup: Dict[str, List[str]]) -> Dict[str, Tuple[ps.feature_library.base.BaseFeatureLibrary, List[str]]]:
  libraries = {key: None for key in library_setup.keys()}
  feature_names = {key: [key] + library_setup[key] for key in library_setup.keys()}
  for key in library_setup.keys():
      library = ps.PolynomialLibrary(degree=2)
      library.fit(np.random.rand(10, len(feature_names[key])))
      print(library.get_feature_names_out(feature_names[key]))
      libraries[key] = (library, feature_names[key])
  
  ps.ConcatLibrary([libraries[key][0] for key in libraries.keys()])
  
  return libraries


def constructor_update_rule_sindy(sindy_models):
  def update_rule_sindy(q, h, choice, prev_choice, reward):
      # mimic behavior of rnn with sindy
      if choice == 0:
          # blind update for non-chosen action
          q_update = sindy_models['xQf'].simulate(q, t=2, u=np.array([0]).reshape(1, 1))[-1]
      elif choice == 1:
          # reward-based update for chosen action
          q_update = sindy_models['xQr'].simulate(q, t=2, u=np.array([reward]).reshape(1, 1))[-1]
      if prev_choice == 1:
        h = sindy_models['xH'].simulate(q, t=2, u=np.array([prev_choice]).reshape(1, 1))[-1] - q  # get only the difference between q and q_update
      return q_update, h
    
  return update_rule_sindy
#@title Import libraries
import sys
import os

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import argparse

# warnings.filterwarnings("ignore")

# RL libraries
sys.path.append('resources')  # add source directoy to path
from resources import rnn, rnn_training, bandits, rnn_utils
from utils import convert_dataset, plotting

def main(
  checkpoint = False,
  model: str = None,
  data: str = None,

  # rnn parameters
  hidden_size = 8,
  dropout = 0.1,
  participant_emb = False,

  # data and training parameters
  epochs = 128,
  train_test_ratio = 0.,
  n_trials = 256,
  n_sessions = 256,
  bagging = False,
  sequence_length = 64,
  n_steps = 16,  # -1 for full sequence
  batch_size = -1,  # -1 for one batch per epoch
  learning_rate = 1e-2,
  convergence_threshold = 1e-6,
  parameter_variance = 0.,
  
  # ground truth parameters
  beta_reward = 3.,
  alpha_reward = 0.25,
  alpha_penalty = -1.,
  alpha_counterfactual = 0.,
  beta_choice = 0.,
  alpha_choice = 0.,
  forget_rate = 0.,
  confirmation_bias = 0.,
  reward_prediction_error: Callable = None,
  
  # environment parameters
  n_actions = 2,
  sigma = 0.1,
  counterfactual = False,
  
  analysis: bool = False,
  session_id: int = 0
  ):
  
  # print cuda devices available
  print(f'Cuda available: {torch.cuda.is_available()}')
  
  if not os.path.exists('params'):
    os.makedirs('params')
  
  # tracked variables in the RNN
  x_train_list = ['x_V_LR', 'x_V', 'x_V_nc', 'x_C', 'x_C_nc']
  control_list = ['c_a', 'c_r', 'c_V']
  sindy_feature_list = x_train_list + control_list

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  dataset_test = None
  agent = None
  if data is None:
    print('No path to dataset provided.')
    
    # setup
    environment = bandits.BanditsDrift(sigma, n_actions, counterfactual=counterfactual)
    # environment = bandits.EnvironmentBanditsSwitch(sigma, counterfactual=counterfactual)
    agent = bandits.AgentQ(
      n_actions=n_actions, 
      beta_reward=beta_reward, 
      alpha_reward=alpha_reward, 
      alpha_penalty=alpha_penalty, 
      forget_rate=forget_rate, 
      confirmation_bias=confirmation_bias, 
      beta_choice=beta_choice, 
      alpha_choice=alpha_choice,
      alpha_counterfactual=alpha_counterfactual, 
      parameter_variance=parameter_variance,
      )
    if reward_prediction_error is not None:
      agent.set_reward_prediction_error(reward_prediction_error)
    print('Setup of the environment and agent complete.')
    
    print('Generating the synthetic dataset...')
    dataset, _, _ = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials=n_trials,
        n_sessions=n_sessions,
        sample_parameters=parameter_variance!=0,
        sequence_length=sequence_length,
        device=device)
    
    if train_test_ratio == 0:
      dataset_test, experiment_list, _ = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials=256,
        n_sessions=n_sessions,
        sample_parameters=parameter_variance!=0,
        device=device)

    print('Generation of dataset complete.')
  else:
    dataset, experiment_list, df, _ = convert_dataset.convert_dataset(data, sequence_length=sequence_length)
    dataset_test = rnn_utils.DatasetRNN(dataset.xs, dataset.ys)
    
    # check if groundtruth parameters in data - only applicable to generated data with e.g. utils/create_dataset.py
    if 'mean_beta_reward' in df.columns:
      # get parameters from dataset
      agent = bandits.AgentQ(
        beta_reward = df['beta_reward'].values[(df['session']==session_id).values][0],
        alpha_reward = df['alpha_reward'].values[(df['session']==session_id).values][0],
        alpha_penalty = df['alpha_penalty'].values[(df['session']==session_id).values][0],
        confirmation_bias = df['confirmation_bias'].values[(df['session']==session_id).values][0],
        forget_rate = df['forget_rate'].values[(df['session']==session_id).values][0],
        beta_choice = df['beta_choice'].values[(df['session']==session_id).values][0],
        alpha_choice = df['alpha_choice'].values[(df['session']==session_id).values][0],
      )
      
  n_participants = len(experiment_list)
  
  if train_test_ratio > 0:
    # setup of training and test dataset
    index_train = int(train_test_ratio * dataset.xs.shape[1])
    
    xs_test, ys_test = dataset.xs[:, index_train:], dataset.ys[:, index_train:]
    xs_train, ys_train = dataset.xs[:, :index_train], dataset.ys[:, :index_train]
    dataset_train = bandits.DatasetRNN(xs_train, ys_train, sequence_length=sequence_length)
    if dataset_test is None:
      dataset_test = bandits.DatasetRNN(xs_test, ys_test)  
  else:
    if dataset_test is None:
      dataset_test = dataset
    dataset_train = bandits.DatasetRNN(dataset.xs, dataset.ys, sequence_length=sequence_length)
    
  experiment_test = experiment_list[session_id][-dataset_test.xs.shape[1]:]

  if data is None and model is None:
    params_path = rnn_utils.parameter_file_naming(
      'params/params', 
      alpha_reward=alpha_reward, 
      beta_reward=beta_reward, 
      alpha_counterfactual=alpha_counterfactual,
      forget_rate=forget_rate, 
      beta_choice=beta_choice,
      alpha_choice=alpha_choice,
      alpha_penalty=alpha_penalty,
      confirmation_bias=confirmation_bias, 
      variance=parameter_variance, 
      verbose=True,
      )
  elif data is not None and model is None:
    params_path = 'params/params_' + data.split('/')[-1].replace('.csv', '.pkl')
  else:
    params_path = '' + model

  # define model
  model = rnn.RLRNN(
      n_actions=n_actions, 
      hidden_size=hidden_size, 
      device=device,
      list_sindy_signals=sindy_feature_list,
      dropout=dropout,
      n_participants=n_participants if participant_emb else 0,
      ).to(device)

  optimizer_rnn = torch.optim.Adam(model.parameters(), lr=learning_rate)

  print('Setup of the RNN model complete.')

  if checkpoint:
      model, optimizer_rnn = rnn_utils.load_checkpoint(params_path, model, optimizer_rnn)
      print('Loaded model parameters.')

  loss_test = None
  if epochs > 0:
    start_time = time.time()
    
    #Fit the RNN
    print('Training the RNN...')
    model, optimizer_rnn, _ = rnn_training.fit_model(
        model=model,
        dataset_train=dataset_train,
        optimizer=optimizer_rnn,
        convergence_threshold=convergence_threshold,
        epochs=epochs,
        batch_size=batch_size,
        bagging=bagging,
        n_steps=n_steps,
    )
    
    # save trained parameters
    state_dict = {'model': model.state_dict(), 'optimizer': optimizer_rnn.state_dict()}
    
    print('Training finished.')
    torch.save(state_dict, params_path)
    print(f'Saved RNN parameters to file {params_path}.')
    print(f'Training took {time.time() - start_time:.2f} seconds.')
  else:
    if isinstance(model, list):
      model = model[0]
      optimizer_rnn = optimizer_rnn[0]
  
  # validate model
  if dataset_test is not None:
    print('\nTesting the trained RNN on the test dataset...')
    model.eval()
    with torch.no_grad():
      _, _, loss_test = rnn_training.fit_model(
          model=model,
          dataset_train=dataset_train,
      )
  
  # -----------------------------------------------------------
  # Analysis
  # -----------------------------------------------------------
  
  if analysis:
    # print(f'Betas of model: {(model._beta_reward.item(), model._beta_choice.item())}')
    # Synthesize a dataset using the fitted network
    model.set_device(torch.device('cpu'))
    model.to(torch.device('cpu'))
    agent_rnn = bandits.AgentNetwork(model, n_actions=n_actions, deterministic=True)
    
    # get analysis plot
    if agent is not None:
      agents = {'groundtruth': agent, 'rnn': agent_rnn}
    else:
      agents = {'rnn': agent_rnn}

    fig, axs = plotting.session(agents, experiment_test)
    
    title_ground_truth = ''
    if agent is not None: 
      title_ground_truth += r'GT: $\beta_{reward}=$'+str(np.round(agent._beta_reward, 2)) + r'; $\beta_{choice}=$'+str(np.round(agent._beta_choice, 2))
    title_rnn = r'RNN: $\beta_{reward}=$'+str(np.round(agent_rnn._beta_reward, 2)) + r'; $\beta_{choice}=$'+str(np.round(agent_rnn._beta_choice, 2))
    fig.suptitle(title_ground_truth + '\n' + title_rnn)
    plt.show()
    
  return model, loss_test


if __name__=='__main__':
  
  parser = argparse.ArgumentParser(description='Trains the RNN on behavioral data to uncover the underlying Q-Values via different cognitive mechanisms.')
  
  # Training parameters
  parser.add_argument('--checkpoint', action='store_true', help='Whether to load a checkpoint')
  parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
  parser.add_argument('--data', type=str, default=None, help='Path to dataset')
  parser.add_argument('--n_actions', type=int, default=2, help='Number of possible actions')
  parser.add_argument('--epochs', type=int, default=1024, help='Number of epochs for training')
  parser.add_argument('--n_steps', type=int, default=16, help='Number of steps per call')
  parser.add_argument('--bagging', action='store_true', help='Whether to use bagging')
  parser.add_argument('--batch_size', type=int, default=-1, help='Batch size')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the RNN')
  parser.add_argument('--sequence_length', type=int, default=-1, help='Length of training sequences')

  # RNN parameters
  parser.add_argument('--hidden_size', type=int, default=8, help='Hidden size of the RNN')
  parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
  parser.add_argument('--embedding', action='store_true', help='Whether to use participant embedding')

  # Ground truth parameters
  parser.add_argument('--n_trials', type=int, default=200, help='Number of trials per session')
  parser.add_argument('--n_sessions', type=int, default=512, help='Number of sessions')
  parser.add_argument('--alpha', type=float, default=0.25, help='Alpha parameter for the Q-learning update rule')
  parser.add_argument('--beta', type=float, default=3, help='Beta parameter for the Q-learning update rule')
  parser.add_argument('--forget_rate', type=float, default=0., help='Forget rate')
  parser.add_argument('--perseverance_bias', type=float, default=0., help='perseverance bias')
  parser.add_argument('--alpha_p', type=float, default=-1., help='Learning rate for negative outcomes; if -1: same as alpha')
  parser.add_argument('--confirmation_bias', type=float, default=0., help='Whether to include confirmation bias')

  # Environment parameters
  parser.add_argument('--sigma', type=float, default=0.2, help='Drift rate of the reward probabilities')
  parser.add_argument('--non_binary_reward', action='store_true', help='Whether to use non-binary rewards')

  # Analysis parameters
  parser.add_argument('--analysis', action='store_true', help='Whether to perform analysis')

  args = parser.parse_args()  
  
  main(
    # train = True, 
    checkpoint = args.checkpoint,
    model = args.model,
    data = args.data,
    n_actions=args.n_actions,

    # training parameters
    epochs=args.epochs,
    n_trials = args.n_trials,
    n_sessions = args.n_sessions,
    n_steps = args.n_steps,
    bagging = args.bagging,
    batch_size=args.batch_size,
    learning_rate=args.lr,

    # rnn parameters
    hidden_size = args.hidden_size,
    dropout = args.dropout,
    participant_emb=args.embedding,
    
    # ground truth parameters
    alpha_reward = args.alpha,
    beta_reward = args.beta,
    forget_rate = args.forget_rate,
    beta_choice = args.perseverance_bias,
    alpha_penalty = args.alpha_p,
    confirmation_bias = args.confirmation_bias,

    # environment parameters
    sigma = args.sigma,
    
    analysis = args.analysis,
  )
  
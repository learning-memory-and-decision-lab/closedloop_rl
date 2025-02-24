import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Iterable
import pysindy as ps
import numpy as np


class BaseRNN(nn.Module):
    def __init__(
        self, 
        n_actions, 
        hidden_size, 
        device=torch.device('cpu'),
        list_sindy_signals=['x_V', 'c_a', 'c_r'],
        ):
        super(BaseRNN, self).__init__()
        
        # define general network parameters
        self.device = device
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self.emb_size = 0
        self._init_values = tuple([0.5])
    
        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c' 
        self.recording = {key: [] for key in list_sindy_signals}
        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()
        self.submodules_sindy = dict()
        
        self._state = self.set_initial_state()
        
    def forward(self, inputs, prev_state, batch_first=False):
        raise NotImplementedError('This method is not implemented.')
    
    def set_initial_state(self, batch_size=1):
        """this method initializes the hidden state
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.

        Returns:
            Tuple[torch.Tensor]: initial hidden state
        """
                
        for key in self.recording.keys():
            self.recording[key] = []
        
        # state dimensions: (habit_state, value_state, habit, value)
        # dimensions of states: (batch_size, substate, hidden_size)
        self.set_state(*[init_value + torch.zeros([batch_size, 1, self._n_actions], dtype=torch.float, device=self.device) for init_value in self._init_values])
        
        return self.get_state()
        
    def set_state(self, *args):
        """this method sets the latent variables
        
        Args:
            state (Tuple[torch.Tensor]): hidden state
        """
        
        # self._state = dict(hidden_habit=habit_state, hidden_value=value_state, habit=habit, value=value)
        self._state = args
      
    def get_state(self, detach=False):
        """this method returns the hidden state
        
        Returns:
            Tuple[torch.Tensor]: tuple of habit state, value state, habit, value
        """
        
        state = self._state
        if detach:
            state = [s.detach() for s in state]

        return state
    
    def set_device(self, device: torch.device): 
        self.device = device
        
    def record_signal(self, key, old_value, new_value: Optional[torch.Tensor] = None):
        """appends a new timestep sample to the recording. A timestep sample consists of the value at timestep t-1 and the value at timestep t

        Args:
            key (str): recording key to which append the sample to
            old_value (_type_): value at timestep t-1 of shape (batch_size, feature_dim)
            new_value (_type_): value at timestep t of shape (batch_size, feature_dim)
        """
        
        if new_value is None:
            new_value = torch.zeros_like(old_value) - 1
        
        old_value = old_value.view(-1, 1, old_value.shape[-1]).clone().detach().cpu().numpy()
        new_value = new_value.view(-1, 1, new_value.shape[-1]).clone().detach().cpu().numpy()
        sample = np.concatenate([old_value, new_value], axis=1)
        self.recording[key].append(sample)
        
    def get_recording(self, key):
        return self.recording[key]
    
    def call_module(self, key: str, value: torch.Tensor, inputs=None, participant_embedding=None, participant_index=None):
        
        record_signal = False
        
        if inputs is None:
            inputs = torch.zeros((*value.shape[:-1], 0), dtype=torch.float32, device=value.device)
        
        if key in self.submodules_sindy.keys():                
            # sindy module
            
            # convert to numpy
            value = value.detach().cpu().numpy()
            inputs = inputs.detach().cpu().numpy()
            
            if inputs.shape[-1] == 0:
                # create dummy control inputs
                inputs = torch.zeros((*inputs.shape[:-1], 1))
            
            new_value = np.zeros_like(value)
            for index_batch in range(value.shape[0]):
                new_value[index_batch] = np.concatenate(
                    [self.submodules_sindy[key][participant_index[index_batch].item()].predict(value[index_batch, index_action], inputs[index_batch, index_action]) for index_action in range(self._n_actions)], 
                    axis=0,
                    )
            new_value = torch.tensor(new_value, dtype=torch.float32, device=self.device)

        elif key in self.submodules_rnn.keys():
            # rnn module
            record_signal = True if not self.training else False
            inputs = torch.concat((value, inputs, participant_embedding), dim=-1)
            new_value = self.submodules_rnn[key](inputs)
                
        elif key in self.submodules_eq.keys():
            # hard-coded equation
            new_value = self.submodules_eq[key](value, inputs)

        else:
            raise ValueError(f'Invalid module key {key}.')

        if record_signal:
            # record sample for SINDy training 
            self.record_signal(key, value.view(-1, self._n_actions), new_value.view(-1, self._n_actions))
        
        return new_value
    
    def setup_module(self, input_size, hidden_size, dropout):
        module = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            # nn.Dropout(dropout),
            )
        
        return module
    
    def integrate_sindy(self, modules: Dict[str, Iterable[ps.SINDy]]):
        # check that all provided modules find a place in the RNN
        checked = 0
        for m in modules:
            if m in self.submodules_rnn.keys():
                checked += 1
        assert checked == len(modules), f'Not all provided SINDy modules {tuple(modules.keys())} found a corresponding RNN module or vice versa.\nSINDy integration aborted.'
        
        # replace rnn modules with sindy modules
        self.submodules_sindy = modules


class RLRNN(BaseRNN):
    def __init__(
        self,
        n_actions:int, 
        hidden_size:int,
        n_participants:int=0,
        list_sindy_signals=['x_V', 'x_V_nc', 'x_C', 'x_C_nc', 'c_a', 'c_r'],
        dropout=0.,
        device=torch.device('cpu'),
        ):
        
        super(RLRNN, self).__init__(n_actions, hidden_size, device, list_sindy_signals)

        # define additional network parameters
        self._prev_action = torch.zeros(self._n_actions)
        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()
        self._n_participants = n_participants
        self._init_values = (0.5, 0., 0.)  # (value_reward, learning_rate, value_choice)
        
        # participant-embedding layer
        if n_participants > 0:
            self.emb_size = 8
            # self.participant_embedding = nn.Sequential(nn.Embedding(n_participants, self.emb_size), nn.Tanh())
            self.participant_embedding = nn.Embedding(n_participants, self.emb_size)
            self._beta_reward = nn.Sequential(nn.Linear(self.emb_size, 1), nn.ReLU())
            self._beta_choice = nn.Sequential(nn.Linear(self.emb_size, 1), nn.ReLU())
        else:
            self._beta_reward = nn.Parameter(torch.tensor(1.))
            self._beta_choice = nn.Parameter(torch.tensor(1.))
        
        # action-based subnetwork
        self.submodules_rnn['x_C'] = self.setup_module(1+self.emb_size, hidden_size, dropout)
        
        # action-based subnetwork for non-repeated action
        self.submodules_rnn['x_C_nc'] = self.setup_module(1+self.emb_size, hidden_size, dropout)

        # reward-based learning-rate subnetwork for chosen action (and counterfactual if applies)
        self.submodules_rnn['x_V_LR'] = self.setup_module(3+self.emb_size, hidden_size, dropout)
        
        # reward-based equation for chosen action
        self.submodules_eq['x_V'] = lambda value, inputs: (inputs @ torch.tensor([[0], [1]], dtype=torch.float32, device=inputs.device))*(inputs @ torch.tensor([[1], [0]], dtype=torch.float32, device=inputs.device) - value)
        
        # reward-based subnetwork for not-chosen action
        self.submodules_rnn['x_V_nc'] = self.setup_module(2+self.emb_size, hidden_size, dropout)
        
        self._state = self.set_initial_state()
        
    def value_network(self, action: torch.Tensor, reward: torch.Tensor, value: torch.Tensor, learning_rate: torch.Tensor, participant_embedding: torch.Tensor, participant_index: torch.Tensor):
        """this method computes the reward-blind and reward-based updates for the Q-Values without considering the habit (e.g. last chosen action)
        
        Args:
            state (torch.Tensor): last hidden state
            value (torch.Tensor): last Q-Values
            action (torch.Tensor): chosen action (one-hot encoded)
            reward (torch.Tensor): received reward

        Returns:
            torch.Tensor: updated Q-Values
        """
        
        # add dimension to inputs
        action = action.unsqueeze(-1)
        reward = reward.unsqueeze(-1)
        value = value.unsqueeze(-1)
        learning_rate = learning_rate.unsqueeze(-1)
        
        # learning rate computation
        inputs = torch.concat([value, reward], dim=-1)
        learning_rate = self.call_module('x_V_LR', learning_rate, inputs, participant_embedding, participant_index)
        learning_rate = self._sigmoid(learning_rate)
        
        # compute update for chosen action (hard-coded equation)
        inputs = torch.concat([reward, learning_rate], dim=-1)
        update_chosen = self.call_module('x_V', value, inputs)
        
        # reward sub-network for non-chosen action
        inputs = reward
        update_not_chosen = self.call_module('x_V_nc', value, inputs, participant_embedding, participant_index)
        
        # apply update_chosen only for chosen option and update_not_chosen for not-chosen option
        next_value = value + update_chosen * action + update_not_chosen * (1-action)
        
        return next_value.squeeze(-1), learning_rate.squeeze(-1)
    
    def choice_network(self, action: torch.Tensor, value: torch.Tensor, participant_embedding: torch.Tensor, participant_index: torch.Tensor):
        
        # add dimension to inputs
        action = action.unsqueeze(-1)
        value = value.unsqueeze(-1)
                
        # choice sub-network for chosen action
        inputs = None#action
        update_chosen = self.call_module('x_C', value, inputs, participant_embedding, participant_index)
        
        # choice sub-network for non-chosen action
        inputs=None
        update_not_chosen = self.call_module('x_C_nc', value, inputs, participant_embedding, participant_index)
        
        next_value = value + update_chosen * action + update_not_chosen * (1-action)
        next_value = self._sigmoid(next_value)
        
        return next_value.squeeze(-1)
    
    def forward(self, inputs: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor]] = None, batch_first=False):
        """this method computes the next hidden state and the updated Q-Values based on the input and the previous hidden state
        
        Args:
            inputs (torch.Tensor): input tensor of form (seq_len, batch_size, n_actions + 1) or (batch_size, seq_len, n_actions + 1) if batch_first
            prev_state (Tuple[torch.Tensor]): tuple of previous state of form (habit state, value state, habit, value)

        Returns:
            torch.Tensor: updated Q-Values
            Tuple[torch.Tensor]: updated habit state, value state, habit, value
        """
        
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        action_array = inputs[:, :, :self._n_actions].float()
        reward_array = inputs[:, :, self._n_actions:2*self._n_actions].float()
        participant_id_array = inputs[:, :, -1:].int()
        
        if prev_state is not None:
            self.set_state(*prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        # get previous model state
        state = [s.squeeze(1) for s in self.get_state()]  # remove model dim for forward pass -> only one model
        value_reward, learning_rate, value_choice = state
        
        # get participant embedding
        if self._n_participants > 0:
            participant_embedding_array = self.participant_embedding(participant_id_array).repeat(1, 1, self._n_actions, 1)
            beta_reward = self._beta_reward(participant_embedding_array[0, :, 0])
            beta_choice = self._beta_choice(participant_embedding_array[0, :, 0])
        else:
            participant_embedding_array = torch.zeros((*inputs.shape[:-1], 2, 0), device=self.device)
            beta_reward = self._beta_reward
            beta_choice = self._beta_choice
        
        timestep_array = torch.arange(inputs.shape[0])
        logits_array = torch.zeros_like(action_array, device=self.device)
        for timestep, action, reward, participant_embedding, participant_id in zip(timestep_array, action_array, reward_array, participant_embedding_array, participant_id_array):         
            # record control signals for SINDy-training if model is not in training mode
            if not self.training:
                self.record_signal('c_a', action)
                self.record_signal('c_r', reward)
                self.record_signal('c_V', value_reward)
            
            # compute the updates
            value_reward, learning_rate = self.value_network(action, reward, value_reward, learning_rate, participant_embedding, participant_id)
            value_choice = self.choice_network(action, value_choice, participant_embedding, participant_id)
            
            logits_array[timestep] += value_reward * beta_reward + value_choice * beta_choice

        # add model dim again and set state
        self.set_state(value_reward.unsqueeze(1), learning_rate.unsqueeze(1), value_choice.unsqueeze(1))
        
        if batch_first:
            logits_array = logits_array.permute(1, 0, 2)
            
        return logits_array, self.get_state()


class MinMaxNormalize(nn.Module):
    def __init__(self, epsilon=1e-6, minmax_threshold=1e-2, momentum=0.9):
        super(MinMaxNormalize, self).__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.minmax_threshold = minmax_threshold
        
        # Parameters to track the global min and max values
        self.register_buffer("global_min", torch.tensor(float("inf")))
        self.register_buffer("global_max", -torch.tensor(float("inf")))

    def forward(self, x):
        if self.training:
            # Compute batch min and max
            batch_min = x.min()
            batch_max = x.max()

            # Update global min and max using exponential moving average
            self.global_min = (
                self.momentum * self.global_min + (1 - self.momentum) * batch_min
            )
            self.global_max = (
                self.momentum * self.global_max + (1 - self.momentum) * batch_max
            )

            # Use the batch min/max for normalization during training
            min_vals, max_vals = batch_min, batch_max
        else:
            # Use global min and max during evaluation
            min_vals, max_vals = self.global_min, self.global_max

        # Normalize using the current min and max
        if x.max() - x.min() > self.minmax_threshold:
            return (x - min_vals) / (max_vals - min_vals + self.epsilon)
        else:
            return x
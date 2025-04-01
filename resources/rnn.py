import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Optional, Tuple, Dict, Iterable, Callable, Union, List
import pysindy as ps
import numpy as np


class BaseRNN(nn.Module):
    def __init__(
        self, 
        n_actions, 
        hidden_size: int = 8,
        device=torch.device('cpu'),
        list_signals=['x_V', 'c_a', 'c_r'],
        ):
        super(BaseRNN, self).__init__()
        
        # define general network parameters
        self.device = device
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self.embedding_size = 0
        self._n_participants = 0
            
        # session recording; used for sindy training; training variables start with 'x' and control parameters with 'c' 
        self.recording = {key: [] for key in list_signals}
        self.submodules_rnn = nn.ModuleDict()
        self.submodules_eq = dict()
        self.submodules_sindy = dict()
        
        self.state = self.set_initial_state()
        
    def forward(self, inputs, prev_state, batch_first=False):
        raise NotImplementedError('This method is not implemented.')
    
    def init_forward_pass(self, inputs, prev_state, batch_first):
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        actions = inputs[:, :, :self._n_actions].float()
        rewards = inputs[:, :, self._n_actions:2*self._n_actions].float()
        participant_ids = inputs[0, :, -1:].int()
        
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        timesteps = torch.arange(actions.shape[0])
        logits = torch.zeros_like(actions)
        
        return (actions, rewards, participant_ids), logits, timesteps
    
    def post_forward_pass(self, logits, batch_first):
        # add model dim again and set state
        # self.set_state(*args)
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
            
        return logits
    
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
        # self.set_state(*[init_value + torch.zeros([batch_size, self._n_actions], dtype=torch.float, device=self.device) for init_value in self.init_values])
        
        state = {key: torch.full(size=[batch_size, self._n_actions], fill_value=self.init_values[key], dtype=torch.float32, device=self.device) for key in self.init_values}
        
        self.set_state(state)
        return self.get_state()
        
    def set_state(self, state_dict):
        """this method sets the latent variables
        
        Args:
            state (Dict[str, torch.Tensor]): hidden state
        """
        
        # self._state = dict(hidden_habit=habit_state, hidden_value=value_state, habit=habit, value=value)
        self.state = state_dict
      
    def get_state(self, detach=False):
        """this method returns the memory state
        
        Returns:
            Dict[str, torch.Tensor]: Dict of latent variables corresponding to the memory state
        """
        
        state = self.state
        if detach:
            state = {key: state[key].detach() for key in state}

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
    
    def setup_module(self, input_size: int, hidden_size: int = None, dropout: float = 0., activation: nn.Module = None):
        """This method creates the standard RNN-module used in computational discovery of cognitive dynamics

        Args:
            input_size (_type_): The number of inputs (excluding the memory state)
            hidden_size (_type_): Hidden size after the input layer
            dropout (_type_): Dropout rate before output layer
            activation (nn.Module, optional): Possibility to include an activation function. Defaults to None.

        Returns:
            torch.nn.Sequential: A sequential module which can be called by one line
        """
        if hidden_size is None:
            hidden_size = self._hidden_size
            
        # Linear network
        layers = [
            nn.Linear(input_size+1, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        ]
        
        # GRU network
        # layers = [
        #     nn.Linear(input_size, hidden_size),
        #     nn.Dropout(dropout),
        #     nn.GRU(hidden_size, 1),
        #     nn.Linear(1, 1),
        # ]
        
        if activation is not None:
            layers.append(activation())
        
        return nn.Sequential(*layers)
    
    def call_module(
        self,
        key_module: str,
        key_state: str,
        action: torch.Tensor = None,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        participant_embedding: torch.Tensor = None, 
        participant_index: torch.Tensor = None,
        activation_rnn: Callable = None,
        scaling: bool = False,
        ):
        """Used to call a submodule of the RNN. Can be either: 
            1. RNN-module (saved in 'self.submodules_rnn')
            2. SINDy-module (saved in 'self.submodules_sindy')
            3. hard-coded equation (saved in 'self.submodules_eq')

        Args:
            key_module (str): _description_
            key_state (str): _description_
            action (torch.Tensor, optional): _description_. Defaults to None.
            inputs (Union[torch.Tensor, Tuple[torch.Tensor]], optional): _description_. Defaults to None.
            participant_embedding (torch.Tensor, optional): _description_. Defaults to None.
            participant_index (torch.Tensor, optional): _description_. Defaults to None.
            activation_rnn (Callable, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        record_signal = False
        
        action = action.unsqueeze(-1)
        value = self.get_state()[key_state].unsqueeze(-1)
        # value = key_state.unsqueeze(-1)
        
        if inputs is None:
            inputs = torch.zeros((*value.shape[:-1], 0), dtype=torch.float32, device=value.device)
            
        if participant_embedding is None:
            participant_embedding = torch.zeros((*value.shape[:-1], 0), dtype=torch.float32, device=value.device)
        elif participant_embedding.ndim == 2:
            participant_embedding = participant_embedding.unsqueeze(1).repeat(1, value.shape[1], 1)
        
        if isinstance(inputs, tuple):
            inputs = torch.concat([inputs_i.unsqueeze(-1) for inputs_i in inputs], dim=-1)
        elif inputs.dim()==2:
            inputs = inputs.unsqueeze(-1)
        
        if key_module in self.submodules_sindy.keys():                
            # sindy module
            
            # convert to numpy
            value = value.detach().cpu().numpy()
            inputs = inputs.detach().cpu().numpy()
            
            if inputs.shape[-1] == 0:
                # create dummy control inputs
                inputs = torch.zeros((*inputs.shape[:-1], 1))
            
            if participant_index is None:
                participant_index = torch.zeros((1, 1), dtype=torch.int32)
            
            next_value = np.zeros_like(value)
            for index_batch in range(value.shape[0]):
                sindy_model = self.submodules_sindy[key_module][participant_index[index_batch].item()] if isinstance(self.submodules_sindy[key_module], dict) else self.submodules_sindy[key_module]
                next_value[index_batch] = np.concatenate(
                    [sindy_model.predict(value[index_batch, index_action], inputs[index_batch, index_action]) for index_action in range(self._n_actions)], 
                    axis=0,
                    )
            next_value = torch.tensor(next_value, dtype=torch.float32, device=self.device)

        elif key_module in self.submodules_rnn.keys():
            # rnn module
            record_signal = True if not self.training else False
            
            # Linear handling
            inputs = torch.concat((value, inputs, participant_embedding), dim=-1)
            update_value = self.submodules_rnn[key_module](inputs)
            
            # GRU handling
            # if inputs.shape[-1] == 0:
            #     inputs = torch.zeros((*value.shape[:-1], 1), dtype=torch.float32, device=value.device)
            # inputs = torch.concat((inputs, participant_embedding), dim=-1)
            # update_value = torch.zeros_like(value)
            # for index_action in self._n_actions:
            #     current_action = torch.eye(self._n_actions)[index_action].view(1, self._n_actions, 1)
            #     value_i = value.squeeze(-1) @ current_action
            #     hidden_representation = self.submodules_rnn[key_module][:2](inputs[:, index_action][:, None].swapaxes(0, 1))
            #     _, update_value_i = self.submodules_rnn[key_module][2](hidden_representation, value)
            #     update_value_i = self.submodules_rnn[key_module][3](update_value)
            #     update_value = torch.eye(self._n_actions)[index_action].view(1, self._n_actions, 1) * update_value_i
            
            next_value = value + update_value
            if activation_rnn is not None:
                next_value = activation_rnn(next_value)
            
        elif key_module in self.submodules_eq.keys():
            # hard-coded equation
            next_value = self.submodules_eq[key_module](value.squeeze(-1), inputs).unsqueeze(-1)

        else:
            raise ValueError(f'Invalid module key {key_module}.')

        if action is not None:
            # keep only actions necessary for that update and set others to zero
            next_value = next_value * action
        
        # clip next_value to a specific range
        next_value = torch.clip(input=next_value, min=-1e1, max=1e1)
        
        if record_signal:
            # record sample for SINDy training 
            self.record_signal(key_module, value.view(-1, self._n_actions), next_value.view(-1, self._n_actions))
        
        if scaling:
            # scale by inverse noise temperature
            scaling_factor = self.betas[key_state] if isinstance(self.betas[key_state], nn.Parameter) else self.betas[key_state](participant_embedding)
            next_value = next_value * scaling_factor
        
        return next_value.squeeze(-1)
    
    def integrate_sindy(self, modules: Dict[str, Iterable[ps.SINDy]]):
        # check that all provided modules find a place in the RNN
        checked = 0
        for m in modules:
            if m in self.submodules_rnn.keys():
                checked += 1
        assert checked == len(modules), f'Not all provided SINDy modules {tuple(modules.keys())} found a corresponding RNN module or vice versa.\nSINDy integration aborted.'
        
        # replace rnn modules with sindy modules
        self.submodules_sindy = modules


# class RLRNN(BaseRNN):
#     def __init__(
#         self,
#         n_actions:int, 
#         hidden_size:int,
#         n_participants:int=0,
#         list_signals=['x_V', 'x_V_nc', 'x_C', 'x_C_nc', 'c_a', 'c_r'],
#         dropout=0.,
#         device=torch.device('cpu'),
#         ):
        
#         super(RLRNN, self).__init__(n_actions, hidden_size, device, list_signals)

#         # define additional network parameters
#         self._prev_action = torch.zeros(self._n_actions)
#         self._relu = nn.ReLU()
#         self._sigmoid = nn.Sigmoid()
#         self._n_participants = n_participants
#         self.init_values = (0.5, 0., 0.)  # (value_reward, learning_rate, value_choice)
        
#         # participant-embedding layer
#         if n_participants > 0:
#             self.emb_size = 8
#             # self.participant_embedding = nn.Sequential(nn.Embedding(n_participants, self.emb_size), nn.Tanh())
#             self.participant_embedding = nn.Embedding(n_participants, self.emb_size)
#             self._beta_reward = nn.Sequential(nn.Linear(self.emb_size, 1), nn.ReLU())
#             self._beta_choice = nn.Sequential(nn.Linear(self.emb_size, 1), nn.ReLU())
#         else:
#             self._beta_reward = nn.Parameter(torch.tensor(1.))
#             self._beta_choice = nn.Parameter(torch.tensor(1.))
        
#         # action-based subnetwork
#         self.submodules_rnn['x_C'] = self.setup_module(1+self.emb_size, hidden_size, dropout)
        
#         # action-based subnetwork for non-repeated action
#         self.submodules_rnn['x_C_nc'] = self.setup_module(1+self.emb_size, hidden_size, dropout)

#         # reward-based learning-rate subnetwork for chosen action (and counterfactual if applies)
#         self.submodules_rnn['x_V_LR'] = self.setup_module(3+self.emb_size, hidden_size, dropout)
        
#         # reward-based equation for chosen action
#         # self.submodules_eq['x_V'] = lambda value, inputs: (inputs @ torch.tensor([[0], [1]], dtype=torch.float32, device=inputs.device))*(inputs @ torch.tensor([[1], [0]], dtype=torch.float32, device=inputs.device) - value)
#         self.submodules_eq['x_V'] = lambda value, inputs: value + (inputs[..., 1] * (inputs[..., 0] - value[..., 0])).view(-1, 2, 1)
        
#         # reward-based subnetwork for not-chosen action
#         self.submodules_rnn['x_V_nc'] = self.setup_module(2+self.emb_size, hidden_size, dropout)
        
#         self._state = self.set_initial_state()
        
#     def value_network(self, action: torch.Tensor, reward: torch.Tensor, value: torch.Tensor, learning_rate: torch.Tensor, participant_embedding: torch.Tensor, participant_index: torch.Tensor):
#         """this method computes the reward-blind and reward-based updates for the Q-Values without considering the habit (e.g. last chosen action)
        
#         Args:
#             state (torch.Tensor): last hidden state
#             value (torch.Tensor): last Q-Values
#             action (torch.Tensor): chosen action (one-hot encoded)
#             reward (torch.Tensor): received reward

#         Returns:
#             torch.Tensor: updated Q-Values
#         """
        
#         # add dimension to inputs
#         action = action.unsqueeze(-1)
#         reward = reward.unsqueeze(-1)
#         value = value.unsqueeze(-1)
#         learning_rate = learning_rate.unsqueeze(-1)
        
#         # learning rate computation
#         next_learning_rate = self.call_module(
#             key_module='x_V_LR', 
#             action=action, 
#             key_state=learning_rate, 
#             inputs = torch.concat([value, reward], dim=-1),
#             participant_embedding=participant_embedding, 
#             participant_index=participant_index, 
#             activation_rnn=f.sigmoid,
#             )
        
#         # compute update for chosen action (hard-coded equation)
#         next_value_chosen = self.call_module(
#             key_module='x_V', 
#             action=action, 
#             key_state=value, 
#             inputs=torch.concat([reward, next_learning_rate], dim=-1),
#             )
        
#         # reward sub-network for non-chosen action
#         next_value_not_chosen = self.call_module(
#             'x_V_nc', 
#             action=1-action,
#             key_state=value, 
#             inputs=reward, 
#             participant_embedding=participant_embedding, 
#             participant_index=participant_index,
#             )
        
#         # get together all updated values
#         next_value = next_value_chosen + next_value_not_chosen
        
#         return next_value.squeeze(-1), next_learning_rate.squeeze(-1)
    
#     def choice_network(self, action: torch.Tensor, value: torch.Tensor, participant_embedding: torch.Tensor, participant_index: torch.Tensor):
        
#         # add dimension to inputs
#         action = action.unsqueeze(-1)
#         value = value.unsqueeze(-1)
        
#         # choice sub-network for chosen action
#         next_value_chosen = self.call_module(
#             'x_C', 
#             action=action,
#             key_state=value, 
#             inputs=None, 
#             participant_embedding=participant_embedding, 
#             participant_index=participant_index,
#             activation_rnn=f.sigmoid,
#             )
        
#         # choice sub-network for non-chosen action
#         next_value_not_chosen = self.call_module(
#             'x_C_nc', 
#             action=1-action,
#             key_state=value, 
#             inputs=None, 
#             participant_embedding=participant_embedding, 
#             participant_index=participant_index,
#             activation_rnn=f.sigmoid,
#             )
        
#         # get together all updated values
#         next_value = next_value_chosen + next_value_not_chosen
        
#         return next_value.squeeze(-1)
    
#     def forward(self, inputs: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor]] = None, batch_first=False):
#         """this method computes the next hidden state and the updated Q-Values based on the input and the previous hidden state
        
#         Args:
#             inputs (torch.Tensor): input tensor of form (seq_len, batch_size, n_actions + 1) or (batch_size, seq_len, n_actions + 1) if batch_first
#             prev_state (Tuple[torch.Tensor]): tuple of previous state of form (habit state, value state, habit, value)

#         Returns:
#             torch.Tensor: updated Q-Values
#             Tuple[torch.Tensor]: updated habit state, value state, habit, value
#         """
        
#         if batch_first:
#             inputs = inputs.permute(1, 0, 2)
        
#         action_array = inputs[:, :, :self._n_actions].float()
#         reward_array = inputs[:, :, self._n_actions:2*self._n_actions].float()
#         participant_id_array = inputs[:, :, -1:].int()
        
#         if prev_state is not None:
#             self.set_state(*prev_state)
#         else:
#             self.set_initial_state(batch_size=inputs.shape[1])
        
#         # get previous model state
#         state = [s.squeeze(1) for s in self.get_state()]  # remove model dim for forward pass -> only one model
#         value_reward, learning_rate, value_choice = state
        
#         # get participant embedding
#         if self._n_participants > 0:
#             participant_embedding_array = self.participant_embedding(participant_id_array).repeat(1, 1, self._n_actions, 1)
#             beta_reward = self._beta_reward(participant_embedding_array[0, :, 0])
#             beta_choice = self._beta_choice(participant_embedding_array[0, :, 0])
#         else:
#             participant_embedding_array = torch.zeros((*inputs.shape[:-1], 2, 0), device=self.device)
#             beta_reward = self._beta_reward
#             beta_choice = self._beta_choice
        
#         timestep_array = torch.arange(inputs.shape[0])
#         logits_array = torch.zeros_like(action_array, device=self.device)
#         for timestep, action, reward, participant_embedding, participant_id in zip(timestep_array, action_array, reward_array, participant_embedding_array, participant_id_array):                         
            
#             # record memory state and control signals for SINDy-training if model is not in training mode
#             if not self.training:
#                 # self.record_signal('x_V_LR', prev_learning_rate, learning_rate)
#                 # self.record_signal('x_V_nc', prev_value_reward, value_reward)
#                 # self.record_signal('x_C', prev_value_choice, value_choice)
#                 # self.record_signal('x_C_nc', prev_value_choice, value_choice)
#                 self.record_signal('c_a', action)
#                 self.record_signal('c_r', reward)
#                 self.record_signal('c_V', value_reward)
            
#             # save memory state at timestep t
#             # prev_value_reward, prev_learning_rate, prev_value_choice = value_reward, learning_rate, value_choice
            
#             # compute the updates
#             value_reward, learning_rate = self.value_network(action, reward, value_reward, learning_rate, participant_embedding, participant_id)
#             value_choice = self.choice_network(action, value_choice, participant_embedding, participant_id)
            
#             logits_array[timestep] += value_reward * beta_reward + value_choice * beta_choice

#         self.post_forward_pass(logits_array, batch_first, value_reward, learning_rate, value_choice)
        
#         return logits_array, self.get_state()


class RLRNN(BaseRNN):
    
    init_values = {
            'x_value_reward': 0.5,
            'x_value_choice': 0.,
            'x_learning_rate_reward': 0.,
            'x_learning_rate_reward_uncertainty': 0.,# PG MR edits

        }
    
    def __init__(
        self,
        n_actions: int,
        n_participants: int,
        hidden_size = 8,
        dropout = 0.,
        device = torch.device('cpu'),
        list_signals = ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen', 'c_action', 'c_reward', 'c_value_reward', 'x_learning_rate_reward_uncertainty'],# PG MR edits
        **kwargs,
    ):
        
        super(RLRNN, self).__init__(n_actions=n_actions, list_signals=list_signals, hidden_size=hidden_size, device=device)
        
        # set up the participant-embedding layer
        self.embedding_size = 8
        self.participant_embedding = torch.nn.Embedding(num_embeddings=n_participants, embedding_dim=self.embedding_size)
        
        # scaling factor (inverse noise temperature) for each participant for the values which are handled by an hard-coded equation
        self.betas = torch.nn.ModuleDict()
        self.betas['x_value_reward'] = torch.nn.Sequential(torch.nn.Linear(self.embedding_size, 1), torch.nn.ReLU())
        self.betas['x_value_choice'] = torch.nn.Sequential(torch.nn.Linear(self.embedding_size, 1), torch.nn.ReLU())
        
        # set up the submodules
        self.submodules_rnn['x_learning_rate_reward'] = self.setup_module(input_size=2+self.embedding_size, dropout=dropout)
        self.submodules_rnn['x_value_reward_not_chosen'] = self.setup_module(input_size=0+self.embedding_size, dropout=dropout)
        self.submodules_rnn['x_value_choice_chosen'] = self.setup_module(input_size=0+self.embedding_size, dropout=dropout)
        self.submodules_rnn['x_value_choice_not_chosen'] = self.setup_module(input_size=0+self.embedding_size, dropout=dropout)
        self.submodules_rnn['x_learning_rate_reward_uncertainty'] = self.setup_module(input_size=2+self.embedding_size, dropout=dropout) # PG MR edits

        # set up hard-coded equations
        self.submodules_eq['x_value_reward_chosen'] = lambda value, inputs: value + inputs[..., 1] * (inputs[..., 0] - value)
        
    def forward(self, inputs, prev_state=None, batch_first=False):
        """Forward pass of the RNN

        Args:
            inputs (torch.Tensor): includes all necessary inputs (action, reward, participant id) to the RNN to let it compute the next action
            prev_state (Tuple[torch.Tensor], optional): That's the previous memory state of the RNN containing the reward-based value. Defaults to None.
            batch_first (bool, optional): Indicates whether the first dimension of inputs is batch (True) or timesteps (False). Defaults to False.
        """
        
        # First, we have to initialize all the inputs and outputs (i.e. logits)
        inputs, logits, timesteps = self.init_forward_pass(inputs, prev_state, batch_first)
        actions, rewards, participant_id = inputs
        
        # Here we compute now the participant embeddings for each entry in the batch
        participant_embedding = self.participant_embedding(participant_id[:, 0].int())
        # beta_value_reward = self.betas['x_value_reward'](participant_embedding)
        # beta_value_choice = self.betas['x_value_choice'](participant_embedding)
        
        for timestep, action, reward in zip(timesteps, actions, rewards):
            
            # updates for x_value_reward
            learning_rate_reward = self.call_module(
                key_module='x_learning_rate_reward',
                key_state='x_learning_rate_reward',
                action=action,
                inputs=(reward, self.state['x_value_reward']),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
            )
            
            next_value_reward_chosen = self.call_module(
                key_module='x_value_reward_chosen',
                key_state='x_value_reward',
                action=action,
                inputs=(reward, learning_rate_reward),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                scaling=True,
                )
            
            next_value_reward_not_chosen = self.call_module(
                key_module='x_value_reward_not_chosen',
                key_state='x_value_reward',
                action=1-action,
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                )
            
            # updates for x_value_choice
            next_value_choice_chosen = self.call_module(
                key_module='x_value_choice_chosen',
                key_state='x_value_choice',
                action=action,
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                scaling=True,
                )
            
            next_value_choice_not_chosen = self.call_module(
                key_module='x_value_choice_not_chosen',
                key_state='x_value_choice',
                action=1-action,
                inputs=None,
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                scaling=True,
                )

            # PG MR edits #################################
            # expansion with state inference module
            # ???
            learning_rate_uncertainty = self.call_module(
                key_module='x_learning_rate_reward_uncertainty',
                key_state='x_learning_rate_reward_uncertainty',
                action=action,
                inputs=(reward, self.state['x_value_reward']),
                participant_embedding=participant_embedding,
                participant_index=participant_id,
                activation_rnn=torch.nn.functional.sigmoid,
                )

            #next_value_reward_chosen_slider_value 
            #next_value_reward_unchosen_slider_value 
            #next_value_choice_chosen_slider_value
            #next_value_choice_unchosen_slider_value


            
            ###############################################

            
            # updating the memory state
            self.state['x_learning_rate_reward'] = learning_rate_reward
            self.state['x_value_reward'] = next_value_reward_chosen + next_value_reward_not_chosen
            self.state['x_value_choice'] = next_value_choice_chosen + next_value_choice_not_chosen
            self.state['x_learning_rate_reward_uncertainty'] = learning_rate_reward_uncertainty # PG MR edits

            # Now keep track of the logit in the output array
            logits[timestep] = self.state['x_value_reward'] + self.state['x_value_choice']
            
            # record the inputs for training SINDy later on
            self.record_signal('c_action', action)
            self.record_signal('c_reward', reward)
            self.record_signal('c_value_reward', self.state['x_value_reward'])
        
        # post-process the forward pass; give here as inputs the logits, batch_first and all values from the memory state
        logits = self.post_forward_pass(logits, batch_first)
        
        return logits, self.get_state()


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


 ## PG MR edits ###############################################################
    def init_forward_pass(self, inputs, prev_state, batch_first):
        if batch_first:
            inputs = inputs.permute(1, 0, 2) ##????
        #states = ...??
        #correct_actions = ...??
        #contrastdiff = ...
        #slidervalues = ...??
        actions = inputs[:, :, :self._n_actions].float()
        rewards = inputs[:, :, self._n_actions:2*self._n_actions].float()
        participant_ids = inputs[0, :, -1:].int()
        
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        timesteps = torch.arange(actions.shape[0])
        logits = torch.zeros_like(actions)
        slider_value = torch.zeros_like(actions)

        return (actions, rewards, participant_ids), logits, timesteps, slider_value
####################################################################################
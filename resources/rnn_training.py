import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
import numpy as np
from resources.rnn import BaseRNN
from resources.rnn_utils import DatasetRNN


def batch_train(
    model: BaseRNN,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    n_steps: int = -1,
    l1_weight_decay: float = 1e-4,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
    ):

    """
    Trains a model with the given batch.
    """
    
    if n_steps == -1:
        n_steps = xs.shape[1]
        
    model.set_initial_state(batch_size=len(xs))
    state = model.get_state(detach=True)
    
    loss_batch = 0
    iterations = 0
    for t in range(0, xs.shape[1], n_steps):
        n_steps = min(xs.shape[1]-t, n_steps)
        xs_step = xs[:, t:t+n_steps]
        ys_step = ys[:, t:t+n_steps]
        
        # mask = xs_step[:, :, :1] > -1
        # xs_step *= mask
        # ys_step *= mask
        
        state = model.get_state(detach=True)
        y_pred = model(xs_step, state, batch_first=True)[0]
        
        loss_step = loss_fn(
            # (y_pred*mask).reshape(-1, model._n_actions), 
            y_pred.reshape(-1, model._n_actions), 
            torch.argmax(ys_step.reshape(-1, model._n_actions), dim=1),
            )
        
        loss_batch += loss_step
        iterations += 1
                
        if torch.is_grad_enabled():
            
            # L1 weight decay to enforce sparsification in the network
            l1_reg = l1_weight_decay * torch.stack([param.abs().sum() for param in model.parameters()]).sum()

            loss = loss_step + l1_reg
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model, optimizer, loss_batch.item()/iterations
    

def fit_model(
    model: BaseRNN,
    dataset_train: DatasetRNN,
    dataset_test: DatasetRNN = None,
    optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-5,
    epochs: int = 1,
    batch_size: int = -1,
    bagging: bool = False,
    n_steps: int = -1,
    penalty_l1: float = 1e-4,
    verbose: bool = True,
    ):
    
    # initialize dataloader
    if batch_size == -1:
        batch_size = len(dataset_train)
        
    # use random sampling with replacement
    if bagging:
        batch_size = max(batch_size, 64)
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=batch_size) if bagging else None
    else:
        sampler = None
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, shuffle=True if sampler is None else False)
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))
    
    # set up learning rate scheduler
    if optimizer is not None:
        warmup_steps = int(epochs * 0.125)
        # Define the LambdaLR scheduler for warm-up
        def warmup_lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0  # No change after warm-up phase

        # Create the scheduler with the Lambda function
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=warmup_steps, T_mult=2)
        # scheduler_warmup, scheduler = None, None
    else:
        scheduler_warmup, scheduler = None, None
        
    if epochs == 0:
        continue_training = False
        msg = 'No training epochs specified. Model will not be trained.'
        if verbose:
            print(msg)
    else:
        continue_training = True
        converged = False
        n_calls_to_train_model = 0
        convergence_value = 1
        last_loss = 1
        recency_factor = 0.5
    
    loss_train = 0
    loss_test = 0
    iterations_per_epoch = max(len(dataset_train), 64) // batch_size
    
    # start training
    while continue_training:
        try:
            loss_train = 0
            loss_test = 0
            t_start = time.time()
            n_calls_to_train_model += 1
            for _ in range(iterations_per_epoch):
                # get next batch
                xs, ys = next(iter(dataloader_train))
                xs = xs.to(model.device)
                ys = ys.to(model.device)
                # train model
                model, optimizer, loss_i = batch_train(
                    model=model,
                    xs=xs,
                    ys=ys,
                    optimizer=optimizer,
                    n_steps=n_steps,
                )
                loss_train += loss_i
            loss_train /= iterations_per_epoch
            
            if dataset_test is not None:
                model.eval()
                with torch.no_grad():
                    xs, ys = next(iter(dataloader_test))
                    xs = xs.to(model.device)
                    ys = ys.to(model.device)
                    # evaluate model
                    _, _, loss_test = batch_train(
                        model=model,
                        xs=xs,
                        ys=ys,
                        optimizer=optimizer,
                    )
                model.train()
            
            # check for convergence
            dloss = last_loss - loss_test if dataset_test is not None else last_loss - loss_train
            convergence_value += recency_factor * (np.abs(dloss) - convergence_value)
            converged = convergence_value < convergence_threshold
            continue_training = not converged and n_calls_to_train_model < epochs
            last_loss = 0
            last_loss += loss_test if dataset_test is not None else loss_train
            
            msg = None
            if verbose:
                msg = f'Epoch {n_calls_to_train_model}/{epochs} --- L(Training): {loss_train:.7f}'                
                if dataset_test is not None:
                    msg += f'; L(Validation): {loss_test:.7f}'
                msg += f'; Time: {time.time()-t_start:.2f}s; Convergence value: {convergence_value:.2e}'
                if scheduler is not None:
                    msg += f'; LR: {scheduler_warmup.get_last_lr()[-1] if n_calls_to_train_model < warmup_steps else scheduler.get_last_lr()[-1]:.2e}'
                if converged:
                    msg += '\nModel converged!'
                elif n_calls_to_train_model >= epochs:
                    msg += '\nMaximum number of training epochs reached.'
                    if not converged:
                        msg += '\nModel did not converge yet.'
                        
            # if scheduler is not None:
            #     if n_calls_to_train_model <= warmup_steps: 
            #         scheduler_warmup.step()
            #     else:
            #         scheduler.step()
                    
        except KeyboardInterrupt:
            continue_training = False
            msg = 'Training interrupted. Continuing with further operations...'
        if verbose:
            print(msg)
            
    return model, optimizer, loss_train
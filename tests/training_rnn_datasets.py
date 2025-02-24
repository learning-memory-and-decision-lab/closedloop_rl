import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

path_datasets = 'data/study_recovery_stepperserverance/'
path_params = 'params/study_recovery_stepperserverance/'

datasets = os.listdir(path_datasets)

losses = []
for d in datasets:
    dataset = os.path.join(path_datasets, d)
    model = os.path.join(path_params, d.replace('.csv', f'.pkl').replace('data', 'params'))
    
    _, loss = rnn_main.main(
        checkpoint=False,
        epochs=1024,
        
        data=dataset,
        model=model,

        n_actions=2,
        
        dropout=0.25,
        participant_emb=True,
        bagging=True,

        learning_rate=1e-2,
        batch_size=-1,
        sequence_length=-1,
        train_test_ratio=0,
        n_steps=16,
        )

    losses.append(loss)

print(losses)
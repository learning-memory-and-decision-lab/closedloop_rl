import os, sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.setup_agents import setup_agent_sindy


# create a mapping of ground truth parameters to library parameters
mapping_x_V_LR = {
    '1': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (alpha_penalty+confirmation_bias*0.25)*beta_reward,
    
    'x_V_LR': 0,
    
    # 'c_r': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (-alpha_reward-0.5*confirmation_bias)*beta_reward,
    'c_r': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (alpha_reward-0.5*confirmation_bias)*beta_reward,
    
    'c_V': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (-0.5*confirmation_bias)*beta_reward,
    
    'x_V_LR c_V': 0,
    'c_V x_V_LR': 0,
    
    'x_V_LR c_r': 0,
    'c_r x_V_LR': 0,
    
    'x_V_LR^2': 0,
    
    'c_V c_r': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (confirmation_bias)*beta_reward,
    'c_r c_V': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (confirmation_bias)*beta_reward,
    
    'c_V^2': 0,
    
    'c_r^2': 0,
}

mapping_x_V_nc = {
    '1': lambda forget_rate, beta_reward: (0.5*forget_rate)*beta_reward,
    
    'x_V_nc': lambda forget_rate, beta_reward: (1-forget_rate)*beta_reward,
    
    'x_V_nc^2': 0,
}

mapping_x_C = {
    '1': 0,
    
    'x_C': lambda alpha_choice, beta_choice: (1-alpha_choice)*beta_choice,
    
    'c_a_repeat': lambda alpha_choice, beta_choice: (alpha_choice)*beta_choice,
    
    'x_C^2': 0,
    
    'x_C c_a_repeat': 0,
    'c_a_repeat x_C': 0,
    
    'c_a_repeat^2': 0,
}

mapping_x_C_nc = {
    '1': 0,
    
    'x_C_nc': lambda alpha_choice, beta_choice: (1-alpha_choice)*beta_choice,
    
    'x_C_nc^2': 0,
}

mapping_betas = {
    'x_V_LR': lambda agent: agent._beta_reward,
    'x_V_nc': lambda agent: agent._beta_reward,
    'x_C': lambda agent: agent._beta_choice,
    'x_C_nc': lambda agent: agent._beta_choice,
}

mappings = {
    'x_V_LR': mapping_x_V_LR,
    'x_V_nc': mapping_x_V_nc,
    'x_C': mapping_x_C,
    'x_C_nc': mapping_x_C_nc,
}

# special-cases-handles
# necessary because some sindy notations in the mappings interpet the parameters differently than AgentQ
def handle_asymmetric_learning_rates(alpha_reward, alpha_penalty):
    # in AgentQ: alpha = alpha_reward if reward > 0.5 else alpha_penalty
    # in SINDy: alpha = alpha_penalty 1 - alpha_reward r 
    if alpha_reward == 0 and alpha_penalty > 0:
        alpha_reward = alpha_penalty
    elif alpha_reward == alpha_penalty:
        alpha_reward = 0
    elif alpha_penalty == 0 and alpha_reward > 0:
        alpha_penalty = 0
        alpha_reward *= -1
    return alpha_reward, alpha_penalty

# argument extractor
def argument_extractor(data, library: str):
    if library == 'x_V_LR':
        return *handle_asymmetric_learning_rates(data['alpha_reward'], data['alpha_penalty']), data['confirmation_bias'], data['beta_reward']
    elif library == 'x_V_nc':
        return data['forget_rate'], data['beta_reward']
    elif library == 'x_C':
        return data['alpha_choice'], data['beta_choice']
    elif library == 'x_C_nc':
        return data['alpha_choice'], data['beta_choice']
    else:
        raise ValueError(f'The argument extractor for the library {library} is not implemented.')

def identified_params(true_coefs: np.ndarray, recovered_coefs: np.ndarray):
    # check which true coefficients are zeros and non-zeros
    non_zero_features = true_coefs != 0
    zero_features = true_coefs == 0
    
    # check which recovered coefficients were recovered (true_pos and true_neg) correctly and which ones were not (false_pos and false_neg)
    true_pos = np.sum(recovered_coefs[non_zero_features] != 0) / np.sum(non_zero_features) if np.sum(non_zero_features) > 0 else -1
    false_neg = np.sum(recovered_coefs[non_zero_features] == 0) / np.sum(non_zero_features) if np.sum(non_zero_features) > 0 else -1
    true_neg = np.sum(recovered_coefs[zero_features] == 0) / np.sum(zero_features) if np.sum(zero_features) > 0 else -1
    false_pos = np.sum(recovered_coefs[zero_features] != 0) / np.sum(zero_features) if np.sum(zero_features) > 0 else -1
    
    return true_pos, true_neg, false_pos, false_neg

def n_true_params(true_coefs):
    # Count number of non-zero coefficients in AgentQ-parameters
    # Filter for parameter groups
    # group parameters by being either reward- or choice-based
    # if beta of one group is 0, then the other parameters can be considered also as 0s because they won't have any influence on the result
    # same goes for alphas.
    # Caution: the values will also change for true_coefs outside this function. In this scenario it's fine because these values should be modelled as 0s anyways 
    # reward-based parameter group
    if true_coefs['beta_reward'] == 0 or (true_coefs['alpha_reward'] == 0 and true_coefs['alpha_penalty'] == 0):
        true_coefs['alpha_reward'] = 0
        true_coefs['alpha_penalty'] = 0
        true_coefs['forget_rate'] = 0
        true_coefs['confirmation_bias'] = 0 
        true_coefs['beta_reward'] = 0
    # choice-based parameter group
    if true_coefs['beta_choice'] == 0 or true_coefs['alpha_choice'] == 0:
        true_coefs['alpha_choice'] = 0
        true_coefs['beta_choice'] = 0
    
    return np.sum([
        true_coefs['alpha_reward'] != 0,
        true_coefs['alpha_penalty'] != 0,
        true_coefs['alpha_choice'] != 0,
        true_coefs['beta_reward'] != 0,
        true_coefs['beta_choice'] != 0,
        true_coefs['forget_rate'] != 0,
        true_coefs['confirmation_bias'] != 0,
        ]).astype(int)

# configuration
random_sampling = [0.25, 0.5, 0.75]
n_sessions = [16, 32, 64, 128, 256, 512]
iterations = 8
base_name_data = 'data/study_recovery_s04_replaced01/data_rldm_SESSp_IT.csv'
base_name_params = 'params/study_recovery_s04/params_rldm_SESSp_IT.pkl'

# meta parameters
mapping_lens = (10, 3, 6, 3)
n_candidate_terms = np.sum(mapping_lens)
n_params_q = 7

# parameter correlation coefficients
true_params = [np.zeros((sess*iterations, n_candidate_terms)) for sess in n_sessions]
recovered_params = [np.zeros((sess*iterations, n_candidate_terms)) for sess in n_sessions]

# parameter identification rates
true_pos_sessions = np.zeros((len(n_sessions)+len(random_sampling), iterations, n_params_q+1))
true_neg_sessions = np.zeros((len(n_sessions)+len(random_sampling), iterations,  n_params_q+1))
false_pos_sessions = np.zeros((len(n_sessions)+len(random_sampling), iterations,  n_params_q+1))
false_neg_sessions = np.zeros((len(n_sessions)+len(random_sampling), iterations,  n_params_q+1))
count_true_pos = np.zeros((len(n_sessions)+len(random_sampling), iterations, n_params_q+1)) + 1e-9
count_true_neg = np.zeros((len(n_sessions)+len(random_sampling), iterations, n_params_q+1)) + 1e-9
count_false_pos = np.zeros((len(n_sessions)+len(random_sampling), iterations, n_params_q+1)) + 1e-9
count_false_neg = np.zeros((len(n_sessions)+len(random_sampling), iterations, n_params_q+1)) + 1e-9

random_sampling, n_sessions = tuple(random_sampling), tuple(n_sessions)
for index_sess, sess in enumerate(n_sessions):
    for it in range(iterations):
        # setup all sindy agents for one dataset
        path_rnn = base_name_params.replace('SESS', str(sess)).replace('IT', str(it))
        path_data = base_name_data.replace('SESS', str(sess)).replace('IT', str(it))
        agent_sindy = setup_agent_sindy(path_rnn, path_data)
        data = pd.read_csv(path_data)
        
        for index_participant, participant in enumerate(agent_sindy):
            # get all true parameters of current participant from dataset
            data_coefs_all = data.loc[data['session']==participant].iloc[-1]
            index_params = n_true_params(data_coefs_all)
            
            index_all_candidate_terms = 0
            for index_library, library in enumerate(mappings):
                # get sindy coefficients
                sindy_coefs_array = agent_sindy[participant]._models[library].model.steps[-1][1].coef_[0]
                # drop every entry feature that contains a u-feature (i.e. dummy-feature)
                feature_names = agent_sindy[participant]._models[library].get_feature_names()
                index_keep = ['u' not in feature for feature in feature_names]
                sindy_coefs_array = sindy_coefs_array[index_keep]
                feature_names = np.array(feature_names)[index_keep]
                sindy_coefs = {f: sindy_coefs_array[i] for i, f in enumerate(feature_names)}
                
                # translate data coefficient to sindy coefficients
                data_coefs = {f: 0 for f in feature_names}
                data_coefs_array = np.zeros(len(feature_names))
                for index_feature, feature in enumerate(feature_names):
                    if not isinstance(mappings[library][feature], int):
                        data_coefs[feature] = mappings[library][feature](*argument_extractor(data_coefs_all, library))
                        data_coefs_array[index_feature] = mappings[library][feature](*argument_extractor(data_coefs_all, library))
                
                # add true and recovered parameters for later parameter correlation
                # true params are already scaled by beta
                true_params[index_sess][sess*it+index_participant, index_all_candidate_terms:index_all_candidate_terms+len(feature_names)] = data_coefs_array
                # recovered params still need beta-scaling -> Was not necessary for mere identification rates (although would have been good)
                recovered_params[index_sess][sess*it+index_participant, index_all_candidate_terms:index_all_candidate_terms+len(feature_names)] = sindy_coefs_array * mapping_betas[library](agent_sindy[participant])
                index_all_candidate_terms += len(feature_names)
                
                # compute number of correctly identified+omitted parameters
                true_pos, true_neg, false_pos, false_neg = identified_params(data_coefs_array, sindy_coefs_array * mapping_betas[library](agent_sindy[participant]))
                
                # add identification rates
                true_pos_sessions[index_sess, it, index_params] += true_pos if true_pos > -1 else 0
                true_neg_sessions[index_sess, it, index_params] += true_neg if true_neg > -1 else 0
                false_pos_sessions[index_sess, it, index_params] += false_pos if false_pos > -1 else 0
                false_neg_sessions[index_sess, it, index_params] += false_neg if false_neg > -1 else 0
                
                # add identification rate counter
                count_true_pos[index_sess, it, index_params] += 1 if true_pos > -1 else 0
                count_true_neg[index_sess, it, index_params] += 1 if true_neg > -1 else 0
                count_false_pos[index_sess, it, index_params] += 1 if false_pos > -1 else 0
                count_false_neg[index_sess, it, index_params] += 1 if false_neg > -1 else 0
                
                # sample random coefficients for biggest dataset
                if n_sessions[index_sess] == max(n_sessions):
                    # sample random coefficients for each random sampling strategy
                    for index_rnd, rnd in enumerate(random_sampling):
                        rnd_coefs = np.random.choice((1, 0), p=(rnd, 1-rnd), size=sindy_coefs_array.shape)
                        rnd_betas = np.random.choice((1, 0), p=(rnd, 1-rnd), size=2)
                        
                        # do same stuff as with sindy coefs
                        # compute number of correctly identified+omitted parameters
                        true_pos, true_neg, false_pos, false_neg = identified_params(data_coefs_array, rnd_coefs)# * (rnd_betas[1] if 'x_C' in library else rnd_betas[0]))
                        
                        # add identification rates
                        true_pos_sessions[len(n_sessions)+index_rnd, it, index_params] += true_pos if true_pos > -1 else 0
                        true_neg_sessions[len(n_sessions)+index_rnd, it, index_params] += true_neg if true_neg > -1 else 0
                        false_pos_sessions[len(n_sessions)+index_rnd, it, index_params] += false_pos if false_pos > -1 else 0
                        false_neg_sessions[len(n_sessions)+index_rnd, it, index_params] += false_neg if false_neg > -1 else 0
                        
                        # add identification rate counter
                        count_true_pos[len(n_sessions)+index_rnd, it, index_params] += 1 if true_pos > -1 else 0
                        count_true_neg[len(n_sessions)+index_rnd, it, index_params] += 1 if true_neg > -1 else 0
                        count_false_pos[len(n_sessions)+index_rnd, it, index_params] += 1 if false_pos > -1 else 0
                        count_false_neg[len(n_sessions)+index_rnd, it, index_params] += 1 if false_neg > -1 else 0

# ------------------------------------------------
# post-processing coefficient correlation
# ------------------------------------------------

# remove outliers in both true and recovered coefs where recovered coefficients are bigger than a big threshold (e.g. abs(recovered_coeff) > 1e1)
threshold = 1e1
removed_params = 0
for index_sess in range(len(n_sessions)):
    index_remove = recovered_params[index_sess] > threshold
    true_params[index_sess][index_remove] = np.nan
    recovered_params[index_sess][index_remove] = np.nan
    removed_params += np.sum(index_remove)
print(f'excluded parameters because of high values: {removed_params}')

# get all features across all libraries; remove dummy features with are named 'u'
feature_names = []
for library in mappings:
    feature_names += agent_sindy[participant]._models[library].get_feature_names()
index_keep = ['u' not in feature for feature in feature_names]
feature_names = np.array(feature_names)[index_keep]

correlation_matrix, recovery_errors, recovery_errors_median, recovery_errors_std = [], [], [], []
for index_sess in range(len(n_sessions)):
    correlation_matrix.append(
        pd.DataFrame(
            np.corrcoef(
                true_params[index_sess], 
                recovered_params[index_sess], 
                rowvar=False,
                )[len(feature_names):, :len(feature_names)],
            columns=feature_names,
            index=feature_names,
            )
        )

    # normalizing params
    v_max = np.nanmax(true_params[index_sess], axis=0)
    v_min = np.nanmin(true_params[index_sess], axis=0)
    index_normalize = v_max-v_min != 0
    
    true_params[index_sess] = (true_params[index_sess] - v_min)
    true_params[index_sess][:, index_normalize] = true_params[index_sess][:, index_normalize] / (v_max - v_min)[index_normalize]
    
    recovered_params[index_sess] = (recovered_params[index_sess] - v_min) 
    recovered_params[index_sess][:, index_normalize] = recovered_params[index_sess][:, index_normalize]/ (v_max - v_min)[index_normalize]
    
    recovery_errors.append(
        true_params[index_sess] - recovered_params[index_sess]
        )

# ------------------------------------------------
# post-processing identification rates
# ------------------------------------------------

# average across counts
true_pos_sessions /= count_true_pos
true_neg_sessions /= count_true_neg
false_pos_sessions /= count_false_pos
false_neg_sessions /= count_false_neg

true_pos_sessions_mean = np.mean(true_pos_sessions, axis=1)
true_neg_sessions_mean = np.mean(true_neg_sessions, axis=1)
false_pos_sessions_mean = np.mean(false_pos_sessions, axis=1)
false_neg_sessions_mean = np.mean(false_neg_sessions, axis=1)

true_pos_sessions_std = np.std(true_pos_sessions, axis=1)
true_neg_sessions_std = np.std(true_neg_sessions, axis=1)
false_pos_sessions_std = np.std(false_pos_sessions, axis=1)
false_neg_sessions_std = np.std(false_neg_sessions, axis=1)

# ------------------------------------------------
# configuration identification rates plots
# ------------------------------------------------

v_min = 0
v_max = np.nanmax(np.stack((true_pos_sessions_mean, false_pos_sessions_mean, true_neg_sessions_mean, false_neg_sessions_mean)), axis=(-1, -2, -3))

identification_matrix_mean = [
    [true_pos_sessions_mean , false_pos_sessions_mean], 
    [false_neg_sessions_mean , true_neg_sessions_mean],
    ]

identification_matrix_std = [
    [true_pos_sessions_std, false_pos_sessions_std], 
    [false_neg_sessions_std, true_neg_sessions_std],
    ]

identification_headers = [
    ['true positive', 'false pos'],
    ['false negative', 'true negative'],
]

identification_x_axis_labels = [
    [None, None],
    ['$n_{parameters}$', '$n_{parameters}$'],
]

bin_edges_params = np.arange(0, n_params_q+1)
identification_x_axis_ticks = [
    [bin_edges_params, bin_edges_params],
    [bin_edges_params, bin_edges_params],
]

identification_y_axis_labels = [
    ['$n_\{sessions\}$', None],
    ['$n_\{sessions\}$', None],
]

y_tick_labels = n_sessions + random_sampling

linestyles = ['-'] * len(n_sessions) + ['--'] * len(random_sampling)
alphas = [0.3] * len(n_sessions) + [0] * len(random_sampling)

# ------------------------------------------------
# parameter identification rates
# ------------------------------------------------

# heatmaps 

fig, axs = plt.subplots(
    nrows=len(identification_matrix_mean), 
    ncols=len(identification_matrix_mean[0])+1,
    gridspec_kw={
        'width_ratios': [10]*len(identification_matrix_mean[0]) + [1],
        },
    )

for index_row, row in enumerate(identification_matrix_mean):
    for index_col, col in enumerate(row):
        if col is not None:
            sns.heatmap(
                col, 
                annot=True, 
                cmap='viridis',
                center=0,
                ax=axs[index_row, index_col],
                cbar=True if index_col == len(identification_matrix_mean[0])-1 else False, 
                cbar_ax=axs[index_row, len(identification_matrix_mean[0])],
                xticklabels=np.arange(0, n_params_q+1) if index_row == len(identification_matrix_mean)-1 else ['']*n_params_q, 
                yticklabels=y_tick_labels if index_col == 0 else ['']*len(n_sessions), 
                vmin=v_min,
                vmax=v_max,
                )
plt.show()

# line plots

fig, axs = plt.subplots(
    nrows=len(identification_matrix_mean), 
    ncols=len(identification_matrix_mean[0]),
    sharey=True,
    )

for index_row, row in enumerate(identification_matrix_mean):
    for index_col, col in enumerate(row):
        if col is not None:
            ax = axs[index_row, index_col]
            for index_sample_size, sample_size in enumerate(y_tick_labels):
                std = np.std(col[index_sample_size])
                ax.fill_between(
                    x=np.arange(0, n_params_q+1),
                    y1=col[index_sample_size] + identification_matrix_std[index_row][index_col][index_sample_size],
                    y2=col[index_sample_size] - identification_matrix_std[index_row][index_col][index_sample_size],
                    alpha=alphas[index_sample_size]
                    )
                ax.plot(
                    np.arange(0, n_params_q+1),
                    col[index_sample_size],
                    linestyles[index_sample_size],
                    label=sample_size,
                    )
                # Set titles for individual subplots
                ax.set_title(identification_headers[index_row][index_col])
                
                # Configure x-axis labels and ticks only for the lowest row
                if index_row == len(identification_matrix_mean) - 1:
                    ax.set_xlabel('$n_{parameters}$')
                    ax.set_xticks(bin_edges_params)
                else:
                    ax.set_xticklabels([])

                ax.set_ylim([0, 1])
axs[0, 0].legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------
# parameter correlation coefficients
# ------------------------------------------------

# heatmaps per dataset size

v_min, v_max = -1, 1
fig, axs = plt.subplots(
    nrows=1, 
    ncols=len(n_sessions)+1,
    gridspec_kw={
        'width_ratios': [10]*len(n_sessions) + [1],
        },
    )

for index_sess in range(len(n_sessions)):
    sns.heatmap(
        correlation_matrix[index_sess],
        annot=True,
        cmap='viridis',
        center=0,
        ax=axs[index_sess],
        cbar=True if index_sess == len(n_sessions)-1 else False,
        cbar_ax=axs[index_sess+1],
        xticklabels=feature_names,
        yticklabels=feature_names if index_sess == 0 else ['']*len(feature_names),
        vmin=v_min,
        vmax=v_max,
        )
plt.show()

# box plot

fig, axs = plt.subplots(
    nrows=1, 
    ncols=max((len(n_sessions), 2)),
    sharey=True,
    )

for index_sess in range(len(n_sessions)):
    ax = axs[index_sess]
    sns.boxplot(
        pd.DataFrame(
            recovery_errors[index_sess],
            columns=feature_names,
        ),
        ax=ax,
        showfliers=False,
    )
    ax.plot(feature_names[-1], 0, '--', color='tab:gray', linewidth=0.5)
    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title(n_sessions[index_sess])
plt.show()

#  line plot

fig, axs = plt.subplots(
    nrows=max((len(n_sessions), 2)),
    ncols=len(feature_names),
    )

for index_sess in range(len(n_sessions)):
    for index_feature in range(len(feature_names)):
        ax = axs[index_sess, index_feature]
        
        ax.plot(true_params[index_sess][:, index_feature], recovered_params[index_sess][:, index_feature], 'o', color='tab:red', alpha=0.2)
        ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', color='tab:gray')
        
        # Axes settings
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        
        if index_sess != len(n_sessions) - 1:
            ax.tick_params(axis='x', which='both', labelbottom=False)  # No x-axis ticks
        else:
            ax.set_xlabel('True', fontsize=10)
            ax.tick_params(axis='x', labelsize=8)
        
        if index_feature != 0:
            ax.tick_params(axis='y', which='both', labelleft=False)  # No y-axis ticks
        else:
            ax.set_ylabel(f'{n_sessions[index_sess]} participants\nRecovered', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
        
        if index_sess == 0:
            ax.set_title(feature_names[index_feature], fontsize=10)
        
plt.show()

# box plot
# like line plot before but with box-whisker

# Parameters for binning
num_bins = 10  # Number of bins for 5% width (0.05 * 100 = 20 bins)
bin_edges = np.linspace(0, 1, num_bins + 1)

fig, axs = plt.subplots(
    nrows=max(len(n_sessions), 2),
    ncols=len(feature_names),
    figsize=(12, 6)
)

for index_sess in range(len(n_sessions)):
    for index_feature in range(len(feature_names)):
        ax = axs[index_sess, index_feature]
        
        # Prepare the data
        true_vals = true_params[index_sess][:, index_feature]
        recovered_vals = recovered_params[index_sess][:, index_feature]
        
        # Bin the data based on true parameters
        bins = np.digitize(true_vals, bin_edges) - 1  # Binning index
        bins = np.clip(bins, 0, num_bins - 1)  # Ensure bin indices are within range
        
        # Compute box-plot statistics for each bin
        box_data = [recovered_vals[bins == i] for i in range(num_bins)]
        median = [np.mean(data) if len(data) > 0 else np.nan for data in box_data]

        # Create box-whisker plot
        ax.boxplot(
            box_data, 
            positions=bin_edges[:-1] + 0.025, 
            widths=0.04, 
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="tab:red", color="tab:red", alpha=0.2),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="tab:red", linestyle="--", alpha=0.5),
            capprops=dict(color="tab:red", alpha=0.5)
            )

        # Reference line
        ax.plot([0, 1], [0, 1], '--', color='tab:gray', linewidth=1)
        
        # Axes settings
        ax.set_ylim(-.1, 1.1)
        ax.set_xlim(-.1, 1.1)
        
        if index_sess != len(n_sessions) - 1:
            ax.tick_params(axis='x', which='both', labelbottom=False)  # No x-axis ticks
        else:
            ax.set_xlabel('True', fontsize=10)
            ax.tick_params(axis='x', labelsize=8)
        
        if index_feature != 0:
            ax.tick_params(axis='y', which='both', labelleft=False)  # No y-axis ticks
        else:
            ax.set_ylabel(f'{n_sessions[index_sess]} participants\nRecovered', fontsize=10)
            ax.tick_params(axis='y', labelsize=8)
        
        if index_sess == 0:
            ax.set_title(feature_names[index_feature], fontsize=10)

plt.tight_layout()
plt.show()
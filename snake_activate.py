import json
import numpy as np

with open("activation_history_episode_10001.json", 'r') as f:
    history = json.load(f)

# Load weights from step 0 (assuming constant)
step_0_data = history[0]['step_0']
weights = np.array(step_0_data['weights'])  # (4, 32)
if weights.shape[0] == 4 and weights.shape[1] == 32:
    weights = weights.T  # (32, 4)
bias = np.array(step_0_data['bias'])  # (4,)

# Track safe and bad actions per neuron
safe_counts = np.zeros(32, dtype=int)
bad_counts = np.zeros(32, dtype=int)

for step_data in history:
    step_key = list(step_data.keys())[0]
    pre_relu = np.array(step_data[step_key]['pre_relu'])
    active = np.array(step_data[step_key]['active'])
    chosen_action = step_data[step_key]['chosen_action']
    is_bad_action = step_data[step_key]['is_bad_action']
    
    h = np.where(active, pre_relu, 0)  # Post-ReLU activations
    
    for i in range(32):
        if h[i] > 0:  # Active neuron
            contribs = weights[i, :] * h[i]
            preferred_action = np.argmax(contribs)
            if preferred_action == chosen_action:
                if is_bad_action:
                    bad_counts[i] += 1
                else:
                    safe_counts[i] += 1

# Compute win rate (safe / (safe + bad))
total_counts = safe_counts + bad_counts
win_rates = np.where(total_counts > 0, safe_counts / total_counts, 0.0)

# Print results
print(f"Total steps analyzed: {len(history)}")
print("\nNeuron Performance (Safe vs. Bad Actions):")
for i in range(32):
    print(f"Neuron {i}: Safe={safe_counts[i]}, Bad={bad_counts[i]}, Win Rate={win_rates[i]:.2%}")
    if safe_counts[i] == 0 and bad_counts[i] == 0:
        print("  (Inactive or never matched chosen action)")

# Best performing neurons
best_neurons = np.argsort(win_rates)[::-1]  # Sort descending
print("\nTop 5 Best Performing Neurons:")
for i in best_neurons[:5]:
    print(f"Neuron {i}: Win Rate={win_rates[i]:.2%} (Safe={safe_counts[i]}, Bad={bad_counts[i]})")

# Worst performing neurons
print("\nBottom 5 Performing Neurons (excluding inactive):")
active_neurons = [i for i in range(32) if total_counts[i] > 0]
worst_neurons = np.argsort(win_rates[active_neurons])  # Sort ascending among active
for idx in worst_neurons[:5]:
    i = active_neurons[idx]
    print(f"Neuron {i}: Win Rate={win_rates[i]:.2%} (Safe={safe_counts[i]}, Bad={bad_counts[i]})")

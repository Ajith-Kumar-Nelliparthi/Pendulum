import gymnasium as gym
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# Create the environment
env = gym.make('Pendulum-v1', render_mode='rgb_array')
BEST_MODEL_FILE = "pendulum_q_table.npy"
best_score = float('-inf')
video_folder = "Videos"


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
SHOW_EVERY = 1000

DISCRETE_OS_SIZE = [20, 20, 20]  # Discretizing the state space
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Discretizing the action space
DISCRETE_ACTIONS = 10  # Define 10 discrete actions
ACTION_SPACE = np.linspace(env.action_space.low[0], env.action_space.high[0], DISCRETE_ACTIONS)

epsilon = 1.0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 3
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Initialize Q-table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [DISCRETE_ACTIONS]))

def get_discrete_state(state):
    """Convert continuous state to discrete state and prevent out-of-bounds indexing."""
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(np.clip(discrete_state.astype(int), 0, np.array(DISCRETE_OS_SIZE) - 1))

successful_episode = None
frames = []

for episode in range(EPISODES):
    total_reward = 0
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    render = episode % SHOW_EVERY == 0

    if render:
        print(f"Episode: {episode}")
        
    while not done:
        # Choose action
        action_idx = np.argmax(q_table[discrete_state]) if np.random.random() > epsilon else np.random.randint(0, DISCRETE_ACTIONS)
        action = [ACTION_SPACE[action_idx]]  # Convert discrete index to continuous action
        
        # Take action
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        done = terminated or truncated

        # Penalize large pole angles, but keep it mild
        reward -= 0.1 * abs(new_state[2])

        if render:
            frames.append(env.render())  # Only store frames in "rgb_array" mode

        total_reward += reward

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action_idx,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action_idx,)] = new_q
        else:
            q_table[discrete_state + (action_idx,)] = reward
        
        discrete_state = new_discrete_state

    # Decay epsilon
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon = max(0.01, epsilon - epsilon_decay_value) 

    if total_reward > best_score:
        best_score = total_reward
        np.save(BEST_MODEL_FILE, q_table)
        print(f"New best model saved at Episode {episode}, Score: {best_score}")
        successful_episode = episode

env.close()

# Save video of the successful episode
if successful_episode is not None and frames:
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(f"{video_folder}/success.mp4", codec="libx264")
    print("Saved successful episode as success.mp4")

# Load trained Q-table for testing
q_table = np.load(BEST_MODEL_FILE)

# Testing the trained model
env = gym.make('Pendulum-v1', render_mode='human')
state, _ = env.reset()
discrete_state = get_discrete_state(state)
done = False

while not done:
    action_idx = np.argmax(q_table[discrete_state])
    action = [ACTION_SPACE[action_idx]]  # Convert discrete index to continuous action
    new_state, reward, terminated, truncated, _ = env.step(action)
    discrete_state = get_discrete_state(new_state)
    done = terminated or truncated
    env.render()

env.close()

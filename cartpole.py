import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import gymnasium as gym 
import os
import torch
import unittest
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from os import listdir
from tensorflow.python.summary.summary_iterator import summary_iterator


class LogStepsCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        self.log_dir = log_dir
        super(LogStepsCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        self.results = pd.DataFrame(columns=['Reward', 'Done'])
        print("Training starts!")

    def _on_step(self) -> bool:
        if 'reward' in self.locals:
            keys = ['reward', 'done']
        else:
            keys = ['rewards', 'dones']
        self.results.loc[len(self.results)] = [self.locals[keys[0]][0], self.locals[keys[1]][0]]
        return True

    def _on_training_end(self) -> None:
        self.results.to_csv(self.log_dir + 'training_data.csv', index=False)
        print("Training ends!")


class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals['total_timesteps'])

    def _on_step(self):
        self.progress_bar.update(1)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None


def save_dict_to_file(dict, path, txt_name='hyperparameter_dict'):
    f = open(path + '/' + txt_name + '.txt', 'w')
    f.write(str(dict))
    f.close()


def calc_episode_rewards(training_data):
    # Calculate the rewards for each training episode
    episode_rewards = []
    temp_reward_sum = 0

    for step in range(training_data.shape[0]):
        reward, done = training_data.iloc[step, :]
        temp_reward_sum += reward
        if done:
            episode_rewards.append(temp_reward_sum)
            temp_reward_sum = 0

    result = pd.DataFrame(columns=['Reward'])
    result['Reward'] = episode_rewards
    return result


def learning_curve(episode_rewards, log_dir, window=10):
    # Calculate rolling window metrics
    rolling_average = episode_rewards.rolling(window=window, min_periods=window).mean().dropna()
    rolling_max = episode_rewards.rolling(window=window, min_periods=window).max().dropna()
    rolling_min = episode_rewards.rolling(window=window, min_periods=window).min().dropna()

    # Change column name
    rolling_average.columns = ['Average Reward']
    rolling_max.columns = ['Max Reward']
    rolling_min.columns = ['Min Reward']
    rolling_data = pd.concat([rolling_average, rolling_max, rolling_min], axis=1)

    # Plot
    sns.set()
    plt.figure(0)
    ax = sns.lineplot(data=rolling_data)
    ax.fill_between(rolling_average.index, rolling_min.iloc[:, 0], rolling_max.iloc[:, 0], alpha=0.2)
    ax.set_title('Learning Curve')
    ax.set_ylabel('Reward')
    ax.set_xlabel('Episodes')

    # Save figure
    plt.savefig(log_dir + 'learning_curve' + str(window) + '.png')


def learning_curve_baselines(log_dir, window=10):
    # Read data
    training_data = pd.read_csv(log_dir + 'training_data.csv', index_col=None)

    # Calculate episode rewards
    episode_rewards = calc_episode_rewards(training_data)

    learning_curve(episode_rewards=episode_rewards, log_dir=log_dir, window=window)


def learning_curve_tianshou(log_dir, window=10):
    # Find event file
    files = listdir(log_dir)
    for f in files:
        if 'events' in f:
            event_file = f
            break

    # Read episode rewards
    episode_rewards_list = []
    episode_rewards = pd.DataFrame(columns=['Reward'])
    try:
        for e in summary_iterator(log_dir + event_file):
            if len(e.summary.value) > 0:
                if e.summary.value[0].tag == 'train/reward':
                    episode_rewards_list.append(e.summary.value[0].simple_value)
    except Exception as e:
        pass
    episode_rewards['Reward'] = episode_rewards_list

    # Learning curve
    learning_curve(episode_rewards, log_dir, window=window)


def learning_curve_tianshou_multiple_runs(log_dirs, window=10):
    episode_rewards_list = []
    episode_rewards = pd.DataFrame(columns=['Reward'])

    for log_dir in log_dirs:
        # Find event file
        files = listdir(log_dir)
        for f in files:
            if 'events' in f:
                event_file = f
                break

        # Read episode rewards

        try:
            for e in summary_iterator(log_dir + event_file):
                if len(e.summary.value) > 0:
                    if e.summary.value[0].tag == 'train/reward':
                        episode_rewards_list.append(e.summary.value[0].simple_value)
        except Exception as e:
            pass
    episode_rewards['Reward'] = episode_rewards_list

    # Learning curve
    learning_curve(episode_rewards, log_dir, window=window)



# Create log directory
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# Create the environment
env = gym.make('CartPole-v1')

# Initialize the PPO agent with CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PPO('MlpPolicy', env, verbose=1, device=device)

# Define callbacks
log_steps_callback = LogStepsCallback(log_dir)
tqdm_callback = TqdmCallback()

# Train the model
model.learn(total_timesteps=10000, callback=[log_steps_callback, tqdm_callback])

# Save the model
model.save(log_dir + 'ppo_cartpole')

# Optional: Plot the learning curve
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the training data
training_data = pd.read_csv(log_dir + 'training_data.csv')
episode_rewards = training_data['Reward']
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

class CustomLogCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, 'training_data.csv')
        self.episode_rewards = []

    def _on_step(self):
        if done := self.locals.get("done", False):
            reward = self.locals["rewards"][0]  # Reward of the last completed episode
            self.episode_rewards.append(reward)
            pd.DataFrame({"Reward": self.episode_rewards}).to_csv(self.log_file, index=False)
        return True

class TestPPOTraining(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.log_dir = './test_logs/'
        os.makedirs(cls.log_dir, exist_ok=True)

        # Create environment
        cls.env = gym.make('CartPole-v1')
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = PPO('MlpPolicy', cls.env, verbose=1, device=cls.device)

        # Define custom log callback
        cls.log_callback = CustomLogCallback(cls.log_dir)

    def test_environment_creation(self):
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.unwrapped.spec.id, 'CartPole-v1')
    
    def test_model_initialization(self):
        self.assertEqual(self.model.device, torch.device(self.device))
        self.assertTrue(hasattr(self.model, 'policy'))
    
    def test_training_process(self):
        try:
            self.model.learn(total_timesteps=1000, callback=[self.log_callback])
        except Exception as e:
            self.fail(f"Training process failed with an exception: {e}")

    def test_model_saving(self):
        save_path = os.path.join(self.log_dir, 'ppo_cartpole_test')
        self.model.save(save_path)
        self.assertTrue(os.path.isfile(save_path + '.zip'))

    def test_training_log(self):
        log_file = os.path.join(self.log_dir, 'training_data.csv')
        self.assertTrue(os.path.isfile(log_file))
        
        # Verify training data is logged
        training_data = pd.read_csv(log_file)
        self.assertGreater(len(training_data), 0)
        self.assertIn('Reward', training_data.columns)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        shutil.rmtree(cls.log_dir)

suite = unittest.TestLoader().loadTestsFromTestCase(TestPPOTraining)
unittest.TextTestRunner().run(suite)

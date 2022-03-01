import torch
from torch import nn
from pathlib import Path
import random
import shutil
import pickle

# Preprocessing wrappers
from Preprocess import *
from Logger import *
# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY


class DQNSolver(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers
    """

    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        print(conv_out_size)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class DuelDDQNSolver(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers separated into A/V branches
    """

    def __init__(self, input_shape, n_actions):
        super(DuelDDQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU())

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU())

        self.value = nn.Sequential(nn.Linear(512, 1))

        self.advantage = nn.Sequential(nn.Linear(512, n_actions))

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """ Forward bass including collapsing of trunks"""

        conv_out = self.conv(x).view(x.size()[0], -1)

        value = self.fc_value(conv_out)
        advantage = self.fc_advantage(conv_out)

        value = self.value(value)
        advantage = self.advantage(advantage)

        avg_advantage = torch.mean(advantage, dim=1, keepdim=True)
        Q = value + advantage - avg_advantage  # Combination of trunks

        return Q


class Agent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr
                 , exploration_max, exploration_min, exploration_decay, network, pretrained, pretrain_dir,
                 replay):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.network_type = network

        self.DQN = True
        self.DDQN = False
        self.DuelDDQN = False

        if self.network_type == "DQN":
            self.DQN = True
        elif self.network_type == "DDQN":
            self.DDQN = True
        elif self.network_type == "DuelDDQN":
            self.DuelDDQN = True
        else:
            raise ValueError('Incorrect network type: {}'.format(network))

        # self.double_dqn = double_dqn
        self.pretrained = pretrained
        self.replay = replay

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using {}".format(self.device))

        # Double DQN network
        if self.DDQN:
            self.local_net = DQNSolver(state_space, action_space).to(self.device)
            self.target_net = DQNSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                self.local_net.load_state_dict(
                    torch.load(pretrain_dir + "DQN1.pt", map_location=torch.device(self.device)))
                self.target_net.load_state_dict(
                    torch.load(pretrain_dir + "DQN2.pt", map_location=torch.device(self.device)))

            self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
            self.copy = 5000
            self.step = 0

        # DQN network
        elif self.DuelDDQN:
            self.local_net = DuelDDQNSolver(state_space, action_space).to(self.device)
            self.target_net = DuelDDQNSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                self.local_net.load_state_dict(
                    torch.load(pretrain_dir + "DQN1.pt", map_location=torch.device(self.device)))
                self.target_net.load_state_dict(
                    torch.load(pretrain_dir + "DQN2.pt", map_location=torch.device(self.device)))

            self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
            self.copy = 5000
            self.step = 0

        elif self.DQN:
            self.dqn = DQNSolver(state_space, action_space).to(self.device)

            if self.pretrained:
                self.dqn.load_state_dict(torch.load(pretrain_dir + "DQN.pt", map_location=torch.device(self.device)))
            self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained and not self.replay:

            self.state_memory = torch.load(pretrain_dir + "state_memory.pt")
            self.action_memory = torch.load(pretrain_dir + "action_memory.pt")
            self.reward_memory = torch.load(pretrain_dir + "reward_memory.pt")
            self.state2_memory = torch.load(pretrain_dir + "state2_memory.pt")
            self.done_memory = torch.load(pretrain_dir + "done_memory.pt")

            with open(pretrain_dir + "ending_position.pkl", 'rb') as f:
                self.ending_position = pickle.load(f)
            with open(pretrain_dir + "num_in_queue.pkl", 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            self.state_memory = torch.zeros(max_memory_size, *self.state_space)
            self.action_memory = torch.zeros(max_memory_size, 1)
            self.reward_memory = torch.zeros(max_memory_size, 1)
            self.state2_memory = torch.zeros(max_memory_size, *self.state_space)
            self.done_memory = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        if self.replay:
            self.exploration_max = exploration_min
            self.exploration_rate = exploration_min
            self.exploration_min = exploration_min
            self.exploration_decay = 1

    def remember(self, state, action, reward, state2, done):
        """Store state within replay buffers"""

        self.state_memory[self.ending_position] = state.float()
        self.action_memory[self.ending_position] = action.float()
        self.reward_memory[self.ending_position] = reward.float()
        self.state2_memory[self.ending_position] = state2.float()
        self.done_memory[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        """Sample with uniform weights from replay buffer"""

        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        state = self.state_memory[idx]
        action = self.action_memory[idx]
        reward = self.reward_memory[idx]
        state2 = self.state2_memory[idx]
        done = self.done_memory[idx]
        return state, action, reward, state2, done

    def act(self, state):
        """Epsilon-greedy action"""
        if self.DDQN or self.DuelDDQN:
            self.step += 1
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])

        if self.DDQN or self.DuelDDQN:
            return torch.argmax(self.local_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    def copy_model(self):
        """Clone weights for NN"""
        self.target_net.load_state_dict(self.local_net.state_dict())

    def experience_replay(self):
        """Use the double Q-update or Q-update equations to update the network weights"""

        if (self.DDQN or self.DuelDDQN) and self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        # Sample from replay buffer
        state, action, reward, state2, done = self.batch_experiences()
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        state2 = state2.to(self.device)
        done = done.to(self.device)

        self.optimizer.zero_grad()
        if self.DDQN:
            target = reward + torch.mul((self.gamma * self.target_net(state2).max(1).values.unsqueeze(1)), 1 - done)
            current = self.local_net(state).gather(1, action.long())
        else:
            target = reward + torch.mul((self.gamma * self.dqn(state2).max(1).values.unsqueeze(1)), 1 - done)

            current = self.dqn(state).gather(1, action.long())

        # Updating network
        loss = self.l1(current, target)
        loss.backward()
        self.optimizer.step()

        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

        return loss.item()

    def freeze_layers(self):
        pass


def main():

    # World configuration
    world = "1"
    level = "1"
    pretrained = True
    replay = True
    checkpoint_dir = 'Agent_NN/Duels/1-1/'
    movement_space = RIGHT_ONLY
    network = "DuelDDQN"
    exploration_max = 0.01
    exploration_min = 0.01
    exploration_decay = 1

    if replay:
        print("\nStarting replay of {} on level {}-{}".format(network, world, level))
        print("Slowing down gameplay for human viewing")
        print("Epsilon = {}".format(exploration_max))

    # Configuring environment
    env = gym_super_mario_bros.make('SuperMarioBros-{0}-{1}-v0'.format(world, level))
    env = FrameSkip(env, frames=4, limit=1 / 150, render_game=True)
    env = Rescale(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = JoypadSpace(env, movement_space)
    print('Environment configured')

    Mario = Agent(state_space=env.observation_space.shape,
                  action_space=env.action_space.n,
                  max_memory_size=30000,
                  batch_size=32,
                  gamma=0.90,
                  lr=0.00025,
                  exploration_max=exploration_max,
                  exploration_min=exploration_min,
                  exploration_decay=exploration_decay,
                  network=network,
                  replay=replay,
                  pretrained=pretrained,
                  pretrain_dir=checkpoint_dir)

    total_rewards = []
    save_dir = Path('checkpoint') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if not replay:
        save_dir.mkdir(parents=True)
    temp_save_dir = str(save_dir) + "/"

    episode = 0
    total_steps = 0
    if pretrained and not replay:
        with open(checkpoint_dir + "total_rewards.pkl", 'rb') as f:
            total_rewards = pickle.load(f)
        with open(checkpoint_dir + "episode_num.pkl", 'rb') as f:
            episode, Mario.exploration_rate, total_steps = pickle.load(f)
            episode += 1
        shutil.copy(checkpoint_dir + "log_checkpoint", temp_save_dir + "log")
        print('copied log data')

    num_episodes = 10000
    max_steps = 50000

    logger = MetricLogger(save_dir)

    for ep_num in range(episode, num_episodes):

        state = env.reset()
        state = torch.Tensor(np.array([state]))

        total_reward = 0
        steps = 0
        episode_reward = []
        completed_level = False

        for step in range(max_steps):

            action = Mario.act(state)
            steps += 1
            total_steps += 1
            state_next, reward, done, info = env.step(int(action[0]))

            completed_level = info['flag_get']
            total_reward += reward
            episode_reward.append(reward)

            if np.sum(episode_reward[-50:]) < 0:
                break

            state_next = torch.Tensor(np.array([state_next]))
            reward_tensor = torch.tensor([reward]).unsqueeze(0)

            terminal = torch.tensor(np.array([int(done)])).unsqueeze(0)

            loss = 0
            if not Mario.replay:
                Mario.remember(state, action, reward_tensor, state_next, terminal)
                if ep_num > 200:
                    loss = Mario.experience_replay()

            logger.log_step(reward, loss, q=0)
            state = state_next

            if terminal:
                break

        Mario.exploration_rate *= Mario.exploration_decay
        total_rewards.append(total_reward)
        num_episodes += 1
        logger.log_episode()

        if ep_num % 1 == 0:
            logger.record(episode=ep_num,
                          epsilon=Mario.exploration_rate,
                          step=total_steps,
                          done=completed_level,
                          save=not replay)

        if not Mario.replay:
            if ep_num != 0 and ep_num % 500 == 0:

                print("saving_checkpoint")
                with open(temp_save_dir + "ending_position.pkl", "wb") as f:
                    pickle.dump(Mario.ending_position, f)
                with open(temp_save_dir + "num_in_queue.pkl", "wb") as f:
                    pickle.dump(Mario.num_in_queue, f)
                with open(temp_save_dir + "total_rewards.pkl", "wb") as f:
                    pickle.dump(total_rewards, f)
                if Mario.DDQN or Mario.DuelDDQN:
                    torch.save(Mario.local_net.state_dict(), temp_save_dir + "DQN1.pt")
                    torch.save(Mario.target_net.state_dict(), temp_save_dir + "DQN2.pt")
                else:
                    torch.save(Mario.dqn.state_dict(), temp_save_dir + "DQN.pt")
                torch.save(Mario.state_memory, temp_save_dir + "state_memory.pt")
                torch.save(Mario.action_memory, temp_save_dir + "action_memory.pt")
                torch.save(Mario.reward_memory, temp_save_dir + "reward_memory.pt")
                torch.save(Mario.state2_memory, temp_save_dir + "state2_memory.pt")
                torch.save(Mario.done_memory, temp_save_dir + "done_memory.pt")

                # Saving episode number, such that data can be loaded from logger corresponding to checkpoint
                with open(temp_save_dir + "episode_num.pkl", "wb") as f:
                    pickle.dump([ep_num, Mario.exploration_rate, total_steps], f)

                shutil.copy(temp_save_dir + "log", temp_save_dir + "log_checkpoint")
                print('copied log')

    env.close()


if __name__ == '__main__':
    main()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_sum = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        output = self.fc_out(weighted_sum)
        return output


class ConvWithSelfAttention(nn.Module):
    def __init__(self, input_channels, conv_out_channels, kernel_size, grid_number, num_heads, fc_out_features=128):
        super(ConvWithSelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, conv_out_channels, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.self_attention = MultiHeadSelfAttention(conv_out_channels, conv_out_channels, num_heads)
        self.fc = nn.Linear((grid_number // 4) ** 2 * conv_out_channels, fc_out_features)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # Reshape to (batch_size, seq_len, input_dim)
        x = self.self_attention(x)
        x = x.view(batch_size, -1)  # Flatten the output
        x = F.relu(self.fc(x))
        return x


class RecurrentActor(nn.Module):
    def __init__(self, input_channels, conv_out_channels, kernel_size, action_dim, grid_number, num_heads):
        super(RecurrentActor, self).__init__()
        self.conv_with_attention = ConvWithSelfAttention(input_channels, conv_out_channels, kernel_size, grid_number,
                                                         num_heads)
        self.policy_head = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.conv_with_attention(x)
        action_probs = F.softmax(self.policy_head(x), dim=-1)
        return action_probs


class RecurrentCritic(nn.Module):
    def __init__(self, input_channels, conv_out_channels, kernel_size, grid_number, num_heads):
        super(RecurrentCritic, self).__init__()
        self.conv_with_attention = ConvWithSelfAttention(input_channels, conv_out_channels, kernel_size, grid_number,
                                                         num_heads)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_with_attention(x)
        value = self.value_head(x)
        return value


class MAPPO:
    def __init__(self, obs_dim, action_dim, hidden_dim, num_agents, gamma=0.99, clip_epsilon=0.2, lr=3e-4,
                 target_update_freq=200, tau=0.005):
        self.num_agents = num_agents
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.lr = lr
        self.target_update_freq = target_update_freq
        self.tau = tau  # Soft update factor
        self.update_counter = 0
        self.min_rewards = 0
        self.max_rewards = 400  # Clipping range

        self.actors = nn.ModuleList([RecurrentActor(1, 16, 3, action_dim, grid_number=7, num_heads=8)
                                     for _ in range(self.num_agents)])
        self.critics = nn.ModuleList([RecurrentCritic(1, 16, 3, grid_number=8, num_heads=8)
                                      for _ in range(self.num_agents)])

        # Target networks
        self.target_actors = nn.ModuleList([RecurrentActor(1, 16, 3, action_dim, grid_number=7, num_heads=8)
                                            for _ in range(self.num_agents)])
        self.target_critics = nn.ModuleList([RecurrentCritic(1, 16, 3, grid_number=8, num_heads=8)
                                             for _ in range(self.num_agents)])

        # Initialize target networks
        for target_actor, actor in zip(self.target_actors, self.actors):
            target_actor.load_state_dict(actor.state_dict())
        for target_critic, critic in zip(self.target_critics, self.critics):
            target_critic.load_state_dict(critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actors.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=self.lr)

        self.mse_loss = nn.MSELoss()

    def soft_update(self, source, target):
        """Soft update of target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def train(self, states, actions, individual_rewards, global_rewards, next_states, dones, old_log_probs,
              global_states):
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for agent_id in range(self.num_agents):
            # Preprocess data
            states = np.array(states)
            state = torch.tensor(states[:, agent_id, :, :], dtype=torch.float32).unsqueeze(
                1)  # Specific agent's observations
            actions = np.array(actions)
            action = torch.tensor(actions[:, agent_id], dtype=torch.int64).unsqueeze(1)
            individual_rewards = np.array(individual_rewards)
            individual_reward = torch.tensor(individual_rewards[:, agent_id], dtype=torch.float32)
            global_rewards = np.array(global_rewards)
            global_reward = torch.tensor(global_rewards, dtype=torch.float32)
            next_states = np.array(next_states)
            next_state = torch.tensor(next_states, dtype=torch.float32).unsqueeze(1)
            done = torch.tensor(dones, dtype=torch.float32)
            old_log_probs = np.array(old_log_probs)
            old_log_prob = torch.tensor(old_log_probs[:, agent_id], dtype=torch.float32)
            global_states = np.array(global_states)
            global_state = torch.tensor(global_states, dtype=torch.float32).unsqueeze(1)

            # Clip and normalize rewards
            clipped_reward = torch.clamp(individual_reward, self.min_rewards, self.max_rewards)
            normalized_reward = (clipped_reward - self.min_rewards) / (self.max_rewards - self.min_rewards)

            # Actor-critic forward pass
            action_probs = self.actors[agent_id](state)
            values = self.critics[agent_id](global_state)
            action_log_probs = torch.log(action_probs.gather(1, action))

            # Target critic for next state value
            with torch.no_grad():
                next_value = self.target_critics[agent_id](next_state)

            # Compute advantages
            advantages = self.compute_advantage(normalized_reward, values, next_value, done)

            # PPO Loss calculations
            ratio = torch.exp(action_log_probs - old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            target_value = torch.tanh(global_reward).view(-1, 1) + self.gamma * next_value * (1 - done.view(-1, 1))
            value_loss = self.mse_loss(values, target_value)

            entropy_loss = -(action_probs * action_log_probs).sum(dim=-1).mean()

            # Update actor and critic
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Update target networks periodically
            if self.update_counter % self.target_update_freq == 0:
                self.soft_update(self.actors[agent_id], self.target_actors[agent_id])
                self.soft_update(self.critics[agent_id], self.target_critics[agent_id])

        avg_policy_loss = total_policy_loss / self.num_agents
        avg_value_loss = total_value_loss / self.num_agents
        avg_entropy_loss = total_entropy_loss / self.num_agents

        self.update_counter += 1
        return avg_policy_loss, avg_value_loss, avg_entropy_loss

    @staticmethod
    def compute_advantage(reward, values, next_value, done, gamma=0.99, lam=0.95):
        td_target = reward + gamma * next_value * (1 - done)
        delta = td_target - values
        advantage = torch.zeros_like(delta)
        running_add = 0
        for t in reversed(range(len(delta))):
            running_add = delta[t] + gamma * lam * running_add * (1 - done[t])
            advantage[t] = running_add
        return advantage

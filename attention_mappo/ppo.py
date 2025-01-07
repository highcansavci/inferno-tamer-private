import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config.config import Config


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
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=kernel_size, padding=1)
        self.prelu2 = nn.PReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.self_attention = MultiHeadSelfAttention(conv_out_channels, conv_out_channels, num_heads)
        self.prelu3 = nn.PReLU()
        self.fc = nn.Linear((grid_number // 4) ** 2 * conv_out_channels, fc_out_features)

    def forward(self, x):
        x = self.pool(self.prelu1(self.conv1(x)))
        x = self.pool(self.prelu2(self.conv2(x)))
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # Reshape to (batch_size, seq_len, input_dim)
        x = self.self_attention(x)
        x = x.view(batch_size, -1)  # Flatten the output
        x = self.prelu3(self.fc(x))
        return x


class Actor(nn.Module):
    def __init__(self, input_channels, conv_out_channels, kernel_size, action_dim, grid_number, num_heads, comm_grid_number,
                 hidden_size=128):
        super(Actor, self).__init__()

        # Conv + Attention Block
        self.conv_with_attention = ConvWithSelfAttention(input_channels, conv_out_channels, kernel_size, grid_number,
                                                         num_heads)

        self.conv_with_attention_comm = ConvWithSelfAttention(input_channels, conv_out_channels, kernel_size, comm_grid_number,
                                                         num_heads)
        # Policy head to output action probabilities
        self.policy_head = nn.Linear(128 * 2, action_dim)

    def forward(self, x, comm):
        # Apply Conv + Attention layers
        x = self.conv_with_attention(x)
        comm = self.conv_with_attention_comm(comm)
        # Reshape for LSTM
        x = torch.cat((x, comm), dim=1)

        # Apply the policy head to get action probabilities
        action_probs = F.gumbel_softmax(self.policy_head(x), dim=-1)

        return action_probs


class Critic(nn.Module):
    def __init__(self, input_channels, conv_out_channels, kernel_size, grid_number, num_heads, hidden_size=128):
        super(Critic, self).__init__()

        # Conv + Attention Block
        self.conv_with_attention = ConvWithSelfAttention(input_channels, conv_out_channels, kernel_size, grid_number,
                                                         num_heads)
        # Value head to output state value
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # Apply Conv + Attention layers
        x = self.conv_with_attention(x)

        # Apply the value head to get the value
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

        self.actors = nn.ModuleList([Actor(1, 16, 3, action_dim, grid_number=7, num_heads=8, comm_grid_number=Config.GRID_NUMBER)
                                     for _ in range(self.num_agents)])
        self.critics = nn.ModuleList([Critic(1, 16, 3, grid_number=8, num_heads=8)
                                      for _ in range(self.num_agents)])

        # Target networks
        self.target_actors = nn.ModuleList([Actor(1, 16, 3, action_dim, grid_number=7, num_heads=8, comm_grid_number=Config.GRID_NUMBER)
                                            for _ in range(self.num_agents)])
        self.target_critics = nn.ModuleList([Critic(1, 16, 3, grid_number=8, num_heads=8)
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

    @staticmethod
    def generate_action_mask(grid):
        """
        Generate an action mask for the agent based on its observation, prioritizing
        actions deterministically.

        Priorities:
            1. Extinguish adjacent fires.
            2. Move toward refill stacks (4).
            3. Allow valid movement actions (0 or 4).
            4. Noop always valid.

        :param grid: 2D numpy array representing the agent's view.
                     Agent at the center (-2), fires (1-3), other planes (-1),
                     stack refillment (4), and empty cells (0).
        :return: A list of booleans representing valid actions [move, extinguish, noop].
        """
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Up-left, Up, Up-right
            (0, -1), (0, 1),  # Left, Right
            (1, -1), (1, 0), (1, 1)  # Down-left, Down, Down-right
        ]

        agent_refill = -2
        agent_position = np.argwhere(grid == -2)
        if agent_position.size == 0:
            agent_refill = -3
            agent_position = np.argwhere(grid == -3)

        if agent_position.size == 0:
            raise ValueError("Agent not found in the grid.")

        agent_x, agent_y = agent_position[0]

        # Initialize action mask with False (all actions invalid initially)
        action_mask = [0] * 10

        # Step 1: Check for adjacent fires (Extinguish priority)
        extinguishable = False
        for dx, dy in directions:
            adj_x, adj_y = agent_x + dx, agent_y + dy
            if 0 <= adj_x < grid.shape[0] and 0 <= adj_y < grid.shape[1]:  # Check bounds
                if grid[adj_x, adj_y] in [1, 2, 3]:  # Fire present
                    extinguishable = True
                    break
        if extinguishable and agent_refill == -2:
            # Extinguish action (Index 8) and Noop (Index 9) are valid
            action_mask[8] = 1
            action_mask[9] = 1
            return torch.tensor(action_mask, dtype=torch.float32)

        # Step 2: Check for refill stacks (Move toward refill priority)
        refill_positions = []
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = agent_x + dx, agent_y + dy
            if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:  # Check bounds
                if grid[new_x, new_y] == 4:  # Stack refill present
                    refill_positions.append(i)

        if refill_positions and agent_refill == -3:
            # Allow moves toward all visible refill stacks
            for idx in refill_positions:
                action_mask[idx] = 1
            action_mask[9] = 1  # Noop is also valid
            return torch.tensor(action_mask, dtype=torch.float32)

        # Step 3: Validate general movement actions (Empty cells, stacks, and fires)
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = agent_x + dx, agent_y + dy
            if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:  # Check bounds
                cell_value = grid[new_x, new_y]

                # Prioritize moving towards fires
                if cell_value in [1, 2, 3]:  # Fire detected in the adjacent cell
                    action_mask[i] = 1  # Allow movement towards fire

                # Also allow movement to empty cells or stack refill (if no fire is detected)
                elif 0 <= cell_value <= 4:  # Valid move: empty cell or stack refill
                    if action_mask[i] == 0:  # If no action has been assigned (no fire), then allow other valid moves
                        action_mask[i] = 1

        # Step 4: Noop action (Index 9) is always valid
        action_mask[9] = 1

        return torch.tensor(action_mask, dtype=torch.float32)

    def train(self, states, actions, individual_rewards, global_rewards, next_states, dones, old_log_probs,
              global_states, communications):
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
            communications = np.array(communications)
            communication = torch.tensor(communications, dtype=torch.float32).unsqueeze(1)

            # Clip and normalize rewards
            clipped_reward = torch.clamp(individual_reward, self.min_rewards, self.max_rewards)
            normalized_reward = (clipped_reward - self.min_rewards) / (self.max_rewards - self.min_rewards)

            # Actor-critic forward pass
            action_probs = self.actors[agent_id](state, global_state)
            values = self.critics[agent_id](global_state)
            action_log_probs = torch.log(action_probs.gather(1, action))

            # Target critic for next state value
            with torch.no_grad():
                next_value = self.target_critics[agent_id](next_state)[0]

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
        td_target = reward.view(-1, 1) + gamma * next_value * (1 - done.view(-1, 1))
        delta = td_target - values[0]
        advantage = torch.zeros_like(delta)
        running_add = 0
        for t in reversed(range(len(delta))):
            running_add = delta[t] + gamma * lam * running_add * (1 - done[t])
            advantage[t] = running_add
        return advantage

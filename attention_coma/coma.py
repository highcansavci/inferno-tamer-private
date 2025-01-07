import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch.optim import Adam

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

        # Compute queries, keys, and values
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(weights, V)

        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.fc_out(attended_values)


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
    def __init__(self, input_channels, conv_out_channels, kernel_size, action_dim, grid_number, num_heads, comm_grid_number, hidden_size=128):
        super(Actor, self).__init__()
        self.conv_with_attention = ConvWithSelfAttention(input_channels, conv_out_channels, kernel_size, grid_number,
                                                         num_heads)
        self.conv_with_attention_comm = ConvWithSelfAttention(input_channels, conv_out_channels, kernel_size,
                                                              comm_grid_number,
                                                              num_heads)
        self.prelu = nn.PReLU()
        # LSTM layer to introduce recurrence
        self.policy_head = nn.Linear(256, action_dim)

    def forward(self, x, comm):
        x = self.conv_with_attention(x)
        comm = self.conv_with_attention_comm(comm)
        x = torch.cat((x, comm), dim=1)
        logits = self.prelu(self.policy_head(x))
        return logits


class Critic(nn.Module):
    def __init__(self, input_channels, conv_out_channels, kernel_size, grid_number, hidden_dim, action_dim, num_agents, num_heads):
        super(Critic, self).__init__()
        self.conv_with_attention = ConvWithSelfAttention(input_channels, conv_out_channels, kernel_size, grid_number,
                                                         num_heads)
        self.prelu = nn.PReLU()
        self.fc1 = nn.Linear(128 + action_dim * num_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions, hidden_state=None):
        x = self.conv_with_attention(states)
        x = torch.cat([x, actions.view(actions.size(0), -1)], dim=-1)
        x = self.prelu(self.fc1(x))
        value = self.fc2(x)
        return value


class MultiAgentCOMA:
    def __init__(self, state_dim, action_dim, num_agents, hidden_dim=128, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.gamma = gamma

        # Initialize actor and critic for each agent
        self.actors = nn.ModuleList([Actor(1, 16, 3, action_dim, grid_number=13, num_heads=8, comm_grid_number=Config.GRID_NUMBER) for _ in range(num_agents)])
        self.critic = Critic(1, 16, 3, grid_number=8, num_heads=8, hidden_dim=hidden_dim, action_dim=Config.NUM_ACTIONS, num_agents=Config.NUM_OF_PLANES)

        # Optimizers
        self.actor_optimizers = [Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

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
        action_mask = [0] * Config.NUM_ACTIONS

        # Step 1: Check for adjacent fires (Extinguish priority)
        extinguishable = False
        for dx, dy in directions:
            adj_x, adj_y = agent_x + dx, agent_y + dy
            if 0 <= adj_x < grid.shape[0] and 0 <= adj_y < grid.shape[1]:  # Check bounds
                if grid[adj_x, adj_y] in [1, 2, 3]:  # Fire present
                    extinguishable = True
                    break
        if extinguishable and agent_refill == -2:
            # Extinguish action (Index 8)
            action_mask[8] = 1
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
            return torch.tensor(action_mask, dtype=torch.float32)

        # Step 3: Move toward visible fires (not just adjacent ones)
        visible_fires = []
        for i, (dx, dy) in enumerate(directions):
            for step in range(1, max(grid.shape)):  # Check progressively outward in this direction
                new_x, new_y = agent_x + dx * step, agent_y + dy * step
                if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:  # Check bounds
                    cell_value = grid[new_x, new_y]
                    if cell_value in [1, 2, 3]:  # Fire detected
                        visible_fires.append((i, step))  # Store direction and distance
                        break  # Stop looking further in this direction
                    elif cell_value not in [0, 4]:  # Obstacle blocks visibility
                        break

        # If visible fires exist, move toward the closest one
        if visible_fires and agent_refill == -2:
            closest_fire = min(visible_fires, key=lambda x: x[1])  # Find the closest fire by distance
            action_mask[closest_fire[0]] = 1  # Mark the direction toward the closest fire as valid
        else:
            # Allow movement to empty cells or stack refills if no fire is visible
            for i, (dx, dy) in enumerate(directions):
                new_x, new_y = agent_x + dx, agent_y + dy
                if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:  # Check bounds
                    cell_value = grid[new_x, new_y]
                    if cell_value in [0, 4]:  # Valid moves: empty cell or stack refill
                        action_mask[i] = 1

        return torch.tensor(action_mask, dtype=torch.float32)

    def select_action(self, local_state, comm, agent_id, action_mask=None):
        """
        Select an action using the actor network for the given agent, with improved action masking.

        :param comm: Communication matrix.
        :param local_state: Local state of the agent (Tensor).
        :param agent_id: ID of the agent.
        :param action_mask: Optional action mask (1 for valid, 0 for invalid).
        :return: Action and its log probability.
        """
        logits = self.actors[agent_id](local_state.unsqueeze(0), comm.unsqueeze(0))  # Pass through the actor network

        if action_mask is not None:
            # Ensure the mask matches the shape of logits
            action_mask = action_mask.unsqueeze(0)  # Add batch dimension to match logits shape

            # Convert action_mask to boolean
            valid_mask = action_mask.bool()

            # Use logits only for valid actions; mask out invalid actions
            logits[~valid_mask] = float('-inf')

        # Create a Categorical distribution over the valid logits
        valid_logits = logits - logits.logsumexp(dim=-1, keepdim=True)  # Re-normalize to avoid NaN issues
        dist = Categorical(logits=valid_logits)

        # Sample action and compute log-probabilities
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def train(self, global_states, communications, local_states, actions, rewards, next_global_states, dones):
        """
        Train the COMA model.

        :param communications: Communication matrix.
        :param global_states: Global states (shared among all agents).
        :param local_states: Local states for each agent.
        :param actions: Actions taken by agents.
        :param rewards: Reward for each agent.
        :param next_global_states: Next global states.
        :param dones: Done flags.
        :return: Actor loss, Critic loss.
        """
        # Convert inputs to tensors
        global_states = torch.tensor(global_states, dtype=torch.float32)
        local_states = torch.tensor(local_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_global_states = torch.tensor(next_global_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)
        communications = torch.tensor(communications, dtype=torch.float32)

        # Centralized critic: Compute Q-values
        with torch.no_grad():
            next_actions = torch.stack([self.actors[agent](local_states[:, agent, :, :].unsqueeze(1), global_states.unsqueeze(1)).argmax(dim=-1)
                                        for agent in range(self.num_agents)], dim=1)
            one_hot_next_actions = F.one_hot(next_actions, num_classes=self.action_dim).float()
            target_q_values = self.critic(next_global_states.unsqueeze(1), one_hot_next_actions)
            target_values = rewards.view(-1, 1) + self.gamma * (1 - dones) * target_q_values

        # Compute current Q-values and advantage
        one_hot_actions = F.one_hot(actions, num_classes=self.action_dim).float()
        q_values = self.critic(global_states.unsqueeze(1), one_hot_actions)
        critic_loss = F.mse_loss(q_values, target_values)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train actor for each agent
        total_actor_loss = 0
        for agent_id in range(self.num_agents):
            # Compute counterfactual baseline
            q_values_no_action = []
            for alt_action in range(self.action_dim):
                alt_actions = actions.clone()
                alt_actions[:, agent_id] = alt_action
                one_hot_alt_actions = F.one_hot(alt_actions, num_classes=self.action_dim).float()
                q_values_no_action.append(self.critic(global_states.unsqueeze(1), one_hot_alt_actions))
            q_values_no_action = torch.stack(q_values_no_action, dim=1)
            counterfactual_baseline = (q_values_no_action * F.softmax(q_values_no_action, dim=1)).sum(dim=1, keepdim=True)

            # Compute actor loss
            logits = self.actors[agent_id](local_states[:, agent_id, :, :].unsqueeze(1), global_states.unsqueeze(1))
            dist = Categorical(logits=F.log_softmax(logits, dim=-1))
            log_probs = dist.log_prob(actions[:, agent_id])
            advantages = (q_values - counterfactual_baseline.squeeze(-1)).squeeze(-1)
            actor_loss = -(log_probs * advantages.detach()).mean()

            # Update actor
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_id].step()

            total_actor_loss += actor_loss.item()

        return total_actor_loss / self.num_agents, critic_loss.item()

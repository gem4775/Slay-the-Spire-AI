import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from GameLogic import GameLogic
from EnemyFactory import EnemyFactory

MAX_MONSTERS = 5
CARD_LENGTH = len(GameLogic.encode_card({}))
PLAYER_LENGTH = 3
ENEMY_FEATURES = len(GameLogic.encode_enemy(EnemyFactory.dummy_monster()))

# State: player + (MAX_MONSTERS * ENEMY_FEATURES) + (10 cards * CARD_LENGTH)
DIMENSIONS = PLAYER_LENGTH + MAX_MONSTERS * ENEMY_FEATURES + CARD_LENGTH * 10

#10 card hand * # of targets + self/AoE + end turn
ACTION_DIM = 10 * MAX_MONSTERS + 11  # = 61 for MAX_MONSTERS=5

# 1. Neural Network
class DQN(nn.Module):
    def __init__(self, state_dim=36, action_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# 2. Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# 3. Agent
class DQNAgent:
    def __init__(self, state_dim=36, action_dim=11, lr=0.001):
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.batch_size = 256

    def select_action(self, state, legal_actions):
        """
        state: numpy array (36)
        legal_actions: list of valid action indices [0,1,2,...,10]
        """
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze()

            # Mask illegal actions
            mask = torch.full_like(q_values, float('-inf'))
            mask[legal_actions] = q_values[legal_actions]

            return mask.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Loss and backprop
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 4. Training Loop Skeleton
def train():

    agent = DQNAgent(state_dim=DIMENSIONS, action_dim=ACTION_DIM)
    recent_rewards = deque(maxlen=100)

    for episode in range(3000):
        player_state = {"player_hp": 50, "block": 0, "energy": 3}

        cards = [{'name': 'Strike_R'}, {'name': 'Strike_R'}, {'name': 'Strike_R'}, {'name': 'Strike_R'},{'name': 'Strike_R'},
                 {'name': 'Defend_R'}, {'name': 'Defend_R'}, {'name': 'Defend_R'}, {'name': 'Defend_R'},{'name': 'Bash'},
                 {'name': 'Iron_Wave'}]

        hand = random.sample(cards, 5)
        draw = random.sample(cards, 6)
        vector = {
            "player_hp": 50,
            "block": 0,
            "energy": 3,
            "hand": hand,
            "draw_pile": draw,
            "discard_pile": []
        }

        if random.choice([True, False]):
            enemies = [EnemyFactory.create_jaw_worm()]
        else:
            enemies = [EnemyFactory.create_cultist()]

        state = GameLogic.encode_state(player_state, enemies, hand)

        total_reward = 0
        done = False

        while not done:
            legal_actions = GameLogic.get_legal_actions(vector, enemies)
            action = agent.select_action(state, legal_actions)

            next_vector, next_enemies, reward, done = GameLogic.step(vector, enemies, action)
            next_state = GameLogic.encode_state(next_vector, next_enemies, next_vector['hand'])

            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            vector = next_vector
            enemies = next_enemies
            total_reward += reward

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if episode % 20 == 0:
            agent.update_target_network()

        recent_rewards.append(total_reward)

        if episode % 100 == 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            print(f"Episode {episode}, Last reward: {total_reward:.2f}, Avg last 100: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.policy_net.state_dict(), "trained_model.pth")
    print("Model saved!")


def _describe_action(action, vector, enemies):
    """Return a human-readable string for a given action index."""
    action_type, card_idx, enemy_idx = GameLogic.decode_action(action, MAX_MONSTERS)
    end_turn_action = 10 * MAX_MONSTERS + 10

    if action_type == 'end_turn':
        return f"END TURN (action {action})"
    elif action_type == 'single':
        card_name = vector['hand'][card_idx]['name'] if card_idx < len(vector['hand']) else '?'
        target_name = enemies[enemy_idx]['name'] if enemy_idx < len(enemies) else '?'
        return f"Play {card_name} → {target_name} [card={card_idx}, enemy={enemy_idx}] (action {action})"
    else:
        card_name = vector['hand'][card_idx]['name'] if card_idx < len(vector['hand']) else '?'
        return f"Play {card_name} (AoE/self) [card={card_idx}] (action {action})"


def play_trained_agent(Think=False):
    agent = DQNAgent(state_dim=DIMENSIONS, action_dim=ACTION_DIM)
    agent.policy_net.load_state_dict(torch.load("trained_model.pth"))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    player_state = {"player_hp": 50, "block": 0, "energy": 3}

    cards = [{'name': 'Strike_R'}, {'name': 'Strike_R'}, {'name': 'Strike_R'},
             {'name': 'Strike_R'}, {'name': 'Strike_R'},
             {'name': 'Defend_R'}, {'name': 'Defend_R'},
             {'name': 'Defend_R'}, {'name': 'Defend_R'},
             {'name': 'Bash'},
             {'name': 'Iron_Wave'}]

    hand = random.sample(cards, 5)
    remaining = [c for c in cards if c not in hand]

    vector = {
        "player_hp": 50,
        "block": 0,
        "energy": 3,
        "hand": hand,
        "draw_pile": remaining,
        "discard_pile": []
    }

    enemies = [EnemyFactory.create_cultist()]
    state = GameLogic.encode_state(player_state, enemies, hand)

    done = False
    step_num = 0

    if Think:
        print("=" * 60)
        print("GAME START")
        print("=" * 60)
        print(f"Player HP: {vector['player_hp']}")
        for i, e in enumerate(enemies):
            print(f"Enemy {i} ({e['name']}): HP={e['hp']}")
        print(f"Starting hand: {[c['name'] for c in vector['hand']]}")
        print()

    while not done:
        if Think:
            print(f"\n--- Step {step_num} ---")
            print(f"Player: HP={vector['player_hp']}, Block={vector['block']}, Energy={vector['energy']}")
            for i, e in enumerate(enemies):
                print(f"Enemy {i} ({e['name']}): HP={e['hp']}, Intent={e.get('intent')}, Damage={e.get('intentDamage', 0)}")
            print(f"Hand: {[c['name'] for c in vector['hand']]}")

        legal_actions = GameLogic.get_legal_actions(vector, enemies)

        if Think:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.policy_net(state_t).squeeze().numpy()

            print("Q-values for legal actions:")
            for a in legal_actions:
                desc = _describe_action(a, vector, enemies)
                print(f"  {desc}: {q_values[a]:.3f}")
            best = legal_actions[np.argmax(q_values[legal_actions])]
            print(f"  → Best: {q_values[best]:.3f}")

        action = agent.select_action(state, legal_actions)

        if Think:
            print(f">>> Agent chose: {_describe_action(action, vector, enemies)}")
            action_type, card_idx, enemy_idx = GameLogic.decode_action(action, MAX_MONSTERS)
            if action_type in ('single', 'multi') and card_idx < len(vector['hand']):
                card_data = GameLogic.vectorize_card(vector['hand'][card_idx]['name'])
                print(f"    Cost: {card_data['cost']}, Damage: {card_data['damage']}, Block: {card_data['block']}")

        next_vector, next_enemies, reward, done = GameLogic.step(vector, enemies, action)
        next_state = GameLogic.encode_state(next_vector, next_enemies, next_vector['hand'])

        if Think:
            print(f"Reward: {reward:.2f}")

        state = next_state
        vector = next_vector
        enemies = next_enemies
        step_num += 1

    if Think:
        print("\n" + "=" * 60)
        if vector['player_hp'] > 0:
            print("AGENT WON!")
        else:
            print("AGENT LOST!")
        print(f"Final Player HP: {vector['player_hp']}")
        for i, e in enumerate(enemies):
            print(f"Final Enemy {i} ({e['name']}) HP: {e['hp']}")
        print("=" * 60)

    return vector['player_hp']


if __name__ == "__main__":
    train()
    play_trained_agent(Think=True)
    # hp = []
    # for i in range(10):
    #     hp.append(50 - play_trained_agent())
    # print(hp)
    # print(sum(hp)/len(hp))
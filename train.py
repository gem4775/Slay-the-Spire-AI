import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from numpy.ma.extras import average
import ironclad_dict
from GameLogic import GameLogic
from EnemyFactory import EnemyFactory

MAX_MONSTERS = 5
CARD_LENGTH = len(GameLogic.encode_card({}))
PLAYER_LENGTH = 3
ENEMY_FEATURES = len(GameLogic.encode_enemy(EnemyFactory.dummy_monster()))

# State: player + (MAX_MONSTERS * ENEMY_FEATURES) + (10 cards * CARD_LENGTH)
DIMENSIONS = PLAYER_LENGTH + MAX_MONSTERS * ENEMY_FEATURES + CARD_LENGTH * 10

# 10 card hand * # of targets + self/AoE + end turn
ACTION_DIM = 10 * MAX_MONSTERS + 11  # = 61 for MAX_MONSTERS=5

ENCOUNTERS = [
    (29, lambda: [EnemyFactory.create_jaw_worm()]),
    (28, lambda: [EnemyFactory.create_cultist()]),
    (28, lambda: [EnemyFactory.create('Louse'), EnemyFactory.create('Louse')]),
    (15, lambda: [EnemyFactory.create("Gremlin Nob")])
]

def sample_encounter():
    weights = [w for w, _ in ENCOUNTERS]
    fns     = [fn for _, fn in ENCOUNTERS]
    return random.choices(fns, weights=weights, k=1)[0]()

def shuffle_enemies(enemies):
    """
    Pad enemies list to MAX_MONSTERS with dead placeholders, then shuffle.
    Forces the agent to learn targeting by slot position rather than always
    assuming slot 0 is the only live enemy.
    """
    dead = EnemyFactory.dummy_monster()  # hp=0, all zeros when encoded
    padded = enemies + [dead] * (MAX_MONSTERS - len(enemies))
    random.shuffle(padded)
    return padded

def get_early_deck():
    starter_deck = [
        {'name': 'Strike_R'}, {'name': 'Strike_R'}, {'name': 'Strike_R'},
        {'name': 'Strike_R'}, {'name': 'Strike_R'},
        {'name': 'Defend_R'}, {'name': 'Defend_R'}, {'name': 'Defend_R'},
        {'name': 'Defend_R'}, {'name': 'Bash'}
    ]
    eligible_keys = list(ironclad_dict.CARD_LIBRARY.keys())[3:]
    num_cards = random.randint(1, 3)
    selected_keys = random.sample(eligible_keys, min(num_cards, len(eligible_keys)))
    starter_deck.extend([{'name': key} for key in selected_keys])
    return starter_deck


# 1. Neural Network
class DQN(nn.Module):
    def __init__(self, state_dim=36, action_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# 2. Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, is_aoe=False):
        if is_aoe:
            for _ in range(4):
                self.buffer.append((state, action, reward, next_state, done))
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
    def __init__(self, state_dim=36, action_dim=11, lr=0.0001):
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_decay = 0.9998
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.batch_size = 256

    def select_action(self, state, legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze()
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

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.SmoothL1Loss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 4. Training Loop
def train():
    agent = DQNAgent(state_dim=DIMENSIONS, action_dim=ACTION_DIM)
    recent_rewards = deque(maxlen=100)
    recent_wins = deque(maxlen=100)
    recent_nob_results = deque(maxlen=100)  # only populated on Nob fights
    total_wins = 0

    for episode in range(10000):
        cards = get_early_deck()
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

        enemies = sample_encounter()
        is_nob = any(e['name'] == 'Gremlin Nob' for e in enemies)
        enemies = shuffle_enemies(enemies)

        state = GameLogic.encode_state(vector, enemies, hand)

        total_reward = 0
        done = False

        while not done:
            legal_actions = GameLogic.get_legal_actions(vector, enemies)
            action = agent.select_action(state, legal_actions)

            next_vector, next_enemies, reward, done = GameLogic.step(vector, enemies, action)
            next_state = GameLogic.encode_state(next_vector, next_enemies, next_vector['hand'])

            action_type, _, _ = GameLogic.decode_action(action, MAX_MONSTERS)
            agent.memory.push(state, action, reward, next_state, done, is_aoe=(action_type == 'multi'))
            agent.train_step()

            state = next_state
            vector = next_vector
            enemies = next_enemies
            total_reward += reward

        # Update epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Track results
        won = vector['player_hp'] > 0
        if won:
            total_wins += 1
        recent_wins.append(1 if won else 0)
        recent_rewards.append(total_reward)

        if is_nob:
            recent_nob_results.append(1 if won else 0)

        # Target network update
        if episode % 300 == 0:
            agent.update_target_network()

        # Checkpoint
        if episode % 1000 == 0 and episode > 0:
            torch.save(agent.policy_net.state_dict(), f"checkpoint_{episode}.pth")
            print(f"  >>> Checkpoint saved: checkpoint_{episode}.pth")

        # Logging
        if episode % 100 == 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            win_rate = sum(recent_wins) / len(recent_wins)
            nob_rate = sum(recent_nob_results) / len(recent_nob_results) if recent_nob_results else 0.0
            overall_win_rate = total_wins / (episode + 1)

            print(
                f"Ep {episode:5d} | "
                f"Avg Reward: {avg_reward:7.2f} | "
                f"Win Rate (100): {win_rate:.2%} | "
                f"Nob Win Rate (100): {nob_rate:.2%} | "
                f"Overall Win: {overall_win_rate:.2%} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    torch.save(agent.policy_net.state_dict(), "trained_model.pth")
    print("Model saved!")


def _describe_action(action, vector, enemies):
    action_type, card_idx, enemy_idx = GameLogic.decode_action(action, MAX_MONSTERS)

    if action_type == 'end_turn':
        return f"END TURN (action {action})"
    elif action_type == 'single':
        card_name = vector['hand'][card_idx]['name'] if card_idx < len(vector['hand']) else '?'
        target_name = enemies[enemy_idx]['name'] if enemy_idx < len(enemies) else '?'
        return f"Play {card_name} → {target_name} [card={card_idx}, enemy={enemy_idx}] (action {action})"
    else:
        card_name = vector['hand'][card_idx]['name'] if card_idx < len(vector['hand']) else '?'
        return f"Play {card_name} (AoE/self) [card={card_idx}] (action {action})"


def play_trained_agent(think=False, agent=None):
    if agent is None:
        agent = DQNAgent(state_dim=DIMENSIONS, action_dim=ACTION_DIM)
        agent.policy_net.load_state_dict(torch.load("trained_model.pth"))
        agent.policy_net.eval()
        agent.epsilon = 0.0

    cards = get_early_deck()
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

    enemies = sample_encounter()
    enemies = shuffle_enemies(enemies)

    # Fix: encode state using vector, not a separate player_state
    state = GameLogic.encode_state(vector, enemies, hand)

    done = False
    step_num = 0

    if think:
        print("=" * 60)
        print("GAME START")
        print("=" * 60)
        print(f"Player HP: {vector['player_hp']}")
        for i, e in enumerate(enemies):
            print(f"Enemy {i} ({e['name']}): HP={e['hp']}")
        print(f"Starting hand: {[c['name'] for c in vector['hand']]}")
        print()

    while not done:
        if think:
            print(f"\n--- Step {step_num} ---")
            print(f"Player: HP={vector['player_hp']}, Block={vector['block']}, Energy={vector['energy']}")
            for i, e in enumerate(enemies):
                if e['name'] != 'ERR':
                    print(f"Enemy {i} ({e['name']}): HP={e['hp']}, Intent={e.get('intent')}, Damage={e.get('intentDamage', 0)}")
            print(f"Hand: {[c['name'] for c in vector['hand']]}")

        legal_actions = GameLogic.get_legal_actions(vector, enemies)

        if think:
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

        if think:
            print(f">>> Agent chose: {_describe_action(action, vector, enemies)}")
            action_type, card_idx, enemy_idx = GameLogic.decode_action(action, MAX_MONSTERS)
            if action_type in ('single', 'multi') and card_idx < len(vector['hand']):
                card_data = GameLogic.vectorize_card(vector['hand'][card_idx]['name'])
                print(f"    Cost: {card_data['cost']}, Damage: {card_data['damage']}, Block: {card_data['block']}")

        next_vector, next_enemies, reward, done = GameLogic.step(vector, enemies, action)
        next_state = GameLogic.encode_state(next_vector, next_enemies, next_vector['hand'])

        if think:
            print(f"Reward: {reward:.2f}")

        state = next_state
        vector = next_vector
        enemies = next_enemies
        step_num += 1

    if think:
        print("\n" + "=" * 60)
        if vector['player_hp'] > 0:
            print("AGENT WON!")
        else:
            print("AGENT LOST!")
        print(f"Final Player HP: {vector['player_hp']}")
        for i, e in enumerate(enemies):
            if e['name'] != 'ERR':
                print(f"Final Enemy {i} ({e['name']}) HP: {e['hp']}")
        print("=" * 60)

    return vector['player_hp']

def eval_all_checkpoints(num_rounds=100):
    import os
    checkpoints = ["checkpoint_1000.pth", "checkpoint_2000.pth", "checkpoint_3000.pth",
                   "checkpoint_4000.pth", "checkpoint_5000.pth", "checkpoint_6000.pth",
                   "checkpoint_7000.pth", "checkpoint_8000.pth", "checkpoint_9000.pth",
                   "trained_model.pth"]

    agent = DQNAgent(state_dim=DIMENSIONS, action_dim=ACTION_DIM)
    agent.epsilon = 0.0

    print(f"\n{'Checkpoint':<25} {'Win Rate':<15} {'Avg Damage Taken'}")
    print("-" * 55)

    for ckpt in checkpoints:
        if not os.path.exists(ckpt):
            print(f"{ckpt:<25} not found, skipping")
            continue

        agent.policy_net.load_state_dict(torch.load(ckpt))
        agent.policy_net.eval()

        hp_results = []
        for _ in range(num_rounds):
            hp_results.append(play_trained_agent(think=False, agent=agent))

        wins = sum(1 for h in hp_results if h > 0)
        avg_damage = 50 - average(hp_results)
        print(f"{ckpt:<25} {wins}/{num_rounds} ({wins/num_rounds:.2%})    {avg_damage:.2f}")



if __name__ == "__main__":
    #train()
    # Backup trained model before eval
    import shutil
    shutil.copy("trained_model.pth", "trained_model_backup.pth")

    #play_trained_agent(Think=True)
    eval_all_checkpoints(100)
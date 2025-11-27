import json
import os
import ironclad_dict
import copy
import numpy as np
from sklearn.linear_model import LinearRegression
import simulator
import random

SAVE_DIR = "received_json"

class BaseCase:
    @staticmethod
    def vectorize_state(filename):
        path = os.path.join(SAVE_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # direct fields (your JSON is flat)
        player_hp = data.get("player_hp", 0)
        block = data.get("block", 0)
        energy = data.get("energy", 0)

        # hand: list of dicts with "id" + "upgraded"
        raw_hand = data.get("hand", [])
        hand = [
            {
                "name": card.get("id"),
                "upgraded": card.get("upgraded", False)
            }
            for card in raw_hand
        ]

        enemies = data.get("enemies", [])
        for i in range(len(enemies)):
            enemies[i].update({'pos': i})
        vector = {
            "player_hp": player_hp,
            "block": block,
            "energy": energy,
            "hand": hand,
        }

        return vector, enemies

    @staticmethod
    def vectorize_card(cardName):
        delta = copy.deepcopy(ironclad_dict.CARD_LIBRARY[cardName])
        return delta

    @staticmethod
    def apply_card_player(player_state, vector):
        player_state['energy'] = player_state['energy'] - vector['cost']
        player_state['block'] = player_state['block'] + vector['block']
        player_state['player_hp'] = player_state['player_hp'] + vector['self_hp_change']
        return player_state

    @staticmethod
    def apply_card_monster(player_state, monster_state, vector):
        monster_state['hp'] = monster_state['hp'] - vector['damage']
        #TODO Special logic
        return monster_state

    @staticmethod
    def monster_action(enemies, player_state):
        for e in enemies:
            if e['intent'] == 'ATTACK':
                new_block = player_state['block'] - e['intentDamage'] * e['intentHits']
                if new_block > 0:
                    player_state['block'] = new_block
                else:
                    player_state['block'] = 0
                    player_state['player_hp'] = player_state['player_hp'] + new_block
            #TODO BUFFS and DEBUFFS

        return player_state




def compute_reward(player_before, player_after, enemies_before, enemies_after):
    reward = 0.0

    # 1️⃣ Player HP change
    delta_hp = player_after['player_hp'] - player_before['player_hp']
    reward += 2.0 * delta_hp

    # 2️⃣ Effective block (only mitigates incoming damage)
    incoming_damage = 0
    for e in enemies_before:
        if e['hp'] <= 0:
            continue
        if e['intent'] == 'ATTACK':
            incoming_damage += e['intentDamage'] * e['intentHits']

    delta_block = player_after['block'] - player_before['block']
    effective_block = min(delta_block, incoming_damage)
    reward += 1.5 * effective_block

    # 3️⃣ Damage to alive enemies
    delta_enemy_hp = 0
    for before, after in zip(enemies_before, enemies_after):
        if before['hp'] <= 0:
            continue
        hp_before = before['hp']
        hp_after = max(0, after['hp'])
        delta_enemy_hp += hp_before - hp_after

    reward += 2.0 * delta_enemy_hp

    # 4️⃣ Small bonus for leftover energy
    delta_energy = player_after['energy'] - player_before['energy']
    reward += 0.1 * delta_energy

    return reward



training_data = simulator.generate_training_data(num_turns=200)

# ----------------------------
# 1️⃣ Extract features & rewards
# ----------------------------
features = []
rewards = []

for turn in training_data:
    player_before = turn['player_before']
    player_after = turn['player_after']
    enemies_before = turn['enemies_before']
    enemies_after = turn['enemies_after']

    # Features: deltas for the model
    delta_player_hp = player_after['player_hp'] - player_before['player_hp']
    delta_block = player_after['block'] - player_before['block']
    delta_energy = player_after['energy'] - player_before['energy']
    delta_enemy_hp = sum(e['hp'] for e in enemies_before) - sum(e['hp'] for e in enemies_after)

    feat_vector = [delta_player_hp, delta_block, delta_energy, delta_enemy_hp]
    features.append(feat_vector)

    # Reward based on your heuristic
    reward = compute_reward(player_before, player_after, enemies_before, enemies_after)
    rewards.append(reward)

X = np.array(features)
y = np.array(rewards)

# ----------------------------
# 2️⃣ Train linear regression model
# ----------------------------
model = LinearRegression()
model.fit(X, y)

print("Learned weights:", model.coef_)
print("Intercept:", model.intercept_)

# ----------------------------
# 3️⃣ Optional: function to predict reward for a given card play
# ----------------------------
def predict_card_reward(player_before, player_after, enemies_before, enemies_after):
    delta_player_hp = player_after['player_hp'] - player_before['player_hp']
    delta_block = player_after['block'] - player_before['block']
    delta_energy = player_after['energy'] - player_before['energy']
    delta_enemy_hp = sum(e['hp'] for e in enemies_before) - sum(e['hp'] for e in enemies_after)
    feature_vector = np.array([[delta_player_hp, delta_block, delta_energy, delta_enemy_hp]])
    return model.predict(feature_vector)[0]

# Example usage:
# reward_estimate = predict_card_reward(player_before, player_after, enemies_before, enemies_after)

# ----------------------------
# 4️⃣ Feature vector helper
# ----------------------------
def feature_vector_from_states(player_before, player_after, enemies_before, enemies_after):
    delta_player_hp = player_after['player_hp'] - player_before['player_hp']
    delta_block = player_after['block'] - player_before['block']
    delta_energy = player_after['energy'] - player_before['energy']
    delta_enemy_hp = sum(e['hp'] for e in enemies_before) - sum(e['hp'] for e in enemies_after)
    return np.array([[delta_player_hp, delta_block, delta_energy, delta_enemy_hp]])


# ----------------------------
# 5️⃣ Choose best sequence using trained model
# ----------------------------
def choose_best_sequence(player_state, enemies, deck, model):
    # Use simulator's full turn sequences
    turn_examples = simulator.simulate_full_turn(player_state, enemies, deck)
    best_reward = -float('inf')
    best_turn = None

    for turn in turn_examples:
        fv = feature_vector_from_states(turn['player_before'], turn['player_after'],
                                        turn['enemies_before'], turn['enemies_after'])
        reward = model.predict(fv)[0]
        if reward > best_reward:
            best_reward = reward
            best_turn = turn

    return best_turn, best_reward

deck = list(simulator.CARD_LIBRARY.keys())
player_state = {"player_hp": 50, "block": 0, "energy": 3}
enemies = [copy.deepcopy(random.choice(simulator.ENEMY_LIBRARY)) for _ in range(random.randint(1,2))]

best_turn, predicted_reward = choose_best_sequence(player_state, enemies, deck, model)




def choose_best_sequence(player_state, enemies, deck, model):
    # Simulate all possible card sequences for this hand
    turn_examples = simulator.simulate_full_turn(player_state, enemies, deck)

    best_reward = -float('inf')
    best_turn = None

    for turn in turn_examples:
        pred_reward = predict_card_reward(
            turn['player_before'],
            turn['player_after'],
            turn['enemies_before'],
            turn['enemies_after']
        )
        if pred_reward > best_reward:
            best_reward = pred_reward
            best_turn = turn

    return best_turn, best_reward


# Deck list
deck = list(simulator.CARD_LIBRARY.keys())

# Iterate through all JSON files
for filename in os.listdir(SAVE_DIR):
    if not filename.endswith(".json"):
        continue

    # Pass only the filename; BaseCase will prepend SAVE_DIR
    player_state, enemies = BaseCase.vectorize_state(filename)
    deck = [card['name'] for card in player_state['hand']]

    # Simulate all sequences of this hand
    turn_examples = simulator.simulate_full_turn(player_state, enemies, deck)

    # Find sequence with highest predicted reward
    best_reward = float("-inf")
    best_sequence = None
    for turn in turn_examples:
        reward_est = predict_card_reward(
            turn["player_before"], turn["player_after"],
            turn["enemies_before"], turn["enemies_after"]
        )
        if reward_est > best_reward:
            best_reward = reward_est
            best_sequence = turn["played_cards"]

    print(f"{filename}: Best sequence -> {best_sequence} (predicted reward {best_reward:.2f})")
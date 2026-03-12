import json
import os
import ironclad_dict
import numpy as np
import copy
from EnemyFactory import EnemyFactory

SAVE_DIR = "received_json"

class GameLogic:
    @staticmethod
    def vectorize_state(filename):
        path = os.path.join(SAVE_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        player_hp = data.get("player_hp", 0)
        block = data.get("block", 0)
        energy = data.get("energy", 0)

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
        draw_pile = data.get("draw_pile", [])
        vector = {
            "player_hp": player_hp,
            "block": block,
            "energy": energy,
            "hand": hand,
            "draw_pile": draw_pile
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
        # Calculate damage first, THEN apply vulnerable modifier
        damage = vector['damage']
        if monster_state.get('vulnerable', 0) > 0:
            damage = int(damage * 1.5)

        # Apply vulnerable debuff from the card
        if vector.get('vulnerable', 0) > 0:
            monster_state['vulnerable'] = monster_state.get('vulnerable', 0) + vector['vulnerable']
            monster_state['is_vulnerable'] = 1
        if vector.get('weak', 0) > 0:
            monster_state['weak'] = monster_state.get('weak', 0) + vector['weak']
            if monster_state['is_weak'] == 0:
                monster_state["intentDamage"] = int(monster_state.get("intentDamage", 0) * .75)
            monster_state['is_weak'] = 1

        monster_state['block'] = monster_state.get('block', 0) - damage
        if monster_state['block'] < 0:
            damage = abs(monster_state['block'])
            monster_state['block'] = 0
        else:
            damage = 0
        monster_state['hp'] = monster_state['hp'] - damage
        monster_state['hp'] = max(0, monster_state['hp'])

        # Curl: taking an attack triggers block gain (e.g. Red Louse)
        if vector.get('type') == 0 and monster_state.get('curl', 0) > 0:
            monster_state['block'] = monster_state['curl']
            monster_state['curl'] = 0

        return monster_state

    @staticmethod
    def monster_action(enemies, player_state):
        for e in enemies:
            r_flag = e.get('ritual')
            if e['hp'] <= 0:
                continue

            if e['intent'] == 'ATTACK':
                total_damage = e['intentDamage'] * e['intentHits']

                if player_state['block'] >= total_damage:
                    player_state['block'] -= total_damage
                else:
                    remaining_damage = total_damage - player_state['block']
                    player_state['block'] = 0
                    player_state['player_hp'] -= remaining_damage

                if e.get('block_gain', 0) > 0:
                    e['block'] = e.get('block', 0) + e['block_gain']

            elif e['intent'] == 'OTHER':
                if e.get('strength_gain', 0) > 0:
                    e['strength'] = e.get('strength', 0) + e['strength_gain']
                if e.get('ritual_gain', 0) > 0:
                    e['ritual'] += e['ritual_gain']

            # Ritual applies every turn regardless of intent
            if e.get('ritual', 0) > 0 and r_flag:
                e['strength'] = e.get('strength', 0) + e['ritual']

            if e.get('vulnerable', 0) > 0:
                e['vulnerable'] -= 1
                e['is_vulnerable'] = 1 if e['vulnerable'] > 0 else 0
            if e.get('weak', 0) > 0:
                e['weak'] -= 1
                e['is_weak'] = 1 if e['weak'] > 0 else 0

            e['block'] = 0

        return player_state

    @staticmethod
    def is_card_playable(player_state, vector):
        return player_state['energy'] >= vector['cost']

    @staticmethod
    def encode_card(card_dict):
        """Convert card dict to fixed-size feature vector."""
        features = [
            card_dict.get("cost", 0),
            card_dict.get("damage", 0),
            card_dict.get("block", 0),
            card_dict.get("self_hp_change", 0),
            card_dict.get("vulnerable", 0),
            card_dict.get("weak", 0),
            card_dict.get("draw", 0),
            card_dict.get("attack_num", 0),
            card_dict.get("target_single", 0),
            card_dict.get("target_all", 0),
            card_dict.get("target_self", 0),
        ]
        return np.array(features, dtype=np.float32)

    @staticmethod
    def encode_enemy(enemy):
        """Encode a single enemy into a fixed-size feature vector."""
        if enemy.get('hp', 0) <= 0:
            return [0.0] * 16  # dead enemy — all zeros

        return [
            enemy.get('hp', 0) / (enemy.get('max_hp', 40) + 1),
            enemy.get('max_hp', 40) / 100,
            enemy.get('strength', 0) / 10,
            enemy.get('intentDamage', 0) / 15,
            enemy.get('vulnerable', 0) / 5,
            enemy.get('intentHits', 0) / 1,
            enemy.get('is_vulnerable', 0),
            enemy.get('weak', 0) / 5,
            enemy.get('is_weak', 0),
            1.0 if enemy.get('intent') == 'ATTACK' else 0.0,
            enemy.get('block_gain', 0) / 10,
            enemy.get('strength_gain', 0) / 5,
            enemy.get('block', 0) / 10,
            enemy.get('ritual_gain', 0) / 3,
            enemy.get('ritual', 0) / 3,
            enemy.get('curl', 0) / 4,
        ]

    @staticmethod
    def encode_state(player_state, enemies, hand, max_enemies=5):
        # Player (3 features)
        player_features = [
            player_state['player_hp'] / 80,
            player_state['block'] / 20,
            player_state['energy'] / 3
        ]

        # Enemies: pad to max_enemies slots; dead/missing = zeros
        _dummy = [0.0] * len(GameLogic.encode_enemy(EnemyFactory.dummy_monster()))
        enemy_features = []
        for i in range(max_enemies):
            if i < len(enemies) and enemies[i].get('hp', 0) > 0:
                enemy_features.extend(GameLogic.encode_enemy(enemies[i]))
            else:
                enemy_features.extend(_dummy)

        # Hand: up to 10 card slots
        hand_features = []
        dummy_card_size = len(GameLogic.encode_card({}))
        for i in range(10):
            if i < len(hand):
                card = hand[i]
                card_vec = GameLogic.encode_card(GameLogic.vectorize_card(card['name']))
                hand_features.extend(card_vec)
            else:
                hand_features.extend([0] * dummy_card_size)

        return np.array(player_features + enemy_features + hand_features, dtype=np.float32)

    def get_legal_actions(vector, enemies, max_enemies=5):
        """
        Action encoding:
          - Single-target cards: action = card_index * max_enemies + target_index
          - AoE / self-target cards: action = 10 * max_enemies + card_index
          - End turn: action = 10 * max_enemies + 10  (always legal)
        """
        legal_actions = []
        hand = vector['hand']
        energy = vector['energy']
        live_enemy_indices = [i for i, e in enumerate(enemies) if e['hp'] > 0]

        for card_idx, card in enumerate(hand):
            card_data = GameLogic.vectorize_card(card['name'])
            if energy < card_data['cost']:
                continue

            if card_data.get('target_single', 0) == 1:
                for enemy_idx in live_enemy_indices:
                    legal_actions.append(card_idx * max_enemies + enemy_idx)
            else:
                legal_actions.append(10 * max_enemies + card_idx)

        legal_actions.append(10 * max_enemies + 10)

        return legal_actions

    @staticmethod
    def decode_action(action, max_enemies=5):
        """
        Decode a flat action index back into (type, card_idx, enemy_idx).
          ('end_turn', None, None)
          ('single',   card_idx, enemy_idx)
          ('multi',    card_idx, None)
        """
        end_turn_action = 10 * max_enemies + 10
        aoe_base = 10 * max_enemies

        if action == end_turn_action:
            return ('end_turn', None, None)
        elif action >= aoe_base:
            card_idx = action - aoe_base
            return ('multi', card_idx, None)
        else:
            card_idx = action // max_enemies
            enemy_idx = action % max_enemies
            return ('single', card_idx, enemy_idx)

    @staticmethod
    def calculate_reward(action, card_data, player_state_before, player_state_after,
                         enemies_before, enemies_after, done, max_enemies=5, vector=None):
        """
        Calculate reward for a single step.
        Tune all reward weights here.
        """
        reward = 0

        DAMAGE_REWARD = 1.0
        BLOCK_REWARD = 0.2
        VULNERABLE_REWARD = 2.0
        DAMAGE_TAKEN_PENALTY = 2.0
        WASTED_ENERGY_PENALTY = 3.0
        VULNERABLE_WASTE_PENALTY = 3.0
        WASTED_BLOCK_PENALTY = 10.0
        WIN_REWARD = 50
        LOSS_PENALTY = 50
        ENEMY_SURVIVED_TURN_PENALTY = 1.0
        WEAK_REWARD = 2.0
        WEAK_WASTE_PENALTY = 1.0

        end_turn_action = 10 * max_enemies + 10

        # Penalty for targeting a dead/invalid enemy
        action_type, decoded_card_idx, target_idx = GameLogic.decode_action(action, max_enemies)
        if action_type == 'single':
            if target_idx >= len(enemies_before) or enemies_before[target_idx].get('hp', 0) <= 0:
                reward -= 50

        # Aggregate damage dealt to all enemies
        total_damage_dealt = 0
        enemy_killed = False
        for eb, ea in zip(enemies_before, enemies_after):
            hp_before = eb['hp']
            hp_after = ea['hp']
            damage = max(0, hp_before - hp_after)
            total_damage_dealt += damage

            if hp_before > 0 and hp_after <= 0:
                enemy_killed = True
                reward += 50  # kill bonus per enemy

        # Aggregate live enemy info for reward shaping
        live_enemies_before = [e for e in enemies_before if e['hp'] > 0]
        first_live = live_enemies_before[0] if live_enemies_before else None
        any_attacking = any(e.get('intent') == 'ATTACK' for e in live_enemies_before)
        all_buffing = all(e.get('intent') == 'OTHER' for e in live_enemies_before) and len(live_enemies_before) > 0
        any_killable = any(e['hp'] <= 10 for e in live_enemies_before)
        any_low = any(e['hp'] <= 20 for e in live_enemies_before)

        # Scale damage reward
        if any_low:
            DAMAGE_REWARD += 3.0
        if any_killable:
            DAMAGE_REWARD += 3.0
        if all_buffing:
            DAMAGE_REWARD *= 3.0

        reward += total_damage_dealt * DAMAGE_REWARD

        # Vulnerable handling
        if card_data and card_data.get('vulnerable', 0) > 0 and first_live:
            action_type, card_idx, enemy_idx = GameLogic.decode_action(action, max_enemies)
            if action_type == 'single' and enemy_idx < len(enemies_before):
                target_before = enemies_before[enemy_idx]
            else:
                target_before = first_live

            vuln_before = target_before.get('vulnerable', 0)
            hp_scaling = target_before['hp'] / target_before.get('max_hp', 44)

            if vuln_before == 0:
                reward += VULNERABLE_REWARD * card_data['vulnerable'] * (1.0 + hp_scaling)
            elif vuln_before >= 2:
                reward -= VULNERABLE_WASTE_PENALTY
            elif vuln_before == 1:
                reward += 1.0

        # Weak handling
        if card_data and card_data.get('weak', 0) > 0 and first_live:
            action_type, card_idx, enemy_idx = GameLogic.decode_action(action, max_enemies)
            if action_type == 'single' and enemy_idx < len(enemies_before):
                target_before = enemies_before[enemy_idx]
            else:
                target_before = first_live

            weak_before = target_before.get('weak', 0)

            if weak_before == 0:
                reward += WEAK_REWARD * card_data['weak']
            elif weak_before >= 2:
                reward -= WEAK_WASTE_PENALTY
            elif weak_before == 1:
                reward += 1.0

        # Block handling — penalize harder when enemy is in kill range
        if card_data and not enemy_killed:
            block_gained = card_data.get('block', 0)
            if block_gained > 0:
                if any_killable:
                    if card_data.get('damage', 0) == 0:
                        reward -= WASTED_BLOCK_PENALTY * 3.0
                elif any_attacking:
                    reward += block_gained * BLOCK_REWARD
                else:
                    if card_data.get('damage', 0) == 0:
                        reward -= WASTED_BLOCK_PENALTY

        # Damage taken
        hp_before = player_state_before['player_hp']
        hp_after = player_state_after['player_hp']
        damage_taken = max(0, hp_before - hp_after)
        reward -= damage_taken * DAMAGE_TAKEN_PENALTY

        # End turn penalties
        if action == end_turn_action:
            for e in enemies_before:
                if e['hp'] <= 0:
                    continue
                reward -= e['hp'] * ENEMY_SURVIVED_TURN_PENALTY
                if e['hp'] <= 10:
                    reward -= e['hp'] * ENEMY_SURVIVED_TURN_PENALTY * 3.0

            wasted_energy = player_state_before.get('energy', 0)
            if wasted_energy > 0:
                playable = vector is not None and any(
                    player_state_before['energy'] >= GameLogic.vectorize_card(c['name'])['cost']
                    for c in vector.get('hand', [])
                )
                if playable:
                    reward -= wasted_energy * WASTED_ENERGY_PENALTY * 3.0
                # no penalty if nothing was playable — end turn was forced

        # Terminal rewards
        if done:
            if player_state_after['player_hp'] <= 0:
                reward -= LOSS_PENALTY
            elif all(e['hp'] <= 0 for e in enemies_after):
                reward += WIN_REWARD

        return reward

    @staticmethod
    def step(vector, enemies, action, max_enemies=5):
        """
        Apply one action (card play or end turn).
        Returns: (next_vector, next_enemies, reward, done)
        """
        import random

        player_state_before = {
            'player_hp': vector['player_hp'],
            'block': vector['block'],
            'energy': vector['energy']
        }
        enemies_before = copy.deepcopy(enemies)

        player_state = copy.deepcopy(player_state_before)
        enemies = copy.deepcopy(enemies_before)
        hand = copy.copy(vector['hand'])
        draw_pile = copy.copy(vector.get('draw_pile', []))
        discard_pile = copy.copy(vector.get('discard_pile', []))

        done = False
        turn_ended = False
        card_data = None
        card_idx = None

        action_type, decoded_card_idx, target_enemy_idx = GameLogic.decode_action(action, max_enemies)

        if action_type == 'end_turn':
            turn_ended = True

        elif action_type in ('single', 'multi') and decoded_card_idx < len(hand):
            card_idx = decoded_card_idx
            card = hand[card_idx]
            card_data = GameLogic.vectorize_card(card['name'])

            player_state = GameLogic.apply_card_player(player_state, card_data)

            if action_type == 'single':
                if (card_data['damage'] > 0 and
                        target_enemy_idx < len(enemies) and
                        enemies[target_enemy_idx]['hp'] > 0):
                    enemies[target_enemy_idx] = GameLogic.apply_card_monster(
                        player_state, enemies[target_enemy_idx], card_data
                    )
            else:
                # AoE: apply to all living enemies
                if card_data['damage'] > 0:
                    for i, e in enumerate(enemies):
                        if e['hp'] > 0:
                            enemies[i] = GameLogic.apply_card_monster(
                                player_state, enemies[i], card_data
                            )

            cards_to_draw = card_data.get('draw', 0)
            for _ in range(cards_to_draw):
                if not draw_pile:
                    draw_pile = discard_pile.copy()
                    random.shuffle(draw_pile)
                    discard_pile = []
                if draw_pile:
                    hand.append(draw_pile.pop(0))

            played_card = hand.pop(card_idx)
            discard_pile.append(played_card)

        if turn_ended:
            discard_pile.extend(hand)
            hand = []

            player_state = GameLogic.monster_action(enemies, player_state)
            for enemy in enemies:
                if enemy['hp'] > 0:
                    EnemyFactory.update_intent(enemy)

            player_state['block'] = 0
            player_state['energy'] = 3

            for _ in range(5):
                if not draw_pile:
                    draw_pile = discard_pile.copy()
                    random.shuffle(draw_pile)
                    discard_pile = []
                if draw_pile:
                    hand.append(draw_pile.pop(0))

        if player_state['player_hp'] <= 0:
            done = True
        elif all(e['hp'] <= 0 for e in enemies):
            done = True

        next_vector = {
            'player_hp': player_state['player_hp'],
            'block': player_state['block'],
            'energy': player_state['energy'],
            'hand': hand,
            'draw_pile': draw_pile,
            'discard_pile': discard_pile
        }

        reward = GameLogic.calculate_reward(
            action=action,
            card_data=card_data,
            player_state_before=player_state_before,
            player_state_after=player_state,
            enemies_before=enemies_before,
            enemies_after=enemies,
            done=done,
            vector=vector,
        )

        return next_vector, enemies, reward, done
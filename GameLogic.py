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
        # Apply vulnerable debuff
        if vector.get('vulnerable', 0) > 0:
            monster_state['vulnerable'] = monster_state.get('vulnerable', 0) + vector['vulnerable']
            monster_state['is_vulnerable'] = 1

        # Calculate damage with vulnerable modifier
        damage = vector['damage']
        if monster_state.get('vulnerable', 0) > 0:
            damage = int(damage * 1.5)  # 50% more damage

        monster_state['block'] = monster_state.get('block', 0) - damage
        if monster_state['block'] < 0:
            damage = abs(monster_state['block'])
            monster_state['block'] = 0
        else:
            damage = 0
        monster_state['hp'] = monster_state['hp'] - damage
        monster_state['hp'] = max(0, monster_state['hp'])  # Don't go negative

        return monster_state

    @staticmethod
    def monster_action(enemies, player_state):
        for e in enemies:
            if e['intent'] == 'ATTACK':
                # Calculate damage (including strength bonus)
                total_damage = e['intentDamage'] * e['intentHits']

                # Apply damage (block absorbs first)
                if player_state['block'] >= total_damage:
                    player_state['block'] -= total_damage
                else:
                    remaining_damage = total_damage - player_state['block']
                    player_state['block'] = 0
                    player_state['player_hp'] -= remaining_damage

                # If enemy gains block this turn (Thrash)
                if e.get('block_gain', 0) > 0:
                    e['block'] = e.get('block', 0) + e['block_gain']

            elif e['intent'] == 'OTHER':
                if e.get('strength_gain', 0) > 0 or e.get('ritual',0):
                    e['strength'] = e.get('strength', 0) + e['strength_gain'] + e['ritual']
                if e.get('ritual_gain', 0) > 0:
                    e['ritual'] += e['ritual_gain']


            # Decrement vulnerable at end of enemy turn
            if e.get('vulnerable', 0) > 0:
                e['vulnerable'] -= 1
                # Update is_vulnerable flag
                e['is_vulnerable'] = 1 if e['vulnerable'] > 0 else 0

            # Enemy block expires at end of turn (StS rule)
            e['block'] = 0

        return player_state

    @staticmethod
    def is_card_playable(player_state, vector):
        return player_state['energy'] >= vector['cost']

    @staticmethod
    def encode_card(card_dict):
        """Convert card dict to fixed-size feature vector"""
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
    def encode_state(player_state, enemies, hand):
        # Player (3 features)
        player_features = [
            player_state['player_hp'] / 80,  # normalize by max HP
            player_state['block'] / 20,  # reasonable scale
            player_state['energy'] / 3  # max energy at start
        ]

        # Enemy (3 features for Jaw Worm)
        enemy = enemies[0]
        enemy_features = [
            enemy.get('hp', 0) / enemy.get('max_hp', 40),
            enemy.get('max_hp', 40) / 100,
            enemy.get('strength', 0) / 10,
            enemy.get('intentDamage', 0) / 15,  # upcoming damage
            enemy.get('vulnerable', 0) / 5,
            enemy.get('intentHits', 0) / 1,
            enemy.get('is_vulnerable', 0),
            1.0 if enemy.get('intent') == 'ATTACK' else 0.0,
            enemy.get('block_gain', 0) / 10,
            enemy.get('strength_gain', 0) /5,
            enemy.get('block', 0) / 10,
            enemy.get('ritual_gain', 0) / 3,
            enemy.get('ritual', 0) / 3,
        ]


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

    def get_legal_actions(vector, enemies):
        """
        Returns list of legal action indices.
        Actions 0-9 = play card at that hand position
        Action 10 = end turn
        """
        legal_actions = []

        hand = vector['hand']
        energy = vector['energy']

        # Check each card in hand
        for i, card in enumerate(hand):
            card_data = GameLogic.vectorize_card(card['name'])

            # Can we afford it?
            if energy >= card_data['cost']:
                # For attacks, check if there are valid targets
                if card_data.get('target_single', 0) == 1:
                    # Need at least one alive enemy
                    if any(e['hp'] > 0 for e in enemies):
                        legal_actions.append(i)
                else:
                    # Self-target or AoE always legal if we have energy
                    legal_actions.append(i)

        # End turn is always legal
        legal_actions.append(10)

        return legal_actions

    @staticmethod
    def calculate_reward(action, card_data, player_state_before, player_state_after,
                         enemies_before, enemies_after, done):
        """
        Calculate reward for a single step.
        Tune all reward weights here.
        """
        reward = 0

        # REWARD WEIGHTS (tune these!)
        DAMAGE_REWARD = 2.0  # ← Increased from 1.0
        BLOCK_REWARD = 0.3  # ← Decreased from 0.5
        VULNERABLE_REWARD = 5.0  # ← Increased from 2.0
        DAMAGE_TAKEN_PENALTY = 2.0  # ← Increased from 0.5
        WASTED_ENERGY_PENALTY = 15.0
        VULNERABLE_WASTE_PENALTY = 8.0
        WASTED_BLOCK_PENALTY = 10.0
        WIN_REWARD = 200
        LOSS_PENALTY = 100

        # Scale up damage reward when enemy is low
        if enemies_before[0]['hp'] <= 20:
            DAMAGE_REWARD += 3.0  # Was +2
        if enemies_before[0].get('intent') == 'OTHER':
            DAMAGE_REWARD *= 2.0

        enemy_killed = False
        if enemies_before and enemies_after:
            enemy_hp_before = enemies_before[0]['hp']
            enemy_hp_after = enemies_after[0]['hp']
            actual_damage = max(0, enemy_hp_before - enemy_hp_after)
            reward += actual_damage * DAMAGE_REWARD

            # Big bonus for killing
            if enemy_hp_before > 0 and enemy_hp_after <= 0:
                reward += 50
                enemy_killed = True

            # Vulnerable handling
            if card_data and card_data.get('vulnerable', 0) > 0:
                vuln_before = enemies_before[0].get('vulnerable', 0)
                enemy_hp_before = enemies_before[0]['hp']

                if vuln_before == 0:
                    hp_scaling = enemy_hp_before / enemies_before[0].get('max_hp', 44)
                    reward += VULNERABLE_REWARD * card_data['vulnerable'] * (1.0 + hp_scaling)

                elif vuln_before >= 2:
                    reward -= VULNERABLE_WASTE_PENALTY

                elif vuln_before == 1:
                    # Meh - extending from 1 stack, small reward
                    reward += 1.0

        # Block handling
        if card_data and not enemy_killed:
            block_gained = card_data.get('block', 0)
            if block_gained > 0:
                enemy_intent = enemies_before[0].get('intent')
                if enemy_intent == 'ATTACK':
                    # Reward block when useful
                    reward += block_gained * BLOCK_REWARD
                else:
                    # PENALIZE playing pure defense cards on non-attack turns
                    # But don't penalize hybrid cards (cards with damage)
                    if card_data.get('damage', 0) == 0:
                        reward -= WASTED_BLOCK_PENALTY

        # Damage taken (from enemy attacks)
        hp_before = player_state_before['player_hp']
        hp_after = player_state_after['player_hp']
        damage_taken = max(0, hp_before - hp_after)
        reward -= damage_taken * DAMAGE_TAKEN_PENALTY

        # Penalty for ending turn with leftover energy
        if action == 10:
            wasted_energy = player_state_before.get('energy', 0)
            reward -= wasted_energy * WASTED_ENERGY_PENALTY

        # Terminal rewards
        if done:
            if player_state_after['player_hp'] <= 0:
                reward -= LOSS_PENALTY
            elif all(e['hp'] <= 0 for e in enemies_after):
                reward += WIN_REWARD

        return reward

    @staticmethod
    def step(vector, enemies, action):
        """
        Apply one action (card play or end turn).
        Returns: (next_vector, next_enemies, reward, done)
        """
        import copy
        import random

        # Save state BEFORE action
        player_state_before = {
            'player_hp': vector['player_hp'],
            'block': vector['block'],
            'energy': vector['energy']
        }
        enemies_before = copy.deepcopy(enemies)

        # Deep copy state for modifications
        player_state = copy.deepcopy(player_state_before)
        enemies = copy.deepcopy(enemies_before)
        hand = copy.copy(vector['hand'])
        draw_pile = copy.copy(vector.get('draw_pile', []))
        discard_pile = copy.copy(vector.get('discard_pile', []))  # ← ADD THIS

        done = False
        turn_ended = False
        card_data = None

        # ACTION PHASE
        if action == 10:
            turn_ended = True

        elif action < len(hand):
            # Play the card
            card = hand[action]
            card_data = GameLogic.vectorize_card(card['name'])

            # Apply card effects
            player_state = GameLogic.apply_card_player(player_state, card_data)

            # Apply damage to enemy at position 0
            if card_data['damage'] > 0 and len(enemies) > 0 and enemies[0]['hp'] > 0:
                enemies[0] = GameLogic.apply_card_monster(player_state, enemies[0], card_data)

            # Handle card draw from card effects
            cards_to_draw = card_data.get('draw', 0)
            for _ in range(cards_to_draw):
                if not draw_pile:
                    # Reshuffle discard into draw pile
                    draw_pile = discard_pile.copy()
                    random.shuffle(draw_pile)
                    discard_pile = []
                if draw_pile:
                    hand.append(draw_pile.pop(0))

            # Remove played card and add to discard
            played_card = hand.pop(action)
            discard_pile.append(played_card)  # ← ADD THIS

        # ENEMY PHASE (only if turn ended)
        if turn_ended:
            # Discard remaining hand
            discard_pile.extend(hand)  # ← ADD THIS
            hand = []  # ← ADD THIS

            # Enemies attack
            player_state = GameLogic.monster_action(enemies, player_state)
            for enemy in enemies:
                if enemy['name'] == 'Jaw Worm':
                    EnemyFactory.update_jaw_worm_intent(enemy)
                elif enemy['name'] == 'Cultist':
                    EnemyFactory.update_cultist_intent(enemy)

            # Block expires at end of turn
            player_state['block'] = 0

            # Start new turn
            player_state['energy'] = 3

            # Draw 5 cards for new turn
            cards_to_draw = 5
            for _ in range(cards_to_draw):
                if not draw_pile:
                    # Reshuffle discard into draw pile
                    draw_pile = discard_pile.copy()
                    random.shuffle(draw_pile)
                    discard_pile = []
                if draw_pile:
                    hand.append(draw_pile.pop(0))

        # CHECK WIN/LOSE
        if player_state['player_hp'] <= 0:
            done = True
        elif all(e['hp'] <= 0 for e in enemies):
            done = True

        # Rebuild vector
        next_vector = {
            'player_hp': player_state['player_hp'],
            'block': player_state['block'],
            'energy': player_state['energy'],
            'hand': hand,
            'draw_pile': draw_pile,
            'discard_pile': discard_pile  # ← ADD THIS
        }

        # CALCULATE REWARD
        reward = GameLogic.calculate_reward(
            action=action,
            card_data=card_data,
            player_state_before=player_state_before,
            player_state_after=player_state,
            enemies_before=enemies_before,
            enemies_after=enemies,
            done=done
        )

        return next_vector, enemies, reward, done








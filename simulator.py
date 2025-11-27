import copy
import itertools
import random
import ironclad_dict


CARD_LIBRARY = ironclad_dict.CARD_LIBRARY

ENEMY_LIBRARY = [
    {"id": "Slime", "hp": 20, "intent": "ATTACK", "intentDamage": 5, "intentHits": 1},
    {"id": "Gremlin", "hp": 15, "intent": "ATTACK", "intentDamage": 3, "intentHits": 2},
    {"id": "Lagavulin", "hp": 100, "intent": "ATTACK", "intentDamage": 20, "intentHits": 1},
    {"id": "Cultist", "hp": 51, "intent": "BUFF", "intentDamage": 0, "intentHits": 0},
]

def draw_hand(deck, hand_size=5):
    """Draws a hand without replacement, respecting deck size."""
    return list(deck)[:hand_size] if len(deck) <= hand_size else list(deck[:hand_size])

def apply_card(player_state, enemies, card_name):
    """Apply a card to the player and enemies."""
    card = CARD_LIBRARY[card_name]
    player_state['energy'] -= card['cost']
    player_state['block'] += card['block']
    player_state['player_hp'] += card['self_hp_change']

    # Apply damage to first alive enemy (can be improved to allow targeting)
    for e in enemies:
        if e['hp'] > 0:
            e['hp'] -= card['damage']
            e['hp'] = max(e['hp'], 0)
            break
    return player_state, enemies

def enemy_phase(player_state, enemies):
    """Enemy intents phase."""
    for e in enemies:
        if e['hp'] <= 0:
            continue
        if e['intent'] == 'ATTACK':
            dmg = e['intentDamage'] * e['intentHits']
            blocked = min(dmg, player_state['block'])
            dmg -= blocked
            player_state['block'] -= blocked
            player_state['player_hp'] -= dmg
    return player_state

def simulate_full_turn(player_state, enemies, deck):
    """
    Simulate all possible sequences of the hand given energy limits.
    Returns a list of turn examples with rewards.
    """
    hand = draw_hand(deck)
    turn_examples = []

    # Generate all possible sequences of cards respecting energy
    for r in range(1, len(hand) + 1):
        for seq in itertools.permutations(hand, r):
            energy = player_state['energy']
            p_state = copy.deepcopy(player_state)
            e_state = copy.deepcopy(enemies)
            played_cards = []

            for card_name in seq:
                card_cost = CARD_LIBRARY[card_name]['cost']
                if card_cost > energy:
                    continue
                p_state, e_state = apply_card(p_state, e_state, card_name)
                energy = p_state['energy']
                played_cards.append(card_name)

            # Enemy phase
            p_state = enemy_phase(p_state, e_state)

            turn_examples.append({
                "played_cards": played_cards,
                "player_before": player_state,
                "player_after": p_state,
                "enemies_before": enemies,
                "enemies_after": e_state
            })

    return turn_examples

def generate_training_data(num_turns=50):
    data = []
    deck = list(CARD_LIBRARY.keys())
    for _ in range(num_turns):
        player_state = {"player_hp": 50, "block": 0, "energy": 3}
        enemies = [copy.deepcopy(random.choice(ENEMY_LIBRARY)) for _ in range(random.randint(1, 2))]
        turn_data = simulate_full_turn(player_state, enemies, deck)
        data.extend(turn_data)
    return data




# Example usage
if __name__ == "__main__":
    training_data = generate_training_data(num_turns=10)
    print(f"Generated {len(training_data)} full-turn examples.")

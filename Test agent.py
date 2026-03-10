import torch
import random
import numpy as np
from train import DQNAgent, DIMENSIONS, ACTION_DIM, MAX_MONSTERS, shuffle_enemies
from GameLogic import GameLogic
from EnemyFactory import EnemyFactory


def load_agent():
    agent = DQNAgent(state_dim=DIMENSIONS, action_dim=ACTION_DIM)
    agent.policy_net.load_state_dict(torch.load("trained_model.pth"))
    agent.policy_net.eval()
    agent.epsilon = 0.0
    return agent


def get_action(agent, vector, enemies):
    state = GameLogic.encode_state(
        {"player_hp": vector["player_hp"], "block": vector["block"], "energy": vector["energy"]},
        enemies,
        vector["hand"]
    )
    legal = GameLogic.get_legal_actions(vector, enemies)
    action = agent.select_action(state, legal)
    action_type, card_idx, enemy_idx = GameLogic.decode_action(action, MAX_MONSTERS)
    card_name = vector["hand"][card_idx]["name"] if action_type in ("single", "multi") else None
    return action, action_type, card_idx, enemy_idx, card_name


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_iron_wave_before_defend():
    """With Iron_Wave and Defend_R both in hand, should play Iron_Wave first
    since it provides both damage and block."""
    agent = load_agent()
    enemies = shuffle_enemies([EnemyFactory.test_attacker(hp=20, damage=11)])
    vector = {
        "player_hp": 40,
        "block": 0,
        "energy": 2,
        "hand": [{"name": "Iron_Wave"}, {"name": "Defend_R"},{"name": "Iron_Wave"},{"name": "Iron_Wave"}],
        "draw_pile": [],
        "discard_pile": []
    }
    _, _, _, _, card_name = get_action(agent, vector, enemies)
    assert card_name == "Iron_Wave", f"Expected Iron_Wave, got {card_name}"


def test_finish_low_hp_enemy():
    """Enemy at 1 HP with a lethal Strike in hand — should kill it, not block or end turn."""
    agent = load_agent()
    enemies = shuffle_enemies([EnemyFactory.test_dying(hp=1, damage=8)])
    vector = {
        "player_hp": 30,
        "block": 0,
        "energy": 3,
        "hand": [{"name": "Strike_R"}, {"name": "Defend_R"}, {"name": "Defend_R"}],
        "draw_pile": [],
        "discard_pile": []
    }
    _, action_type, card_idx, _, card_name = get_action(agent, vector, enemies)
    assert action_type != "end_turn", "Should not end turn with lethal available"
    card_data = GameLogic.vectorize_card(card_name)
    assert card_data["damage"] > 0, f"Should play a damage card to finish enemy, got {card_name}"


def test_no_end_turn_with_energy_and_lethal():
    """Should never end turn when there's energy left and a lethal card in hand."""
    agent = load_agent()
    enemies = shuffle_enemies([EnemyFactory.test_buffer(hp=3)])
    vector = {
        "player_hp": 50,
        "block": 0,
        "energy": 3,
        "hand": [{"name": "Strike_R"}, {"name": "Bash"}, {"name": "Iron_Wave"}],
        "draw_pile": [],
        "discard_pile": []
    }
    _, action_type, _, _, _ = get_action(agent, vector, enemies)
    assert action_type != "end_turn", "Should not end turn when energy and lethal cards remain"


def test_prioritize_attacking_enemy():
    """Two enemies: one attacking, one with OTHER intent. Should target the attacker."""
    agent = load_agent()
    e1 = EnemyFactory.test_attacker(hp=20, damage=12)
    e2 = EnemyFactory.test_buffer(hp=20)
    enemies = [e1, e2] + [EnemyFactory.dummy_monster()] * (MAX_MONSTERS - 2)
    vector = {
        "player_hp": 15,
        "block": 0,
        "energy": 1,
        "hand": [{"name": "Bash"}],
        "draw_pile": [],
        "discard_pile": []
    }
    _, action_type, _, enemy_idx, _ = get_action(agent, vector, enemies)
    if action_type == "single":
        target = enemies[enemy_idx]
        assert target["intent"] == "ATTACK", \
            f"Should target the attacking enemy, targeted {target['intent']} enemy instead"


def test_block_when_low_hp_incoming_damage():
    """At low HP with heavy damage incoming and no lethal, should prioritize block."""
    agent = load_agent()
    enemies = shuffle_enemies([EnemyFactory.test_attacker(hp=40, damage=5)])
    vector = {
        "player_hp": 5,
        "block": 0,
        "energy": 1,
        "hand": [{"name": "Defend_R"}, {"name": "Strike_R"}],
        "draw_pile": [],
        "discard_pile": []
    }
    _, _, _, _, card_name = get_action(agent, vector, enemies)
    assert card_name == "Defend_R", f"Expected Defend_R to survive, got {card_name}"


def test_dont_block_vs_other_intent():
    """Enemy has OTHER intent (not attacking). Should deal damage, not waste energy on block."""
    agent = load_agent()
    enemies = shuffle_enemies([EnemyFactory.test_buffer(hp=30)])
    vector = {
        "player_hp": 50,
        "block": 0,
        "energy": 1,
        "hand": [{"name": "Defend_R"}, {"name": "Strike_R"}],
        "draw_pile": [],
        "discard_pile": []
    }
    _, _, _, _, card_name = get_action(agent, vector, enemies)
    assert card_name != "Defend_R", \
        "Should not waste block when enemy isn't attacking"


def test_target_lowest_hp_enemy():
    """With two enemies and a single-target card, prefer the lower HP enemy to secure a kill."""
    agent = load_agent()
    e1 = EnemyFactory.test_dying(hp=5, damage=5)
    e2 = EnemyFactory.test_attacker(hp=30, damage=5)
    enemies = [e1, e2] + [EnemyFactory.dummy_monster()] * (MAX_MONSTERS - 2)
    vector = {
        "player_hp": 40,
        "block": 0,
        "energy": 1,
        "hand": [{"name": "Strike_R"}],
        "draw_pile": [],
        "discard_pile": []
    }
    _, action_type, _, enemy_idx, _ = get_action(agent, vector, enemies)
    if action_type == "single":
        assert enemies[enemy_idx]["hp"] == 5, \
            f"Should target 5 HP enemy for kill, targeted {enemies[enemy_idx]['hp']} HP enemy"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Iron Wave before Defend",             test_iron_wave_before_defend),
        ("Finish low HP enemy",                 test_finish_low_hp_enemy),
        ("No end turn with energy + lethal",    test_no_end_turn_with_energy_and_lethal),
        ("Prioritize attacking enemy",          test_prioritize_attacking_enemy),
        ("Block when low HP + incoming damage", test_block_when_low_hp_incoming_damage),
        ("Don't block vs OTHER intent",         test_dont_block_vs_other_intent),
        ("Target lowest HP enemy for kill",     test_target_lowest_hp_enemy),
    ]

    print(f"\nRunning {len(tests)} agent behavior tests...\n")
    results = [run_test(name, fn) for name, fn in tests]
    passed = sum(results)
    print(f"\n{passed}/{len(tests)} passed")
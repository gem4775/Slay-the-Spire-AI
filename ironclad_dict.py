
# Card types
TYPE_ATTACK = 0
TYPE_SKILL  = 1
TYPE_POWER  = 2
TYPE_STATUS = 3
TYPE_CURSE  = 4

# Target types
TARGET_ENEMY  = 0
TARGET_ALL    = 1
TARGET_SELF   = 2



CARD_LIBRARY = {
    "Strike_R": {
        "type": TYPE_ATTACK,
        "cost": 1,
        "damage": 6,
        "block": 0,
        "self_hp_change": 0,
        "vulnerable": 0,
        "weak": 0,
        "draw": 0,
        "attack_num": 1,
        "target": TARGET_ALL,
        "special": {}
    },

    "Defend_R":{
        "type": TYPE_SKILL,
        "cost": 1,
        "damage": 0,
        "block": 5,
        "self_hp_change": 0,
        "vulnerable": 0,
        "weak": 0,
        "draw": 0,
        "attack_num": 0,
        "target": TARGET_ALL,
        "special": {}
    },

    "Bash": {
        "type": TYPE_POWER,
        "cost": 2,
        "damage": 8,
        "block": 0,
        "self_hp_change": 0,
        "vulnerable": 2,
        "weak": 0,
        "draw": 0,
        "attack_num": 1,
        "target": TARGET_ENEMY,
        "special": {}
    }
}

UPGRADES = {
    "Strike_R": {
        "damage": 3
    },

    "Defend_R": {
        "block": 3
    },

    "Bash": {
        "damage": 2,
        "vulnerable": 1
    }
}

SPECIAL_FIELDS = [
    "requires_all_attacks",
    "gain_max_hp_on_kill",
    "scales_with_strikes",
    "copy_next_attack",
    "selective_exhaust",
    "ethereal",
    "exhaust",
    "unplayable",
]
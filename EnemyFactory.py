import random


# ---------------------------------------------------------------------------
# Move definitions
# A move is a plain dict describing one action an enemy can take.
# Fields mirror the intent keys used throughout GameLogic/train.
# ---------------------------------------------------------------------------

def attack(damage, hits=1, block_gain=0, strength_gain=0, ritual_gain=0):
    return {
        'intent': 'ATTACK',
        'intentDamage': damage,
        'intentHits': hits,
        'block_gain': block_gain,
        'strength_gain': strength_gain,
        'ritual_gain': ritual_gain,
    }

def buff(strength_gain=0, ritual_gain=0, block_gain=0):
    return {
        'intent': 'OTHER',
        'intentDamage': 0,
        'intentHits': 0,
        'block_gain': block_gain,
        'strength_gain': strength_gain,
        'ritual_gain': ritual_gain,
    }


# ---------------------------------------------------------------------------
# Enemy registry
# Each entry defines everything needed to create and update an enemy.
#
# 'hp'         : int or (min, max) tuple
# 'stats'      : extra fields merged onto the base monster (strength, ritual, …)
# 'move_pattern': 'random' or 'cycle'
# 'moves'      : for 'random' — list of (weight, move_fn) pairs
#                for 'cycle'  — list of move_fn in order; loops back after last
# 'first_move' (optional): move_fn called on creation instead of the normal
#              move picker — useful for enemies with a scripted opening turn.
#              When used with 'cycle', the cycle starts at index 0 after this.
# ---------------------------------------------------------------------------

ENEMY_REGISTRY = {
    'Jaw Worm': {
        'hp': (40, 44),
        'stats': {},
        'move_pattern': 'random',
        'moves': [
            # (weight, move_fn)
            (45, lambda e: attack(11 + e.get('strength', 0))),                          # Chomp
            (30, lambda e: attack(7  + e.get('strength', 0), block_gain=5)),             # Thrash
            (25, lambda e: buff(strength_gain=random.randint(3, 5))),                    # Bellow
        ],
    },

    'Cultist': {
        'hp': (48, 56),
        'stats': {'ritual': 0},
        'move_pattern': 'random',
        'moves': [
            (100, lambda e: attack(6 + e.get('strength', 0))),                          # Dark Strike
        ],
        'first_move': lambda e: buff(ritual_gain=3),                                    # Incantation
    },

    'Louse': {
        'hp': (10, 15),  # range 10–15
        'stats': {},
        'move_pattern': 'random',
        'moves': [
            (75, lambda e: attack(5 + e.get('strength', 0))),
            (25, lambda e: buff(strength_gain=3)),
        ],
    },
}


# ---------------------------------------------------------------------------
# EnemyFactory
# ---------------------------------------------------------------------------

class EnemyFactory:

    # -- Base constructors ---------------------------------------------------

    @staticmethod
    def empty_intent():
        return {
            'intent': 'OTHER',
            'intentDamage': 0,
            'intentHits': 0,
            'block_gain': 0,
            'strength_gain': 0,
            'ritual_gain': 0,
        }

    @staticmethod
    def empty_monster():
        return {
            'name': 'ERR',
            'hp': 0,
            'max_hp': 1,
            'strength': 0,
            'block': 0,
            'vulnerable': 0,
            'is_vulnerable': 0,
            'ritual': 0,
            'curl': 0,
        }

    @staticmethod
    def dummy_monster():
        monster = EnemyFactory.empty_monster()
        monster.update(EnemyFactory.empty_intent())
        return monster

    # -- Generic create / update --------------------------------------------

    @staticmethod
    def create(name: str) -> dict:
        """Create any registered enemy by name."""
        profile = ENEMY_REGISTRY[name]

        monster = EnemyFactory.empty_monster()
        monster['name'] = name

        # HP
        hp_spec = profile['hp']
        if isinstance(hp_spec, tuple):
            hp = random.randint(*hp_spec)
        else:
            hp = hp_spec
        monster['hp'] = hp
        monster['max_hp'] = hp if isinstance(hp_spec, tuple) else hp_spec

        # Extra stats (ritual, etc.)
        monster.update(profile.get('stats', {}))

        # Cycle enemies track which move comes next
        if profile.get('move_pattern') == 'cycle':
            monster['_move_index'] = 0

        # First move or normal move picker
        if 'first_move' in profile:
            intent = profile['first_move'](monster)
        else:
            intent = EnemyFactory._pick_move(monster, profile)

        full_intent = EnemyFactory.empty_intent()
        full_intent.update(intent)
        monster.update(full_intent)

        return monster

    @staticmethod
    def update_intent(enemy: dict):
        """Advance an enemy to its next intent. Works for any registered enemy."""
        name = enemy['name']
        profile = ENEMY_REGISTRY[name]
        intent = EnemyFactory._pick_move(enemy, profile)
        full_intent = EnemyFactory.empty_intent()
        full_intent.update(intent)
        enemy.update(full_intent)

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _pick_move(enemy: dict, profile: dict) -> dict:
        pattern = profile.get('move_pattern', 'random')
        moves = profile['moves']

        if pattern == 'cycle':
            idx = enemy.get('_move_index', 0)
            move_fn = moves[idx]
            enemy['_move_index'] = (idx + 1) % len(moves)
            return move_fn(enemy)
        else:
            weights = [w for w, _ in moves]
            fns     = [fn for _, fn in moves]
            chosen  = random.choices(fns, weights=weights, k=1)[0]
            return chosen(enemy)

    # -- Convenience shorthands (keep old call sites working) ----------------

    @staticmethod
    def create_jaw_worm():
        return EnemyFactory.create('Jaw Worm')

    @staticmethod
    def create_cultist():
        return EnemyFactory.create('Cultist')

    @staticmethod
    def update_jaw_worm_intent(enemy):
        EnemyFactory.update_intent(enemy)

    @staticmethod
    def update_cultist_intent(enemy):
        EnemyFactory.update_intent(enemy)
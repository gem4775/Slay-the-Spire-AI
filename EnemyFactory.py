import random


class EnemyFactory:
    @staticmethod
    def empty_intent():
        intent = {
            'intent': 'OTHER',
                'intentDamage': 0,
                'intentHits': 0,
                'block_gain': 0,
                'strength_gain': 0,
                'ritual_gain': 0,
        }
        return intent

    @staticmethod
    def empty_monster():
        monster = {
            'name': 'ERR',
            'hp': 0,
            'max_hp': 0,
            'strength': 0,
            'block': 0,
            'vulnerable': 0,
            'is_vulnerable': 0,
            'ritual': 0,
        }
        return monster
    @staticmethod
    def dummy_monster():
        monster = EnemyFactory.empty_monster()
        monster.update(EnemyFactory.empty_intent())
        return monster

    @staticmethod
    def create_jaw_worm():
        """Create a Jaw Worm with randomized starting intent"""
        jaw_worm = EnemyFactory.empty_monster()
        jaw_worm_spc = {
            'name': 'Jaw Worm',
            'hp': random.randint(40, 44),  # Jaw Worm has 40-44 HP
            'max_hp': 44,
        }
        jaw_worm.update(jaw_worm_spc)
        # Set initial intent
        jaw_worm.update(EnemyFactory._get_jaw_worm_intent(jaw_worm))
        return jaw_worm

    @staticmethod
    def _get_jaw_worm_intent(jaw_worm):
        """
        Jaw Worm move pattern:
        - 45% Chomp: 11 damage
        - 30% Thrash: 7 damage + gain 5 block
        - 25% Bellow: Gain 3-5 strength
        """
        base = EnemyFactory.empty_intent()
        current_strength = jaw_worm.get('strength', 0)
        roll = random.random()

        if roll < 0.45:
            # Chomp
            base.update({
                'intent': 'ATTACK',
                'intentDamage': 11 + current_strength,
                'intentHits': 1,
                'block_gain': 0,
                'strength_gain': 0,
            })
            return base
        elif roll < 0.75:
            # Thrash
            base.update({
                'intent': 'ATTACK',
                'intentDamage': 7 + current_strength,
                'intentHits': 1,
                'block_gain': 5,
                'strength_gain': 0,
            })
            return base
        else:
            # Bellow
            strength_gain = random.randint(3, 5)
            base.update({
                'intent': 'OTHER',
                'intentDamage': 0,
                'intentHits': 0,
                'block_gain': 0,
                'strength_gain': strength_gain,
            })
            return base

    @staticmethod
    def update_jaw_worm_intent(jaw_worm):
        """Update Jaw Worm's intent for next turn"""
        new_intent = EnemyFactory._get_jaw_worm_intent(jaw_worm)
        jaw_worm.update(new_intent)

    @staticmethod
    def create_cultist():
        """Create a cultist"""
        cultist = EnemyFactory.empty_monster()
        hp = random.randint(48, 56)
        cultist_spc = {
            'name': 'Cultist',
            'hp': hp,
            'max_hp': hp,
        }
        cultist.update(cultist_spc)
        # Set initial intent
        cultist.update(EnemyFactory._get_cultist_intent(cultist, new = True))
        return cultist

    @staticmethod
    def _get_cultist_intent(cultist, new = False):
        base = EnemyFactory.empty_intent()
        current_strength = cultist.get('strength', 0)

        if new:
            # Incantation
            base.update({
                'intent': 'OTHER',
                'intentDamage': 0,
                'intentHits': 0,
                'block_gain': 0,
                'strength_gain': 0,
                'ritual_gain': 3,
            })
            return base
        else:
            # Dark Strike
            base.update({
                'intent': 'ATTACK',
                'intentDamage': 6 + current_strength,
                'intentHits': 1,
                'block_gain': 0,
                'strength_gain': 0,
                'ritual_gain': 0,
            })
            return base
    @staticmethod
    def update_cultist_intent(cultist):
        new_intent = EnemyFactory._get_cultist_intent(cultist)
        cultist.update(new_intent)
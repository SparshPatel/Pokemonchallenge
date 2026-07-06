import sys
sys.path.insert(0,'src'); sys.path.insert(0,'submission')
import cg.api as api

atks = {a.attackId: a for a in api.all_attack()}
rock = atks.get(629)
print(f'Giant Rock: {rock.name}, dmg={rock.damage}, text={rock.text!r}')
jab = atks.get(982)
print(f'Aura Jab: {jab.name}, dmg={jab.damage}, text={jab.text!r}')

cards = list(api.all_card_data())
# Check card fields
c0 = cards[0]
print('Card fields:', [a for a in dir(c0) if not a.startswith('_') and not callable(getattr(c0,a))])

stage2 = [c for c in cards if hasattr(c, 'stage') and getattr(c,'stage',None) == 2 and c.cardType == 0]
print(f'Stage 2 count: {len(stage2)}')

# Check basic field
has_stage = sum(1 for c in cards if hasattr(c,'stage'))
print(f'Cards with stage field: {has_stage}')

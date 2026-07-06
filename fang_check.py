import sys
sys.path.insert(0,'src'); sys.path.insert(0,'submission')
import cg.api as api

cards = {c.cardId: c for c in api.all_card_data()}
atks = {a.attackId: a for a in api.all_attack()}

# Orichalcum Fang = 1409
fang = atks.get(1409)
print(f'Orichalcum Fang: {fang.name}')
print(f'Text: {fang.text}')

# check Koraidon ex card
kor = cards.get(979)
print(f'\nKoraidon ex: {kor.name}')

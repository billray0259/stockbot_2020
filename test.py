from td_api import Account

acc = Account("keys.json")

print(acc.get_positions().keys())
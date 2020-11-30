from td_api import Account
import pandas as pd
import numpy as np

acc = Account("keys.json")

orders = acc.get_orders()
positions = acc.get_positions()
quotes = acc.get_quotes(positions.keys())["mark"]


net_shares = {}
net_value = {}
for symbol in positions:
    net_shares[symbol] = -positions[symbol]
    net_value[symbol] = quotes[symbol] * positions[symbol]

for symbol in set(orders["symbol"]):
    symbol_orders = orders[orders["symbol"] == symbol].sort_index()
    for time, order in symbol_orders.iterrows():
        symbol = order["symbol"]
        amount = order["amount"]
        price = order["price"]
        side = order["side"]

        if symbol not in net_value:
            net_value[symbol] = 0
            net_shares[symbol] = 0

        if side == "BUY":
            net_shares[symbol] += amount
            net_value[symbol] -= amount * price
        else:
            net_shares[symbol] -= amount
            net_value[symbol] += amount * price



# for ticker in orders:
#     amount, price, side = orders[ticker]

#     if side == "BUY":
#         price *= -1
    
#     signed_amount = amount * -1 if side == "SELL" else 1
    
#     if ticker not in net_value:
#         net_value[ticker] = amount * price
#         net_shares[ticker] = signed_amount
#     else:
#         net_value[ticker] += amount * price
#         net_shares += signed_amount


tickers = []
values = []
for ticker in net_shares:
    if net_shares[ticker] == 0:
        tickers.append(ticker)
        values.append(net_value[ticker])
    else:
        print(ticker, net_value[ticker], net_shares[ticker])

net_value = pd.DataFrame({"ticker": tickers, "value": values})
net_value.sort_values(by=["value"], inplace=True)

print(net_value)

print(np.sum(net_value["value"]))

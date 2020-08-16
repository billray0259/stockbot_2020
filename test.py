from td_api import Account
from datetime import datetime, timedelta

acc = Account("keys.json")


symbol = "AMD"
days = 30
strike_count = 50

from_date = datetime.now()
to_date = from_date + timedelta(days=days)
data = acc.get_options_chain(symbol, from_date, to_date, strike_count=strike_count)

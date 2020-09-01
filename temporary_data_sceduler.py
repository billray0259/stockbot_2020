import time
import datetime
import sys
import options_data_collector

ran_collect = False
ran_tickers = False

while True:
    if ((datetime.datetime.now().hour == 9 and datetime.datetime.now().minute >= 30) or datetime.datetime.now().hour > 9) and datetime.datetime.now().hour < 16 and not ran_collect:
        ran_collect = True
        print("Beggining Collection")
        options_data_collector.collect_data()
    elif (datetime.datetime.now().hour == 9 and datetime.datetime.now().minute < 30) and not ran_tickers:
        ran_tickers = True
       # options_data_collector.update_tickers()
        print("Updated Tickers")
        time.sleep(5)
        options_data_collector.gather_symbols()
        print("Gathered Symbols")
    else:
        print("Outside of trading hours")
        time.sleep(3)
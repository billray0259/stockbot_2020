from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import td_api as td
import time
import os
import sys
import math
from multiprocessing import Pool
from bs4 import BeautifulSoup
import requests
from data_handler import DataHandler
import traceback




NUM_TICKERS = 100
STRIKE_COUNT = 50
NUM_QUOTES = 350

PERIOD_SECONDS = 60*10

TIME_MOD_EST = 0

SAVE_DIR = "options_data"

account = td.Account("keys.json")


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_monthly_expiration_dates(num_dates):
    expiration_days = []

    day = datetime.now()
    day = datetime(year=day.year, month=day.month, day=1)
    delta = timedelta(days=1)
    friday_count = 0
    while len(expiration_days) < 6:
        if day.weekday() == 4:
            friday_count += 1

        if friday_count == 3:
            if day > datetime.now() - timedelta(days=1):
                expiration_days.append(datetime(year=day.year, month=day.month, day=day.day))
            friday_count = 0

            if day.month == 12:
                day = datetime(year=day.year+1, month=1, day=1)
            else:
                day = datetime(year=day.year, month=day.month+1, day=1)
        else:
            day += delta

    return expiration_days


def get_fridays(startDate):
    d = startDate
    x = 0
    fridays = []
    while x < 4:
        if d.weekday() == 4:
            x += 1
            fridays.append(d)
        d += timedelta(days=1)
    return fridays


def initalize():
    os.chdir(os.path.dirname(__file__))
    if not os.path.exists(SAVE_DIR):
        print("Creating %s directory" % SAVE_DIR)
        os.mkdir(SAVE_DIR)


def update_available_tickers():
    try:
        dh = DataHandler("optionable", "v=111&f=cap_smallover,sh_curvol_o750,sh_opt_option,sh_price_o2&o=-volume")
        dh.save_finviz()
        return dh.get_save_file()
    except Exception as e:
        print("[%s]" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Exception when updating tickers")
        traceback.print_exc()
        return None

def check_option_volume(data_folder):
    if data_folder is not None:
        data = pd.read_hdf(data_folder)
        to_date = min(get_fridays(datetime.now()))
        from_date = datetime.now()
        vals = []
        last_request_time = time.time()
        for ticker in data.index:
            time.sleep(max(0.001, 0.6 - (time.time() - last_request_time)))
            last_request_time = time.time()
            options_chain = account.get_options_chain(ticker, from_date=from_date, to_date=to_date, strike_count=STRIKE_COUNT)
            if not options_chain is None:
                vals.append((sum(options_chain['totalVolume']), ticker))
        vals.sort(reverse = True)
        final = []
        for tickers in vals:
            if len(final) == NUM_TICKERS:
                break
            final.append(tickers[1])
        return final
    return None

def update_primary():
    print('Gathering potential data and sorting by options volume')
    tickers = check_option_volume(update_available_tickers())
    if tickers is None:
        tickers = []
    now = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame(columns=['Time', 'Ticker'])
    for ticker in tickers:
        df = df.append(pd.DataFrame(data={'Time': [now], 'Ticker': [ticker]}), ignore_index=True)
    print("Saving tickers to primary_tickers.csv")
    df.to_csv('options_data/primary_tickers.csv')


def update_secondary():
    if not os.path.exists('options_data/primary_tickers.csv'):
        update_primary()
    if not os.path.exists('options_data/secondary_tickers.csv'):
        secondary = pd.DataFrame(columns=['Time', 'Ticker'])
    else:
        secondary = pd.read_csv('options_data/secondary_tickers.csv')
    primary = pd.read_csv('options_data/primary_tickers.csv')
    primary_time_tickers = list(zip(primary['Time'], primary['Ticker']))
    secondary_time_tickers = list(zip(secondary['Time'], secondary['Ticker']))
    primary_tickers = list(primary['Ticker'])
    dropped_tickers = []
    for time, ticker in secondary_time_tickers:
        if ticker not in primary_tickers:
            dropped_tickers.append((time, ticker))
    for item in dropped_tickers:
        primary_time_tickers.append(item)
    df = pd.DataFrame(primary_time_tickers, columns=['Time', 'Ticker'])
    df.to_csv('options_data/secondary_tickers.csv')

def update_tickers():
    update_primary()
    update_secondary()

# def update_tickers():
#     """ 
#     options_data/tickers.csv is a series where the index is the tickers and the values are 
#     """
#     tickers_path = os.path.join(SAVE_DIR, "tickers.csv")
#     if not os.path.exists(tickers_path):
#         open(tickers_path, "w").close()
#         saved_tickers = None
#     else:
#         saved_tickers = pd.read_csv(tickers_path)

#     page = requests.get("https://research.investors.com/options-center/reports/option-volume", headers={"User-Agent": "Chrome"})
#     soup = BeautifulSoup(page.text, features="lxml")
#     print(soup)
#     items = soup.findAll("a", {"class": "stockRoll"})
#     print(items)
#     fresh_tickers = [item.text for item in items][:100]
#     date_format = "%Y-%m-%d"
#     date_string = datetime.now().strftime(date_format)

#     dates = [date_string] * len(fresh_tickers)
#     print(dates, 'date')
#     fresh_tickers = pd.Series(dates, index=fresh_tickers)
#     print(saved_tickers, 'saved0')
#     if saved_tickers is None:
#         fresh_tickers.to_csv(tickers_path)
#         return

#     old_tickers = []
#     new_tickers = []
#     for ticker in fresh_tickers:
#         if ticker in saved_tickers:
#             old_tickers.append(ticker)
#         else:
#             new_tickers.append(ticker)
#     print(fresh_tickers, 'fresh')
#     print(old_tickers,'old')
#     print(saved_tickers, 'saved')
#     print(date_string)
#     saved_tickers[old_tickers] = fresh_tickers[date_string]

#     for ticker, date in saved_tickers.iteritems():
#         if datetime.strptime(date, date_format) < datetime.now() - timedelta(days=30):
#             del saved_tickers[ticker]
    
#     saved_tickers.append(fresh_tickers[new_tickers])

#     saved_tickers.to_csv(fresh_tickers)

def gather_symbols():
    symbols = []

    last_request_time = time.time()
    tickers = pd.read_csv('options_data/secondary_tickers.csv')
    new_columns = tickers.columns.values
    print(new_columns)
    new_columns[2] = 'ticker'
    new_columns[1] = 'date'
    tickers.columns = new_columns
    tickers = tickers.set_index('ticker')['date'] 
    print(tickers)
    for ticker, timestamp in tickers.iteritems():
        from_date = datetime.strptime(timestamp, '%Y-%m-%d')
        to_date = max(get_fridays(from_date))
        if datetime.now() < to_date:
            time.sleep(max(0.001, 0.6 - (time.time() - last_request_time)))
            last_request_time = time.time()
            options_chain = account.get_options_chain(ticker, from_date=from_date, to_date=to_date, strike_count=STRIKE_COUNT)
            if not options_chain is None:
                symbols.extend(options_chain.index)
            else:
                open('options_data/failed_tickers.txt', 'a+').write("Failed on " + str(ticker) + '\n')
            print("Collected: " + str(len(symbols)), end='\r', flush=True)
    with open("options_data/symbols.txt", "w+") as symbols_file:
        print("Writing Symbols")
        symbols_file.write("\n".join(symbols))


def collect_data():
    with open("options_data/symbols.txt", "r") as symbols_file:
        symbols = list(map(lambda x: x.strip(), symbols_file.readlines()))

    symbol_batches = list(divide_chunks(symbols, 350))
    past_time = time.time() - 60*10
    while datetime.now().hour < 16 + TIME_MOD_EST:
        now = time.time()
        if now - past_time >= 60*10:
            print("Collecting Data")
            start = datetime.now()
            master_df = pd.DataFrame()
            past_time = now
            last_request_time = time.time()
            for x, symbol_batch in enumerate(symbol_batches):
                quote = account.get_quotes(symbol_batch)
                master_df = master_df.append(quote)
                quote = None
                print('Collected ' + str(x) + 'th Batch out of ' + str(len(symbol_batches)), end='\r', flush=True)
                time.sleep(max(0.001, 0.6 - (time.time() - last_request_time)))
                last_request_time = time.time()
            file_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
            master_df.to_csv('options_data/' + file_name)
            print('Collected Data For ' + file_name)
            print(datetime.now()-start)
    # symbol_batches = list(divide_chunks(list(divide_chunks(symbols, 350)), 12))
    # past_time = time.time() - 60*10
    # while datetime.now().hour > 16 + TIME_MOD_EST:
    #     now = time.time()
    #     if now - past_time >= 60*10:
    #         print("Collecting Data")
    #         master_df = pd.DataFrame()
    #         past_time = now
    #         for x, symbol_batch in enumerate(symbol_batches):
    #             p = Pool(12)
    #             quotes = p.map(account.get_quotes, symbol_batch)
    #             for quote in quotes:
    #                 master_df = master_df.append(quote)
    #             quotes = None
    #             print('Collected ' + str(x))
    #             time.sleep(6)
    #         file_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
    #         master_df.to_csv('options_data/' + file_name)
    #         print('Collected Data For ' + file_name)


if __name__ == '__main__':
    initalize()
    if sys.argv[1] == 'update': # Run saturdays 9:30am
        #update_primary()
        #update_secondary()
        # if sys.argv[2] == 'primary':
        #     update_primary()
        # elif sys.argv[2] == 'secondary':
        #     update_secondary()
        update_tickers()
    elif sys.argv[1] == "list": # Run weekdays at 9:00am
        gather_symbols()

    elif sys.argv[1] == 'collect': # Run weekdays at 9:30am
        collect_data()

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import td_api as td
import time
import os
from os import path
import sys
import math
from multiprocessing import Pool
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

#NUM_TICKERS = 100
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
    if not path.exists('options_data'):
        print("Creating options_data directory")
        os.mkdir('options_data')


def update_primary():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    print("Loading barchart.com")
    driver.get('https://www.barchart.com/options/volume-leaders/stocks?page=all')
    xpath = '/html/body/div[2]/div/div[2]/div[2]/div/div/div/div/div/div[5]/div/div[2]/div/div/ng-transclude/table/tbody/tr[%d]/td[1]/div'
    while len(driver.find_elements_by_xpath(xpath % 1)) == 0:
        time.sleep(0.25)
    print("Collecting tickers")
    tickers = set([])
    for i in range(1, 750):
        ticker = driver.find_element_by_xpath(xpath % i).text
        print(ticker + "\t%.0f%%" % (len(tickers)), end="\r", flush=True)
        tickers.add(ticker)
        if len(tickers) == 100:
            break
    now = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame(columns=['Time', 'Ticker'])
    for ticker in tickers:
        df = df.append(pd.DataFrame(data={'Time': [now], 'Ticker': [ticker]}), ignore_index=True)
    print("Saving tickers to primary_tickers.csv")
    df.to_csv('options_data/primary_tickers.csv')


def update_secondary():
    if not path.exists('options_data/primary_tickers.csv'):
        update_primary()
    if not path.exists('options_data/secondary_tickers.csv'):
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


def gather_symbols():
    symbols = []

    last_request_time = time.time()
    secondary = pd.read_csv('options_data/secondary_tickers.csv')
    tickers = list(zip(secondary['Time'], secondary['Ticker']))
    for timestamp, ticker in tickers:
        from_date = datetime.strptime(timestamp, '%Y-%m-%d')
        to_date = max(get_fridays(from_date))
        if datetime.now() < to_date:
            time.sleep(max(0.001, 0.6 - (time.time() - last_request_time)))
            last_request_time = time.time()
            options_chain = account.get_options_chain(ticker, from_date=from_date, to_date=to_date, strike_count=STRIKE_COUNT)
            symbols.extend(options_chain.index)
            print("Collected: " + str(len(symbols)), end='\r', flush=True)

    with open("options_data/symbols.txt", "w+") as symbols_file:
        print("Writing Symbols")
        symbols_file.write("\n".join(symbols))


def collect_data():
    with open("options_data/symbols.txt", "r") as symbols_file:
        symbols = list(map(lambda x: x.strip(), symbols_file.readlines()))

    symbol_batches = list(divide_chunks(list(divide_chunks(symbols, 350)), 12))
    past_time = time.time() - 60*10
    while datetime.now().hour < 16 + TIME_MOD_EST:
        now = time.time()
        if now - past_time >= 60*10:
            print("Collecting Data")
            master_df = pd.DataFrame()
            past_time = now
            for x, symbol_batch in enumerate(symbol_batches):
                p = Pool(12)
                quotes = p.map(account.get_quotes, symbol_batch)
                for quote in quotes:
                    master_df = master_df.append(quote)
                quotes = None
                print('Collected ' + str(x))
                time.sleep(6)
            file_name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".csv"
            master_df.to_csv('options_data/' + file_name)
            print('Collected Data For ' + file_name)


if __name__ == '__main__':
    initalize()
    if sys.argv[1] == 'update': # Run saturdays 9:30am
        update_primary()
        update_secondary()
        # if sys.argv[2] == 'primary':
        #     update_primary()
        # elif sys.argv[2] == 'secondary':
        #     update_secondary()

    elif sys.argv[1] == "list": # Run weekdays at 9:00am
        gather_symbols()

    elif sys.argv[1] == 'collect': # Run weekdays at 9:30am
        collect_data()

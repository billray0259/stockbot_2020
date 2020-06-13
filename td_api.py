# https://developer.tdameritrade.com/apis

import requests
import json
from datetime import datetime
from datetime import timedelta
import time
import pandas as pd

ERROR_FILE_NAME = ""


class Account:

    def __init__(self, keys_file_name):
        with open(keys_file_name, "r") as keys_file:
            keys = json.load(keys_file)
            self.refresh_token = keys["refresh_token"]
            self.client_id = keys["client_id"]
            self.account_id = keys["account_id"]
        self.update_access_token()

        # https://developer.tdameritrade.com/account-access/apis/get/accounts-0
        headers = {
            "Authorization": "Bearer " + self.access_token
        }
        endpoint = r"https://api.tdameritrade.com/v1/accounts"
        accounts = requests.get(url=endpoint, headers=headers).json()
        self.account = None
        for account in accounts:
            if account["securitiesAccount"]["accountId"] == self.account_id:
                self.account = account
                break
        if not self.account:
            print("Account ID not found")

    def update_access_token(self):
        # https://developer.tdameritrade.com/authentication/apis/post/token-0
        url = r"https://api.tdameritrade.com/v1/oauth2/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        payload = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id
        }
        response = requests.post(url, headers=headers, data=payload)
        access_token_json = requests.post(
            url, headers=headers, data=payload).json()
        self.access_token = access_token_json["access_token"]
        self.access_token_update = datetime.now()

    def place_order(self, ticker, amount, side):
        # https://developer.tdameritrade.com/account-access/apis/post/accounts/%7BaccountId%7D/orders-0
        header = {
            'Authorization': "Bearer " + self.access_token,
            "Content-Type": "application/json"
        }
        endpoint = r"https://api.tdameritrade.com/v1/accounts/{}/orders".format(
            self.account_id)

        payload = {
            'orderType': 'MARKET',
            'session': 'NORMAL',
            'duration': 'DAY',
            'orderStrategyType': 'SINGLE',
            'orderLegCollection': [
                {
                    'instruction': side,
                    'quantity': amount,
                    'instrument': {
                        'symbol': ticker,
                        'assetType': 'EQUITY'
                    }
                }
            ]
        }

        response = requests.post(url=endpoint, json=payload, headers=header)

    def buy(self, ticker, amount):
        return self.place_order(ticker, amount, "BUY")

    def sell(self, ticker, amount):
        return self.place_order(ticker, amount, "SELL")

    def history(self, ticker, frequency, days, days_ago=0, frequency_type="minute", need_extended_hours_data=False):
        day_ms = 1000*60*60*24
        end_date_ms = round(time.time()*1000 - days_ago * day_ms)
        start_date_ms = round(end_date_ms - days * day_ms)

        endpoint = r"https://api.tdameritrade.com/v1/marketdata/{}/pricehistory".format(
            ticker)
        headers = {
            "Authorization": "Bearer " + self.access_token
        }
        all_collected = False
        candles_df = None
        while not all_collected:
            print("Asking for", datetime.fromtimestamp(start_date_ms/1000), "to", datetime.fromtimestamp(end_date_ms/1000))
            payload = {
                "frequencyType": frequency_type,
                "frequency": frequency,
                "endDate": end_date_ms,
                "startDate": start_date_ms,
                "needExtendedHoursData": need_extended_hours_data
            }
            response = requests.get(
                url=endpoint, params=payload, headers=headers)
            data = response.json()
            if data["empty"]:
                all_collected = True

            candles = data["candles"]
            columns = candles[0].keys()
            if candles_df is None:
                candles_df = pd.DataFrame(columns=columns)

            for candle in candles:
                candles_df = candles_df.append(candle, ignore_index=True)

            start_date_given = candles[0]["datetime"]
            if start_date_given <= start_date_ms:
                all_collected = True
            else:
                end_date_ms = start_date_given
                save_df = candles_df.astype(
                    {"volume": "int64", "datetime": "int64"}).set_index("datetime")
                save_df.to_csv("AMD.csv")
                print(candles[0]["datetime"], candles[-1]["datetime"])

        candles_df = candles_df.astype(
            {"volume": "int64", "datetime": "int64"})

        if len(candles_df) == 0:
            return False
        return candles_df.set_index("datetime")


account = Account("keys.json")

candles = account.history("AMD", 15, 2)

candles.to_csv("AMD.csv")

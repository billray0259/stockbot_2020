# https://developer.tdameritrade.com/apis

import requests
import json
from datetime import datetime
from datetime import timedelta
import time
import pandas as pd


class Account:

    def __init__(self, keys_file_name):
        with open(keys_file_name, "r") as keys_file:
            keys = json.load(keys_file)
            self.refresh_token = keys["refresh_token"]
            self.client_id = keys["client_id"]
            self.account_id = keys["account_id"]
        self.session = requests.Session()
        self.update_access_token()

        # https://developer.tdameritrade.com/account-access/apis/get/accounts-0
        headers = {
            "Authorization": "Bearer " + self.access_token
        }
        endpoint = r"https://api.tdameritrade.com/v1/accounts"
        accounts = self.session.get(url=endpoint, headers=headers).json()
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
        response = self.session.post(url, headers=headers, data=payload)
        access_token_json = self.session.post(
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

        response = self.session.post(url=endpoint, json=payload, headers=header)
        if response.status_code == 401:
            self.update_access_token()
            response = self.session.post(url=endpoint, json=payload, headers=header)


    def buy(self, ticker, amount):
        return self.place_order(ticker, amount, "BUY")

    def sell(self, ticker, amount):
        return self.place_order(ticker, amount, "SELL")

    def history(self, ticker, frequency, days, days_ago=0, frequency_type="minute", need_extended_hours_data=False):
        candles_df = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        day_ms = 1000*60*60*24
        end_date_ms = round(time.time()*1000 - days_ago * day_ms)
        start_date_ms = round(end_date_ms - days * day_ms)

        endpoint = r"https://api.tdameritrade.com/v1/marketdata/{}/pricehistory".format(
            ticker)
        headers = {
            "Authorization": "Bearer " + self.access_token
        }

        payload = {
            "frequencyType": frequency_type,
            "frequency": frequency,
            "endDate": end_date_ms,
            "startDate": start_date_ms,
            "needExtendedHoursData": need_extended_hours_data
        }
        response = self.session.get(url=endpoint, params=payload, headers=headers)
        if response.status_code == 401:
            self.update_access_token()
            response = self.session.get(url=endpoint, params=payload, headers=headers)
        elif not response:
            print("Bad response when requesting history for", ticker)
            print(response, response.text)
            return candles_df

        data = response.json()
        if data["empty"]:
            print("No data for", ticker)
            return candles_df

        candles = data["candles"]
        candles_df = pd.read_json(json.dumps(candles), orient="records")
        candles_df = candles_df.astype(
            {"volume": "int64", "datetime": "datetime64[ms]"})
        
        return candles_df.set_index("datetime")


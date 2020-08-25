import requests
import json
from datetime import datetime, timedelta
import time
import pandas as pd


class Account:

    def __init__(self, keys_file_name):
        """ Account object to interact with the TD Ameritrade API

        Args:
            keys_file_name (string): Path to JSON file used to connect to the TD Ameritrade API. Must contain refresh_token, client_id, and account_id.
        """
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

    def update_access_token(self, wait=False):
        """ Uses the refresh_token and client_id to get a fresh access_token.
        """
        if isinstance(wait, (int, float)):
            time.sleep(wait)
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

        access_token_json = self.session.post(url, headers=headers, data=payload).json()
        try:
            self.access_token = access_token_json["access_token"]
        except KeyError as e:
            print(e)
            print("access_token_json was:", access_token_json)
            print('Trying to get access again')
            if isinstance(wait, bool):
                wait = 0
            self.update_access_token(wait+0.5)
            #exit()
        self.access_token_update = datetime.now()

    def place_order(self, ticker, amount, side):
        """ Places makret order with 'DAY' duration and 'NORMAL' session.
        Args:
            ticker (string): Symbol to trade.
            amount ([type]): Number of shares to trade.
            side ([type]): BUY or SELL
        """
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
            headers = {'Authorization': "Bearer {}".format(self.access_token)}
            response = self.session.post(url=endpoint, json=payload, headers=headers)


    def buy(self, ticker, amount):
        """ Buys 'amount' shares of 'ticker'

        Args:
            ticker (string): Ticker to buy.
            amount (int): Number of shares to buy
        """
        self.place_order(ticker, amount, "BUY")

    def sell(self, ticker, amount):
        """ Sells 'amount' shares of 'ticker'

        Args:
            ticker (string): Ticker to sell.
            amount (int): Number of shares to sell
        """
        self.place_order(ticker, amount, "SELL")

    def history(self, ticker, frequency, days, days_ago=0, frequency_type="minute", need_extended_hours_data=False):
        """ Compiles a DataFrame containing candles for the given ticker

        Args:
            ticker (string): Ticker to retrieve history for
            frequency (int): Number of frequency_type in a candle. 1, 5, 10, 15, or 30 if frequency_type is "minute" otherwise 1.
            days (float): Number of days in the time frame of interest.
            days_ago (int, optional): The most recent candle will be this many days old. Defaults to 0.
            frequency_type (str, optional): The units of frequency. minute, daily, weekly, or monthly. Defaults to "minute".
            need_extended_hours_data (bool, optional): True returns extended hours data, False returns market hours only. Defaults to False.

        Returns:
            DataFrame: With columns open, high, low, close, volume and index datetime
        """

        # https://developer.tdameritrade.com/price-history/apis/get/marketdata/%7Bsymbol%7D/pricehistory

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
            headers = {'Authorization': "Bearer {}".format(self.access_token)}
            response = self.session.get(url=endpoint, params=payload, headers=headers)
        elif not response:
            print("Bad response when requesting history for", ticker)
            print(response, response.text)
            return candles_df

        data = response.json()
        try:
            if data["empty"]:
                print("No data for", ticker)
                return candles_df
        except KeyError:
            print("No key 'empty'")
            print(response, response.text)
            return candles_df

        candles = data["candles"]
        candles_df = pd.read_json(json.dumps(candles), orient="records")
        candles_df = candles_df.astype(
            {"volume": "int64", "datetime": "datetime64[ms]"})
        
        return candles_df.set_index("datetime")
    

    def get_options_chain(self, ticker, from_date=None, to_date=None, range_="ALL", strike=None, contract_type="ALL", strike_count=1, include_quotes=False, exp_month="ALL", option_type="ALL"):
        # API documentation: https://developer.tdameritrade.com/option-chains/apis/get/marketdata/chains

        if from_date is None:
            from_date = datetime.now()
        
        if to_date is None:
            to_date = datetime.now() + timedelta(days=30)

        if type(from_date) is not str:
            from_date = from_date.strftime("%Y-%m-%d'T'%H:%M:%S")
        
        if type(to_date) is not str:
            to_date = to_date.strftime("%Y-%m-%d'T'%H:%M:%S")  
        
        headers = {'Authorization': "Bearer {}".format(self.access_token)}
        endpoint = r'https://api.tdameritrade.com/v1/marketdata/chains'
        payload = {
            'apikey': self.client_id, 
            'symbol': ticker,
            'contractType': contract_type,
            'strikeCount': strike_count,
            'includeQuotes': include_quotes,
            'fromDate': from_date,
            'toDate': to_date,
            'expMonth': exp_month,
            'optionType': option_type,
        }

        if strike is not None:
            payload["strike"] = strike
        else:
            payload["range"] = range_
        
        response = requests.get(url=endpoint, headers=headers, params=payload)
        if response.status_code == 401:
            self.update_access_token()
            headers = {'Authorization': "Bearer {}".format(self.access_token)}
            response = requests.get(url=endpoint, headers=headers, params=payload)
        
        json_data = response.json()

        dfs = []
        for map_name in ["putExpDateMap", "callExpDateMap"]:
            try:
                date_map = json_data[map_name]
            except KeyError:
                continue
            for date in date_map:
                strikes = date_map[date]
                for strike in strikes:
                    contract = date_map[date][strike][0]
                    if type(contract["optionDeliverablesList"]) is list:
                        contract["optionDeliverablesList"] = contract["optionDeliverablesList"][0]
                    dfs.append(pd.DataFrame(contract, index=[0]))
        if len(dfs) == 0:
            print("No options data for", ticker)
            return None
        df = pd.concat(dfs)
        df.set_index("symbol", inplace=True)
        # columns = [
        #     "symbol",
        #     "bidPrice",
        #     "bidSize",
        #     "askPrice",
        #     "askSize",
        #     "lastPrice",
        #     "quoteTimeInLong",
        #     "strikePrice",
        #     "contractType",
        #     "underlying",
        #     "delta",
        #     "gamma",
        #     "theta",
        #     "vega",
        #     "rho",
        #     "underlyingPrice"
        # ] + an expiration column
        return df

    # def get_large_quotes(self, symbols, chunk_size):
    #     chunk_list = [symbols[i * chunk_size:(i + 1) * chunk_size] for i in range((len(symbols) + chunk_size - 1) // chunk_size )]
    #     for chunk in chunk_list:


    def get_quotes(self, symbols):
        if type(symbols) is not str:
            symbols = ",".join(symbols)

        headers = {'Authorization': "Bearer {}".format(self.access_token)}
        endpoint = r'https://api.tdameritrade.com/v1/marketdata/quotes'
        payload = {
            'apikey': self.client_id, 
            'symbol': symbols,
        }
        
        response = requests.get(url=endpoint, headers=headers, params=payload)
        if response.status_code == 401:
            """Unathorized Error"""
            # Our access key probably just expired, Just reset it and try again. -CR
            self.update_access_token()
            headers = {'Authorization': "Bearer {}".format(self.access_token)}
            time.sleep(0.5)
            response = requests.get(url=endpoint, headers=headers, params=payload)
        if response.status_code == 429:
            """To Many Attempts Error"""
            # Haven't Really found a good solution to this that would be implemented here
            # I mostly write the code to avoid this error at all costs. -CR
            time.sleep(1)
            return None
        if response.status_code == 400:
            """We Requested a Null Value Error"""
            # Sometimes TD sends back this error on specific requests
            # Im not really sure why, I think it has to do with the request string being too long
            # I recursivly cut the request in half and then ask for it again, this seems to fix it -CR
            first_df = None
            symbols = symbols.split(',')
            time.sleep(0.5)
            while first_df is None:
                first_df = self.get_quotes(symbols[:len(symbols)//2])
            time.sleep(0.5)
            second_df = None
            while second_df is None:
                second_df = self.get_quotes(symbols[len(symbols)//2:])
            return first_df.append(second_df)
            #response = self.get_quotes(symbols)
        if not response:
            print(response, response.text)
            return None

        json_data = response.json()
        
        dfs = []
        for symbol in json_data:
            contract = json_data[symbol]
            dfs.append(pd.DataFrame(contract, index=[0]))
        if len(dfs) == 0:
            print("No symbols in json response")
            print(json_data)
            return None
        df = pd.concat(dfs)
        df.set_index("symbol", inplace=True)
        #print(symbols)
        return df

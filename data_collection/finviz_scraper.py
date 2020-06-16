import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_dataframe(arg_string, pages=None):
    url_base = "https://finviz.com/screener.ashx?" + arg_string + "&r="

    i = 1
    df = None
    while pages is None or i < pages*20:
        url = url_base + str(i)
        response = requests.get(url, headers={"User-Agent": "Chrome 80"})
        soup = BeautifulSoup(response.text, features="lxml")

        tables = soup.find_all("table")
        data_table = tables[16]
        rows = data_table.find_all("tr")

        if len(rows) == 2:
            pages = 0
        
        if df is None:
            columns = []
            for cell in rows[0].find_all("td"):
                columns.append(cell.text)
            df = pd.DataFrame(columns=columns)

        for html_row in rows[1:]:
            row = []
            cells = html_row.find_all("td")
            for cell in cells:
                row.append(cell.text)
            df = df.append(dict(zip(columns, row)), ignore_index=True)
        print("Collected:", len(df))
        i += 20
    return df.drop(columns=["No."]).set_index("Ticker")

if __name__ == "__main__":
    save_directory = "/home/bill/stockbot_2020/data/"
    save_file = "finviz_tech.csv"
    arg_string = "v=111&f=sec_technology,sh_curvol_o500,sh_relvol_u2&o=-volume"
    
    df = get_dataframe(arg_string)
    print(df.info())
    print(df.describe())
    print(df.head())
    df.to_csv(save_directory + save_file)
    

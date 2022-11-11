import requests
from datetime import datetime
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

START = "2022-10-01T00:00:00+00:00"
END = "2022-11-01T23:00:00+00:00"

instruments = ['BTC', 'ETH', 'SOL']


class FTXDataException(Exception):
    pass


def _get_returns(market: str, start: datetime, end: datetime, resolution: int):
    print(f"Fetching return information for {market}")
    url = f"https://ftx.com/api/markets/{market}/candles?resolution={resolution}&start_time={int(start.timestamp())}&end_time={int(end.timestamp())}"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    response_json = json.loads(response.text)
    if not response.status_code == 200:
        raise FTXDataException(f"{response_json['error']}")
    df = pd.DataFrame.from_dict(response_json["result"])
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['log_ret'] = np.log(df.close) - np.log(df.close.shift(1))
    returns = df[['startTime', 'log_ret']]
    returns.columns = ["time", market]
    returns.set_index("time")
    return returns.dropna()


def main():
    start = datetime.fromisoformat(START)
    end = datetime.fromisoformat(END)
    res = 60 * 60
    close_df = None
    for instrument in instruments:
        try:
            df = _get_returns(f"{instrument}-PERP", start, end, res)
        except FTXDataException as e:
            print(str(e))
            return
        if close_df is None:
            close_df = df
        else:
            close_df = close_df.merge(df, how='left')
    cov = close_df.cov(numeric_only=True).to_numpy()
    mean = close_df.mean(numeric_only=True).to_numpy()

    # Calculation of Global Minimum Variance Portfolio
    print("Calculating GMVP")
    inv_cov = np.linalg.inv(cov)
    gmvp = np.dot(inv_cov, np.ones(len(instruments))) / np.sum(np.dot(inv_cov, np.ones(len(instruments))))
    gmvp_dict = {f"{instrument}-PERP": w for instrument, w in zip(instruments, gmvp)}
    print(f"GMVP: {gmvp_dict}")
    with open('gmvp.txt', 'w') as convert_file:
        convert_file.write(json.dumps(gmvp_dict))

    """
    Plotting of efficient frontier.
    From 2 fund theorem, it is known that we can get the whole efficient frontier using a linear comb. of efficient 
    portfolios. GMVP is part of the efficient frontier. The easiest portfolio to find in the efficient frontier
    corresponds to the asset with the maximum mean return. 
    """

    print("Plotting efficient frontier")
    max_return_portfolio = np.zeros(len(instruments))
    max_return_portfolio[np.argmax(mean)] = 1

    # Use linear combination of efficient portfolios.
    frontier_portfolio = lambda w: w * gmvp + (1 - w) * max_return_portfolio
    mu = lambda p: np.dot(p, mean)
    sigma = lambda p: np.matmul(p, np.matmul(cov, p))
    weight = np.arange(51) / 50
    portfolios = [frontier_portfolio(w) for w in weight]
    y = [mu(p) for p in portfolios]
    x = [sigma(p) for p in portfolios]
    plt.figure(figsize=(10, 8))
    plt.plot(x, y)
    plt.savefig('frontier.png')
    print("All done!")


if __name__ == "__main__":
    main()

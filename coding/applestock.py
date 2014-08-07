#!/usr/bin/env python -B -tt
# Apple Stock
# 
# I have an array stockPricesYesterday where:
# 
#     The indices are the time, as a number of minutes past trade opening time, which was 9:30am local time.
#     The values are the price of Apple stock at that time, in dollars.
#     For example, the stock cost $500 at 10:30am, so stockPricesYesterday[60] = 500.
#     Write an efficient algorithm for computing the best profit I could have made from 1 purchase and 1 sale 
#     of 1 Apple stock yesterday. For this problem, we won't allow "shorting"-you must buy before you sell.

def profit(stock_prices):
    """
    INPUT: list -- time series of yesterday's Apple stock prices
    OUTPUT: int -- maximum profit 
    """
    n = len(stock_prices)
    max_profit = 0
    for i in xrange(n - 1):                                     # O(n)
        buy_price = stock_prices[i]
        sell_price = max(stock_prices[i + 1:])                  # O(n) worst 
        max_profit = max([sell_price - buy_price, max_profit])
    print max_profit
    return max_profit

def test(stock_prices=[]):
    if not stock_prices:
        stock_prices = raw_input() # command line pipe stdin
        if stock_prices.find(",") == -1:
            stock_prices = eval(",".join(stock_prices.split()))
        else:
            stock_prices = eval(stock_prices)
    profit(stock_prices)


if __name__ == "__main__":
    test(stock_prices=[5, 2, 4, 3, 2])

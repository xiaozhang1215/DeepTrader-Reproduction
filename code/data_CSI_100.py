
import tushare as ts
import pandas as pd
import numpy as np
import os
import time

TOKEN = '68efd2b46808192e929a974c25c4e60aacce30e204443df0f79094c6'
ts.set_token(TOKEN)          # 设置全局 Token
pro = ts.pro_api()           # 初始化接口实例

# Get the list of CSI 100 constituent stocks
def get_csi_100_stocks(pro, index_code='000903.SH', start_date='20220101', end_date='20221231', top_n = 15):
    """
    Args:
    pro (tushare.pro.client.DataApi): Tushare Pro API instance.
    index_code (str, optional): The index code to query. Defaults to '000903.SH' (CSI 100).
    start_date (str, optional):'YYYYMMDD' ; 
    top_n (int or None, optional): Number of top stocks to select. If None, returns all stocks.
        
    Returns:
        list: A sorted list of constituent stock codes (ts_code).
    """
    try:
        weight_df = pro.index_weight(
            index_code=index_code, 
            start_date=start_date, 
            end_date=end_date)
        
        # Handle empty response to prevent subsequent errors
        if weight_df.empty: 
            print(f"No data found for index {index_code} between {start_date} and {end_date}.")
            return []
        
        # extract the stock codes and sort them
        stock_list = sorted(list(set(weight_df['con_code'].tolist())))

        # If top_n is specified, select the top_n stocks
        if top_n is not None:
            stock_list = stock_list[:top_n]
        num_stocks = len(stock_list)
        print(f"Number of constituent stocks retrieved: {num_stocks}")
        
        return stock_list
    
    # Handle exceptions and print the error message
    except Exception as e:
        print(f"Error retrieving CSI 100 stocks: {e}")
        return []

# practical usage example

my_stock_list = get_csi_100_stocks(
    pro, 
    index_code='000903.SH', 
    start_date='20210101', 
    end_date='20221231', 
    top_n=20
)
print("获取到的股票代码列表:", my_stock_list)

# Fetch stock data for the retrieved stock list
def get_stock_data(stock_list, 
                   start_date='20220101', 
                   end_date='20221231', 
                   sleep_time=0.3,
                   pro=ts.pro_api('68efd2b46808192e929a974c25c4e60aacce30e204443df0f79094c6')):
    """
    Args:
    stock_list (list): List of stock codes to retrieve data for.
    start_date (str, optional); end_date (str, optional): End date of the query period in 'YYYYMMDD' format.
    sleep_time (int, optional): Time to sleep between API calls to avoid rate limits. Defaults to 0.3 second.
        
    Returns:
        dict: A dictionary where keys are stock codes and values are DataFrames containing stock data.
    """

    if not stock_list:
        print("Stock list is empty. No data to retrieve.")
        return {}

    # Loop through each stock code and fetch its daily data
    stock_data_dict = {}

    for stock_code in stock_list:
        try:
            # Fetch daily stock data
            df = ts.pro_bar(ts_code=stock_code, 
                            api=pro,
                            adj= "qfq", 
                            start_date=start_date, 
                            end_date=end_date)
            
            # Handle empty response
            if df is None or df.empty:
                print(f"No data found for stock {stock_code} between {start_date} and {end_date}.")
            # Sort by trade date
            else:
                df = df.sort_values(by='trade_date')
                stock_data_dict[stock_code] = df
            
        except Exception as e:
            print(f"Error retrieving data for stock {stock_code}: {e}")
        finally:
            time.sleep(sleep_time)
    print(f"Data retrieval completed. Successfully retrieved {len(stock_data_dict)} out of {len(stock_list)} stocks.")
    return stock_data_dict

# Converts a dictionary into the required .npy matrices for DeepTrader.
def arrange_stock_data(stock_data_dict, save_dir = "./formatted_data"):
    if not stock_data_dict:
        print("Stock data dictionary is empty.")
        return
    
    # 1. Construct a global trading calendar
    all_dates = set()
    for df in stock_data_dict.values():
        all_dates.update(df['trade_date'].tolist())
    all_dates = sorted(list(all_dates))
    num_dates = len(all_dates)
    stock_list = list(stock_data_dict.keys())
    num_stocks = len(stock_list)

    # 2.difine the number of features
    num_asset_features = 2
    num_market_features = 3
    print(f"Data alignment started: {num_stocks} stocks, {num_dates} trading days.")

    # 3. Initialize matrices
    stock_metrics = np.zeros((num_stocks, num_dates, num_asset_features))
    ror_matrix = np.zeros((num_stocks, num_dates))

    # 4. Align data for each stock
    for i, stock_code in enumerate(stock_list):
        df = stock_data_dict[stock_code]
        df = df.set_index('trade_date')

        df_aligned = df.reindex(all_dates).ffill().bfill()

        for j, date in enumerate(all_dates):
            if pd.notna(df_aligned.loc[date, 'close']):
                close_price = df_aligned.loc[date, 'close']
                volume = df_aligned.loc[date, 'vol']
                
                # Populate asset features
                stock_metrics[i, j, 0] = close_price
                stock_metrics[i, j, 1] = volume / 1e6 
        
                # Calculate rate of return (RoR) for reward computation
                if j > 0 and stock_metrics[i, j-1, 0] > 0:
                    ror_matrix[i, j] = (close_price / stock_metrics[i, j-1, 0]) - 1

    # 5. Generate dummy market data and relation matrix
    market_data = np.random.randn(num_dates, num_market_features) * 0.01
    # Identity matrix for spatial GCN (Self-connection only) as a fallback
    relation_matrix = np.eye(num_stocks)

    # 6. Create directory and save matrices as .npy files
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'stocks_data.npy'), stock_metrics)
    np.save(os.path.join(save_dir, 'ror.npy'), ror_matrix)
    np.save(os.path.join(save_dir, 'market_data.npy'), market_data)
    np.save(os.path.join(save_dir, 'industry_classification.npy'), relation_matrix)
    
    # 7. Output verification shapes
    print(f"\nData arrangement completed. Files saved to: {save_dir}")
    print(f"  - stocks_data.npy: {stock_metrics.shape} (Expected: [{num_stocks}, {num_dates}, {num_asset_features}])")
    print(f"  - ror.npy: {ror_matrix.shape} (Expected: [{num_stocks}, {num_dates}])")
    print(f"  - market_data.npy: {market_data.shape} (Expected: [{num_dates}, {num_market_features}])")
    print(f"  - industry_classification.npy: {relation_matrix.shape} (Expected: [{num_stocks}, {num_stocks}])")    

# main function to demonstrate usage
if __name__ == "__main__":
    my_stock_list = get_csi_100_stocks(
        pro, 
        index_code='000903.SH', 
        start_date='20210101', 
        end_date='20221231', 
        top_n=20
    )
    print("获取到的股票代码列表:", my_stock_list)

    # Fetch stock data for the retrieved stock list
    my_stock_data = get_stock_data(
        my_stock_list,
        start_date='20210101',
        end_date='20221231'
    )
    if my_stock_data:
        # 你可以随意修改保存路径，这里默认保存在 ./formatted_data
        arrange_stock_data(my_stock_data, save_dir='./formatted_data')
    else:
        print("Failed to fetch stock data, skipping matrix conversion.")


    


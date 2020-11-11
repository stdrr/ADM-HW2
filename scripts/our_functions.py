import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from tqdm import tqdm
import datetime

import os, glob

DATA_PATH = './data/'

month_files = ['2019-Oct.csv', '2019-Nov.csv']
total_records = 109950743 


def get_needed_data(month_files:list, columns:list, parse_dates=False):
    global DATA_PATH
    dataframes = []
    date_parser = pd.to_datetime if parse_dates else None
    for month in month_files:
        dataframes.append(pd.read_csv(os.path.join(DATA_PATH, month), usecols=columns, parse_dates=parse_dates, date_parser=date_parser))
    return dataframes


def get_needed_iterator(month_files:list, columns:list, chunksize:int=10**6, parse_dates=False):
    iterators = []
    date_parser = pd.to_datetime if parse_dates else None
    for month in month_files:
        iterators.append(pd.read_csv(os.path.join(DATA_PATH, month), usecols=columns, parse_dates=parse_dates, date_parser=date_parser, chunksize=chunksize))
    return iterators


def load_data(month_files:list, columns:list, chunk=False, chunksize:int=10**6, parse_dates=False):
    if chunk:
        return get_needed_iterator(month_files, columns, chunksize, parse_dates)
    else:
        return get_needed_data(month_files, columns, parse_dates)


def compute_by_chunk(chunks, function, **kargs):
    results = []
    for df in tqdm(chunks):
        results.append(function(df, **kargs))
    return results


def clean_memory(garbage:list):
    size = len(garbage)
    for i in range(size-1, -1, -1):
        del garbage[i]


########## RQ1 ############################################################################################################

## ***** Which is the rate of complete funnels? (WORKS)
def complete_funnels(columns:list):
    global month_files, total_records
    purchases_list = []
    carts_list = []
    dataset_iterators = load_data(month_files, columns, chunk=True)
    for iterator in dataset_iterators:
        for dataset in iterator:
            purchases_list.append(dataset[dataset.event_type == 'purchase'])
            carts_list.append(dataset[dataset.event_type == 'cart'])
    purchases = pd.concat(purchases_list, ignore_index=True)
    clean_memory(purchases_list)
    carts = pd.concat(carts_list, ignore_index=True)
    clean_memory(carts_list)
    n_purchases = purchases.size
    n_carts = carts.size
    purchases_carts = purchases.merge(carts, on=['user_session', 'product_id'], how='left', indicator=True)
    clean_memory([purchases, carts])
    n_one_click = purchases_carts[purchases_carts._merge == 'left_only'].size
    clean_memory([purchases_carts, ])
    print(f'There are {n_purchases} purchases,\nof which {n_one_click} one_click purchases (only view-purchase couples).\nThe number of carts is {n_carts}.', end='\n'*3)
    # completed = (n_purchases - n_one_click) / (total_records - 3*(n_purchases - n_one_click) - 2*(n_carts + n_one_click - n_purchases) - 2*n_one_click) * 100
    completed = [0.0] * 2
    completed[0] = (n_purchases - n_one_click) / (total_records - n_purchases - n_carts) * 100 # completed without one_click purchases
    completed[1] = n_purchases / (total_records - n_purchases - n_carts) * 100 # completed with one_click purchases
    print(f'Rate of complete funnels without one_click purchases:\t{completed[0]}.\n\nRate of complete funnels with one_click purchases:\t{completed[1]}.')



## ***** What’s the operation users repeat more on average within a session? (WORKS)
def average_number_operations(columns:list):
    global month_files
    dataset_iterators = load_data(month_files, columns, chunk=True)
    sessions_list = []
    n_views, n_carts, n_purchases = (0,0,0)
    for iterator in dataset_iterators:
        for dataset in iterator:
            n_views += dataset[dataset.event_type == 'view'].size
            n_carts += dataset[dataset.event_type == 'cart'].size
            n_purchases += dataset[dataset.event_type == 'purchase'].size
            sessions_list.append(dataset['user_session'].drop_duplicates())
            clean_memory([dataset, ])
    n_sessions = pd.concat(sessions_list, ignore_index=True).drop_duplicates().size
    print(f'{n_views} views, {n_carts} carts, {n_purchases} purchases. {n_sessions} sessions.')
    plt.figure(figsize=(8,6))
    plt.bar(['views', 'carts', 'purchases'], [round(n_views / n_sessions, 2), round(n_carts / n_sessions, 2), round(n_purchases / n_sessions, 2)])
    plt.title('Average number for each operation within a session')
    plt.xlabel('Operation')
    plt.ylabel('Frequency')
    plt.show()

# def average_number_operations(columns:list):
#     global month_files
#     count_events_list = []
#     dataset_iterators = load_data(month_files, columns, chunk=True)
#     for iterator in dataset_iterators:
#         for dataset in iterator:
#             r = dataset.groupby(['user_session', 'event_type']).event_type.agg(num_event='count')
#             count_events_list.append(r)
#             del r
#     count_events = pd.concat(count_events_list).groupby(['user_session', 'event_type']).num_event.sum()
#     n_sessions = count_events.index.get_level_values('user_session').nunique()
#     average_n_op = count_events.groupby('event_type').sum().apply(lambda x: x / n_sessions)
#     average_n_op.plot.bar()



## ***** How many times, on average, a user views a product before adding it to the cart? (WORKS)
def average_views_before_cart(columns:list):
    global month_files
    carts_list = []
    views_carts_list = []
    dataset_iterators = load_data(month_files, columns, chunk=True, parse_dates=True)
    for iterator in dataset_iterators:
        for dataset in iterator:
            carts_list.append(dataset[dataset.event_type == 'cart'])
    carts = pd.concat(carts_list, ignore_index=True)
    clean_memory(carts_list)
    dataset_iterators = load_data(month_files, columns, chunk=True, parse_dates=True)
    for iterator in dataset_iterators:
        for dataset in iterator:
            views_per_chunk = dataset[dataset.event_type == 'view']
            views_carts_per_chunk = views_per_chunk.merge(carts, on=['user_id', 'product_id'], suffixes=['_view', '_cart'])
            del views_per_chunk
            views_carts_per_chunk = views_carts_per_chunk[views_carts_per_chunk.event_time_view < views_carts_per_chunk.event_time_cart]
            views_carts_list.append(views_carts_per_chunk)
            del views_carts_per_chunk
    views_carts = pd.concat(views_carts_list, ignore_index=True)
    clean_memory(views_carts_list)
    avg = views_carts.groupby(['user_id', 'product_id']).event_time_view.count().mean()
    clean_memory([views_carts,])
    print(f'On average a user views a product {round(avg, 2)} times before adding it to the cart.')



## ***** What’s the probability that products added once to the cart are effectively bought? (WORKS)
def probability_purchase(columns:list):
    global month_files
    carts_list = []
    purchases_list = []
    dataset_iterators = load_data(month_files, columns, chunk=True)
    for iterator in dataset_iterators:
        for dataset in iterator:
            carts_list.append(dataset[dataset.event_type == 'cart'])
            purchases_list.append(dataset[dataset.event_type == 'purchase'])
    carts = pd.concat(carts_list, ignore_index=True)
    clean_memory(carts_list)
    purchases = pd.concat(purchases_list, ignore_index=True)
    clean_memory(purchases_list)
    carts_purchases = carts.merge(purchases.drop_duplicates(), on=['user_session', 'product_id'], how='left', indicator=True)
    clean_memory([carts, purchases])
    n_purchased = carts_purchases[carts_purchases._merge == 'both'].size
    print(f'The probability that products added once to the cart are effectively bought is {round(n_purchased / carts_purchases.size * 100, 2)}.')



## ***** How much time passes on average between the first view time and a purchase/addition to cart?
def average_time_after_first_view(columns:list):
    global month_files
    user_views_p_min_list = []
    user_purchases_carts_p_min_list = []
    dataset_iterators = load_data(month_files, columns, chunk=True, parse_dates=True)
    for iterator in dataset_iterators:
        for dataset in iterator:
            user_views_p_min_list.append(dataset[dataset.event_type == 'view'].groupby('user_id').event_time.min())
            user_purchases_carts_p_min_list.append(dataset[(dataset.event_type == 'cart') | (dataset.event_type == 'purchase')].groupby('user_id').event_time.min())
            clean_memory([dataset, ])
    user_views_min = pd.concat(user_views_p_min_list).groupby('user_id').event_time.min()
    clean_memory(user_views_p_min_list)
    user_purchases_carts_min = pd.concat(user_purchases_carts_p_min_list).groupby('user_id').event_time.min()
    clean_memory(user_purchases_carts_p_min_list)
    joined = user_views_min.to_frame().merge(user_purchases_carts_min.to_frame(), on='user_id', suffixes=['_views', '_purchases_carts'])
    sum_deltas = 0
    for t_views, t_purchases_carts in zip(joined['event_time_views'], joined['event_time_purchases_carts']):
        sum_deltas += (t_purchases_carts - t_views).total_seconds()
    print(f'The average time between the first view time and a purchase/addition to cart is {round((sum_deltas / joined.size) / 3600, 2)} hours.') # we convert the time from seconds to hours




########## RQ4 ############################################################################################################

def get_purchases(columns:list):
    global month_files
    purchases_months_list = []
    for month in month_files:
        iterator = load_data([month, ], columns, chunk=True)[0]
        purchases_list = []
        for dataset in iterator:
            purchases_list.append(dataset[dataset.event_type == 'purchase'])
        purchases_months_list.append(pd.concat(purchases_list, ignore_index=True))
        clean_memory(purchases_list)
    return purchases_months_list



## ***** How much does each brand earn per month? (WORKS)
def get_profit_per_month(brand_name, dataset_list:list=None):
    global month_files
    if dataset_list == None:
        dataset_list = get_purchases(['brand', 'product_id', 'price', 'event_type'])
    profit_per_month = {}
    for i in range(len(month_files)):
        profit = 0
        month_name = month_files[i][5:8]
        brand_items = dataset_list[i][dataset_list[i].brand == brand_name]
        if not brand_items.empty:
            profit = brand_items[(brand_items.price.notna()) & (brand_items.price.notnull())].price.agg('sum')
        profit_per_month[month_name] = profit
    return profit_per_month



## ***** Is the average price of products of different brands significantly different? (WORKS)
def price_std_dev(columns:list):
    global month_files
    dataset_iterators = load_data(month_files, columns, chunk=True)
    brand_products_list = []
    for iterator in dataset_iterators:
        for dataset in iterator:
            brand_products_list.append(dataset.drop_duplicates())
    brand_products = pd.concat(brand_products_list, ignore_index=True).drop_duplicates()
    std = brand_products.groupby('brand').price.mean().std()
    print(f'The standard deviation of the average price of the products for each brand is {round(std, 2)}.')



## ***** Find the top 3 brands that have suffered the biggest losses in earnings between one month and the next (WORKS)
def top_n_two_months_losses(columns:list, month1:str, month2:str, n=3):
    dataset_list = get_purchases(columns)
    brands = pd.concat(dataset_list, ignore_index=True)
    brands = brands[brands.brand.notna()].brand.unique()
    diff_profit = dict.fromkeys(brands)
    for brand in brands:
        profit = get_profit_per_month(brand, dataset_list)
        diff_profit[brand] = (profit[month2] - profit[month1]) / (profit[month1] + 1) * 100
    diff_profit_list = diff_profit.items()
    diff_profit_list = sorted(diff_profit_list, key=lambda x: (x[1], x[0]))
    for i in range(n):
        print(f'The brand \"{diff_profit_list[i][0]}\" lost {round(diff_profit_list[i][1] * -1)} between {month1} and {month2}.')

            
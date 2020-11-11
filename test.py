import pandas as pd 

import scripts.our_functions as f


# f.complete_funnels(['event_type', 'user_session', 'product_id'])

# f.average_number_operations(['user_session', 'event_type'])

# f.average_views_before_cart(['user_id', 'product_id', 'event_type', 'event_time'])

# f.probability_purchase(['user_session', 'product_id', 'event_type', 'event_time'])

f.average_time_after_first_view(['event_type', 'event_time', 'user_id'])

# print(f.get_profit_per_month('elco'))

# f.price_std_dev(['brand', 'product_id', 'price'])

# f.top_n_two_months_losses(['brand', 'product_id', 'price', 'event_type'], 'Oct', 'Nov')
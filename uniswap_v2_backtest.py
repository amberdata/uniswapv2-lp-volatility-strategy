import pandas as pd
import numpy as np
from enum import Enum
import random
import volatility_calculation
import generate_signal

def get_random_signal(l):
    random.seed(2)
    random_list = random.sample(range(10, 100000-10), 500)
    random_list.sort()
    print(random_list)
    signal_dict = {}
    for i in range(len(random_list)):
        if i % 2 == 0:
            signal_dict[random_list[i]] = 'add'
        else:
            signal_dict[random_list[i]] = 'remove'
    return signal_dict


def add_signal_to_df(sd, df):
    df['signal'] = np.nan
    for i in sd:
        df.iloc[i, df.columns.get_loc('signal')] = sd[i]
    print(df)
    return df


class Mode(Enum):
    BACKTEST = "BACKTEST"
    LIVE = "LIVE"


class UniV2Strategy:
    def __init__(self, initial_usdc, swap_fee=0.003):
        self.df_historical_data = None
        self.mode = None
        self.do_print = None
        self.initial_usdc = initial_usdc
        self.swap_fee = swap_fee

        self.df_record = pd.DataFrame()
        self.row = None
        self.last_row = None
        self.trade_num = 0
        self.status = 0

    def log(self, txt):
        if self.do_print:
            print(txt)

    def historical_data_feed(self, df):
        self.df_historical_data = df

    def output_record(self):
        df = self.df_record
        start = int(df.timestamp.values[0])
        end = int(df.timestamp.values[-1])
        df = df.reset_index(drop=True)
        df.to_csv(f'{start}_{end}.csv')
        print(df)

    def record(self):
        price = self.row['pool_price']

        # holder amount
        if len(self.df_record) == 0:
            self.row['holder_eth_amount'] = self.initial_usdc / 2 / price
            self.row['holder_usdc_amount'] = self.initial_usdc / 2

        elif self.last_row['signal'] == 'remove':
            self.row['holder_eth_amount'] = self.last_row['lp_eth_amount']
            self.row['holder_usdc_amount'] = self.last_row['lp_usdc_amount']

        elif self.status == 1:
            self.row['holder_eth_amount'] = 0
            self.row['holder_usdc_amount'] = 0

        elif self.row['signal'] == 'add':
            eth_value = self.last_row['holder_eth_amount'] * price
            usdc_value = self.last_row['holder_usdc_amount']
            if eth_value > usdc_value:
                self.row['holder_eth_amount'] = self.last_row['holder_eth_amount'] - (
                            eth_value - usdc_value) / price / 2
                self.row['holder_usdc_amount'] = self.last_row['holder_usdc_amount'] + (eth_value - usdc_value) / 2
            elif eth_value < usdc_value:
                self.row['holder_eth_amount'] = self.last_row['holder_eth_amount'] + (
                            usdc_value - eth_value) / price / 2
                self.row['holder_usdc_amount'] = self.last_row['holder_usdc_amount'] - (usdc_value - eth_value) / 2

        elif self.status == 0:
            self.row['holder_eth_amount'] = self.last_row['holder_eth_amount']
            self.row['holder_usdc_amount'] = self.last_row['holder_usdc_amount']

        # lp amount / liquidity
        def cal_after_lp_amount():
            lp_usdc = (self.liquidity * price) ** (1 / 2)
            lp_eth = (self.liquidity / price) ** (1 / 2)
            return lp_eth, lp_usdc

        if len(self.df_record) == 0 or self.status == 0:
            self.row['lp_eth_amount'] = 0
            self.row['lp_usdc_amount'] = 0
            self.row['liquidity'] = 0

        elif self.last_row['signal'] == 'add':
            self.row['lp_eth_amount'] = self.last_row['holder_eth_amount']
            self.row['lp_usdc_amount'] = self.last_row['holder_usdc_amount']
            self.liquidity = self.row['lp_eth_amount'] * self.row['lp_usdc_amount']
            self.row['liquidity'] = self.liquidity

        elif self.status == 1:
            lp_eth, lp_usdc = cal_after_lp_amount()
            self.row['lp_eth_amount'] = lp_eth
            self.row['lp_usdc_amount'] = lp_usdc
            self.row['liquidity'] = self.last_row['liquidity']

        # fee amount
        if len(self.df_record) == 0 or self.status == 0 or self.last_row['signal'] == 'add':
            self.row['fee_eth_amount'] = 0
            self.row['fee_usdc_amount'] = 0
        elif self.status == 1:
            if self.row['pool_price'] > self.last_row['pool_price']:
                self.row['fee_eth_amount'] = 0
                self.row['fee_usdc_amount'] = (self.row['lp_usdc_amount'] - self.last_row['lp_usdc_amount']) * \
                                              self.swap_fee / (1-self.swap_fee)
            elif self.row['pool_price'] < self.last_row['pool_price']:
                self.row['fee_eth_amount'] = (self.row['lp_eth_amount'] - self.last_row['lp_eth_amount']) * \
                                             self.swap_fee / (1-self.swap_fee)
                self.row['fee_usdc_amount'] = 0
            else:
                self.row['fee_eth_amount'] = 0
                self.row['fee_usdc_amount'] = 0

        # status
        self.row['status'] = self.status

        # trade num
        if self.trade_num > 0 and self.status == 1:
            self.row['trade_num'] = self.trade_num

        # add new row
        new_df = pd.DataFrame([self.row.tolist()], columns=self.row.index.tolist())
        self.df_record = pd.concat([self.df_record, new_df])
        print('-' * 100)
        print('last')
        print(self.last_row)
        print('now')
        print(self.row)
        print('-' * 100)

        self.last_row = self.row

    def reset_params(self):
        if len(self.df_record) > 0:
            last_signal = self.last_row['signal']
            if last_signal == 'add':
                self.status = 1
                self.trade_num += 1
            elif last_signal == 'remove':
                self.status = 0

    def loop(self):
        self.reset_params()
        self.record()

    def run(self, mode=Mode, do_print=True):
        self.do_print = do_print
        self.mode = mode

        for index, row in self.df_historical_data.iterrows():
            self.row = row
            self.loop()


if __name__ == '__main__':
    historical_data = pd.read_csv('eth_dex_lp_root_table_100K_rows.csv', index_col=0)
    historical_data = volatility_calculation.main(historical_data)
    historical_data = generate_signal.main(historical_data)

    historical_data = historical_data[['timestamp', 'datetime', 'pool_price', 'signal']]

    simple_test = UniV2Strategy(initial_usdc=10000)
    simple_test.historical_data_feed(df=historical_data)
    simple_test.run(mode=Mode.BACKTEST)
    simple_test.output_record()
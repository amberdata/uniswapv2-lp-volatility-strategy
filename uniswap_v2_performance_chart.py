import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import Range1d, CrosshairTool, Span
from bokeh.layouts import gridplot


def cal_performance(df):
    df['fee_eth_amount_cum'] = df['fee_eth_amount'].cumsum()
    df['fee_usdc_amount_cum'] = df['fee_usdc_amount'].cumsum()

    df['cum_fees'] =  df['fee_eth_amount_cum'] * df['pool_price'] + df['fee_usdc_amount_cum']
    df['lp_value'] = (df['lp_eth_amount'] + df['holder_eth_amount']) * df['pool_price'] + df['lp_usdc_amount'] + df['holder_usdc_amount']
    df['lp_strategy_total_value'] = df['lp_value'] + df['cum_fees']

    df['holder_value'] = df['holder_eth_amount'].values[0] * df['pool_price'] + df['holder_usdc_amount'].values[0]

    df['IL'] = df['holder_value'] - df['lp_strategy_total_value']
    return df

def plot(df_performance, df_signal, trade_info, base_token_symbol, quote_token_symbol):
    base_token = base_token_symbol
    quote_token = quote_token_symbol

    all_datetime_range = np.array(df_performance["datetime"], dtype=np.datetime64)

    xdr = Range1d(start=all_datetime_range[0], end=all_datetime_range[-1])
    tools = ('undo', 'box_zoom', "wheel_zoom", "reset", "pan", 'crosshair', 'save')

    # # P_Position_Change(fees/LP)
    # P_PC = figure(title="Position Change", x_axis_type="datetime", x_axis_label="time",
    #               y_axis_label=f"{quote_token}(%)",
    #               height=250, sizing_mode="stretch_width", x_range=xdr, tools=tools)
    #
    # pc_set = {"time": [], "lp_pnl": [], "fees":[], "IL":[]}
    # for i in trade_info.values():
    #     time = i["end_datetime"]
    #     lp_pnl = i["LP_PNL(%)"]
    #     fees = i["fees(%)"]
    #     IL = -i["IL(%)"]
    #     pc_set["time"].append(time)
    #     pc_set["lp_pnl"].append(lp_pnl)
    #     pc_set["fees"].append(fees)
    #     pc_set["IL"].append(IL)
    #
    # x_t = pc_set["time"]
    # y_lp_pnl = pc_set["lp_pnl"]
    # y_fees = pc_set["fees"]
    # y_IL = pc_set["IL"]
    #
    # P_PC.circle(x_t, y_lp_pnl, legend_label="ΔLP", color="#F1948A", size=5)
    # P_PC.circle(x_t, y_fees, legend_label="Fees", color="#73C6B6", size=5)
    # P_PC.circle(x_t, y_IL, legend_label="IL", color="#A569BD", size=5)
    #
    # h_line = Span(location=0, dimension='width', line_color='#17202A', line_dash='dashed', line_width=1)
    # P_PC.add_layout(h_line)
    # P_PC.legend.click_policy = "hide"

    # P_kline
    y_price = df_performance["pool_price"]
    P_kline = figure(title="Price", x_axis_label="time", x_axis_type="datetime",
                     y_axis_label=f"{base_token}/{quote_token}", height=300, sizing_mode="stretch_width",
                     tools=tools, x_range=xdr)
    P_kline.line(all_datetime_range, y_price, legend_label="price", color="#515A5A", line_width=1.5)

    # entry exit price
    entry_exit_set = {"datetime": [], "price":[]}
    df_add = df_signal.loc[df['signal'] == 'add']
    df_remove = df_signal.loc[df['signal'] == 'remove']

    P_kline.triangle(np.array(df_add['datetime'], dtype=np.datetime64), df_add['pool_price'], legend_label='add', color="#1ABC9C", size=5)
    P_kline.inverted_triangle(np.array(df_remove['datetime'], dtype=np.datetime64), df_remove['pool_price'], legend_label='remove', color="#E74C3C", size=5)

    # plot vol
    y_vol = df_signal['vol']
    P_vol = figure(title="Volatility", x_axis_label="time", x_axis_type="datetime",
                     y_axis_label=f"vol", height=250, sizing_mode="stretch_width",
                     tools=tools, x_range=xdr)
    P_vol.line(all_datetime_range, y_vol, legend_label="volatility", color="#A569BD", line_width=1.5)

    # P_acount
    y_fees = df_performance["cum_fees"]
    y_lp = 100 * (df_performance["lp_strategy_total_value"] / df_performance['holder_value'].values[0] - 1)
    y_holder = 100 * (df_performance["holder_value"] / df_performance['holder_value'].values[0] - 1)
    y_diff = y_lp - y_holder

    P_account = figure(title="Performance", x_axis_label="time", y_axis_label=f"{quote_token}(%)",
                       height=250, sizing_mode="stretch_width", tools=tools, x_range=xdr)
    P_account.line(all_datetime_range, y_lp, legend_label="LP Strategy", color="#E74C3C", line_width=1.5)
    P_account.line(all_datetime_range, y_holder, legend_label="50/50 Holder", color="#F5B041", line_width=1.5)
    h_line = Span(location=0, dimension='width', line_color='#17202A', line_dash='dashed', line_width=1)
    P_account.add_layout(h_line)
    P_account.legend.click_policy = "hide"

    P_diff = figure(title="Difference", x_axis_label="time", y_axis_label=f"{quote_token}(%)",
                       height=250, sizing_mode="stretch_width", tools=tools, x_range=xdr)
    P_diff.line(all_datetime_range, y_diff, legend_label="LP-Holder", color="#16A085", line_width=1.5)
    h_line = Span(location=0, dimension='width', line_color='#17202A', line_dash='dashed', line_width=1)
    P_diff.add_layout(h_line)
    P_diff.legend.click_policy = "hide"

    plots = [P_kline, P_vol, P_account, P_diff]

    # crosshair
    crosshair = CrosshairTool(dimensions="height")
    for i in plots:
        i.add_tools(crosshair)

    # put together
    grid = gridplot(plots, toolbar_location='right', ncols=1)  # sizing_mode="stretch_width"
    show(grid)


if __name__ == '__main__':
    df = pd.read_csv('1640995220_1644345765.csv', index_col=0)
    df_signal = pd.read_csv('data_with_signal.csv', index_col=0)
    df_performance = cal_performance(df)
    plot(df_performance=df_performance, df_signal=df_signal, trade_info=None, base_token_symbol='ETH', quote_token_symbol='USDC')
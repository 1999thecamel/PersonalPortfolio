import pandas as pd
from tabulate import tabulate
import re as regex
import io
import numpy as np
import math
import util.dayTrade as dt
from base64 import b64encode as _b64encode

# 應用函數
# ============================================================================================


def comp(returns):
    """ Calculates total compounded returns """
    return returns.prod()


def group_returns(returns, groupby, compounded=False):
    """ summarize returns
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        return returns.groupby(groupby).apply(comp)
    return returns.groupby(groupby).sum()


# 獲得報表
# ============================================================================================
def attachProfit(df_target, day_cash, discount_fee, slippage_fee):
    if len(df_target) == 0:
        raise Exception('df_target長度須超過0')
    df_target = df_target.sort_index()
    df_target['交易次數'] = 1
    df_target['每日投資成本'] = day_cash

    setPosition = df_target['開盤價(元)']
    offsetPositon = df_target['收盤價(元)']
    longStopLoss = df_target['昨天收盤價(元)'] * 0.91
    shortStopLoss = df_target['昨天收盤價(元)'] * 1.09
    longStopLossCondition = df_target['最低價(元)'] <= longStopLoss
    shortStopLossCondition = df_target['最高價(元)'] >= shortStopLoss

    def add_mkt_return(grp):
        grp['當日交易次數'] = grp['交易次數'].sum()
        return grp

    df_target = df_target.groupby('年月日').apply(add_mkt_return)

    df_target['進場張數'] = ((day_cash / df_target['當日交易次數']) / (df_target['開盤價(元)'] * 1000)
                         ).apply(lambda value: value if math.floor(value) == 0 else math.floor(value))

    df_target['報酬 (多-無成本)'] = 0
    df_target['報酬 (空-無成本)'] = 0
    df_target['停損 (多-無成本)'] = 0
    df_target['停損 (空-無成本)'] = 0

    df_target['進場手續費'] = df_target['開盤價(元)'] * 0.001425 * discount_fee
    df_target['出場手續費'] = df_target['收盤價(元)'] * 0.001425 * discount_fee
    df_target['交易稅'] = df_target['收盤價(元)'] * 0.0015
    df_target['進場滑價'] = df_target['開盤價(元)'] * (slippage_fee / 100)
    df_target['出場滑價'] = df_target['收盤價(元)'] * (slippage_fee / 100)

    df_target['成本'] = (df_target['進場手續費'] + df_target['出場手續費'] + df_target['交易稅'] +
                       df_target['進場滑價'] + df_target['出場滑價']) * df_target['進場張數'] * 1000

    # 報酬 (多-無成本)
    # 停損
    df_target.loc[longStopLossCondition, '報酬 (多-無成本)'] = (longStopLoss[longStopLossCondition] -
                                                          setPosition[longStopLossCondition]) * df_target.loc[longStopLossCondition, '進場張數'] * 1000
    df_target.loc[longStopLossCondition, '停損 (多-無成本)'] = (longStopLoss[longStopLossCondition] -
                                                          setPosition[longStopLossCondition]) * df_target.loc[longStopLossCondition, '進場張數'] * 1000

    # # 正常
    df_target.loc[~longStopLossCondition, '報酬 (多-無成本)'] = (offsetPositon[~longStopLossCondition] -
                                                           setPosition[~longStopLossCondition]) * df_target.loc[~longStopLossCondition, '進場張數'] * 1000

    # 報酬 (空-無成本)
    # 停損
    df_target.loc[shortStopLossCondition, '報酬 (空-無成本)'] = -1 * (shortStopLoss[shortStopLossCondition] -
                                                                setPosition[shortStopLossCondition]) * df_target.loc[shortStopLossCondition, '進場張數'] * 1000
    df_target.loc[shortStopLossCondition, '停損 (空-無成本)'] = -1 * (shortStopLoss[shortStopLossCondition] -
                                                                setPosition[shortStopLossCondition]) * df_target.loc[shortStopLossCondition, '進場張數'] * 1000
    # 正常
    df_target.loc[~shortStopLossCondition, '報酬 (空-無成本)'] = -1 * (offsetPositon[~shortStopLossCondition] -
                                                                 setPosition[~shortStopLossCondition]) * df_target.loc[~shortStopLossCondition, '進場張數'] * 1000

    # df_target = df_target.set_index('年月日')
    df_target['報酬 (多-含成本)'] = df_target['報酬 (多-無成本)'] - df_target['成本']
    df_target['報酬 (多-含成本)'] = df_target['報酬 (多-含成本)'].apply(lambda x: '%.6f' % x)

    df_target['報酬 (空-含成本)'] = df_target['報酬 (空-無成本)'] - df_target['成本']
    # df_target = df_target.sort_values(by=['年月日', '證券代碼'])
    # df_target.insert(0, '年月日', df_target.index)
    # df_target = df_target.reset_index(drop=True)

    df_target['報酬 (多-無成本)'] = pd.to_numeric(df_target['報酬 (多-無成本)'])
    df_target['報酬 (空-無成本)'] = pd.to_numeric(df_target['報酬 (空-無成本)'])
    df_target['報酬 (多-含成本)'] = pd.to_numeric(df_target['報酬 (多-含成本)'])
    df_target['報酬 (空-含成本)'] = pd.to_numeric(df_target['報酬 (空-含成本)'])

    return df_target.sort_values('年月日')


def getDayProfit(df_target):
    df_dayProfit = pd.DataFrame()
    df_dayProfit['總交易次數'] = df_target.groupby('年月日')['交易次數'].sum()
    df_dayProfit['當日報酬 (多-無成本)'] = df_target.groupby('年月日')['報酬 (多-無成本)'].sum()
    df_dayProfit['當日報酬 (空-無成本)'] = df_target.groupby('年月日')['報酬 (空-無成本)'].sum()
    df_dayProfit['當日報酬 (多-含成本)'] = df_target.groupby('年月日')['報酬 (多-含成本)'].sum()
    df_dayProfit['當日報酬 (空-含成本)'] = df_target.groupby('年月日')['報酬 (空-含成本)'].sum()

    df_dayProfit['當日停損 (多-無成本)'] = df_target.groupby('年月日')['停損 (多-無成本)'].sum()
    df_dayProfit['當日停損 (空-無成本)'] = df_target.groupby('年月日')['停損 (空-無成本)'].sum()

    # df_dayProfit['當日停損 (多-含成本)']  = df_target.groupby('年月日')['停損 (多-無成本)'].sum() / df_dayProfit['總交易次數'] - fee / df_dayProfit['總交易次數']
    # df_dayProfit['當日停損 (空-含成本)']  = df_target.groupby('年月日')['停損 (空-無成本)'].sum() / df_dayProfit['總交易次數'] - fee / df_dayProfit['總交易次數']

    df_dayProfit.index = pd.to_datetime(df_dayProfit.index)
    return df_dayProfit


def getMonthProfit(df_dayProfit):
    df_monthGroup = df_dayProfit.groupby(
        df_dayProfit.index.strftime('%Y-%m-01'))
    df_monthProfit = pd.DataFrame()
    df_monthProfit['總交易次數'] = df_monthGroup['總交易次數'].sum()
    df_monthProfit['交易天數'] = df_monthGroup['當日報酬 (空-無成本)'].size()
    df_monthProfit['當月報酬 (多-無成本)'] = df_monthGroup['當日報酬 (多-無成本)'].sum()
    df_monthProfit['當月報酬 (空-無成本)'] = df_monthGroup['當日報酬 (空-無成本)'].sum()
    df_monthProfit['當月報酬 (多-含成本)'] = df_monthGroup['當日報酬 (多-含成本)'].sum()
    df_monthProfit['當月報酬 (空-含成本)'] = df_monthGroup['當日報酬 (空-含成本)'].sum()
    df_monthProfit.index = pd.to_datetime(df_monthProfit.index)
    return df_monthProfit


def getYearProfit(df_monthProfit):
    df_yearProfit = pd.DataFrame()
    df_yearGroup = df_monthProfit.groupby(df_monthProfit.index.strftime('%Y'))
    df_yearProfit['總交易次數'] = df_yearGroup['總交易次數'].sum()
    df_yearProfit['交易天數'] = df_yearGroup['交易天數'].sum()
    df_yearProfit['當年報酬 (多-無成本)'] = df_yearGroup['當月報酬 (多-無成本)'].sum()
    df_yearProfit['當年報酬 (空-無成本)'] = df_yearGroup['當月報酬 (空-無成本)'].sum()
    df_yearProfit['當年報酬 (多-含成本)'] = df_yearGroup['當月報酬 (多-含成本)'].sum()
    df_yearProfit['當年報酬 (空-含成本)'] = df_yearGroup['當月報酬 (空-含成本)'].sum()
    df_yearProfit['空 Sharpe'] = (
        df_yearGroup['當月報酬 (多-含成本)'].mean() / df_yearGroup['當月報酬 (多-含成本)'].std()).apply(dt.dRound)
    df_yearProfit['多 Sharpe'] = (
        df_yearGroup['當月報酬 (空-含成本)'].mean() / df_yearGroup['當月報酬 (空-含成本)'].std()).apply(dt.dRound)
    return df_yearProfit


def getTotalAnalyze(df_target):
    blank = ''
    df_dayProfit = getDayProfit(df_target)
    df_analysis = pd.Series()

    day_cash = df_target.iloc[1]['每日投資成本']
    long_MDD = abs((df_dayProfit['當日報酬 (多-含成本)'].cumsum() -
                   df_dayProfit['當日報酬 (多-含成本)'].cumsum().cummax()).min())
    short_MDD = abs((df_dayProfit['當日報酬 (空-含成本)'].cumsum() -
                    df_dayProfit['當日報酬 (空-含成本)'].cumsum().cummax()).min())

    df_target['報酬 (多-無成本)'] = pd.to_numeric(df_target['報酬 (多-無成本)'])
    df_target['報酬 (空-無成本)'] = pd.to_numeric(df_target['報酬 (空-無成本)'])

    df_analysis['每日投資成本'] = day_cash
    df_analysis['紅K數'] = len(
        df_target[df_target['收盤價(元)'] - df_target['開盤價(元)'] > 0])
    df_analysis['綠K數'] = len(
        df_target[df_target['收盤價(元)'] - df_target['開盤價(元)'] < 0])
    df_analysis['十字K數'] = len(
        df_target[df_target['收盤價(元)'] - df_target['開盤價(元)'] == 0])
    df_analysis['總交易次數'] = df_analysis['紅K數'] + \
        df_analysis['綠K數'] + df_analysis['十字K數']

    df_analysis['夏普值 (多-含成本)'] = getSharpe(df_dayProfit['當日報酬 (多-含成本)'])
    df_analysis['夏普值 (空-含成本)'] = getSharpe(df_dayProfit['當日報酬 (空-含成本)'])

    df_analysis['~~~~~~~~~~~~~'] = blank

    df_analysis['總報酬 (多-無成本)'] = df_dayProfit['當日報酬 (多-無成本)'].sum()
    df_analysis['總報酬 (多-含成本)'] = df_dayProfit['當日報酬 (多-含成本)'].sum()
    df_analysis['總報酬 (空-無成本)'] = df_dayProfit['當日報酬 (空-無成本)'].sum()
    df_analysis['總報酬 (空-含成本)'] = df_dayProfit['當日報酬 (空-含成本)'].sum()

    df_analysis['年化報酬 (多)[投資成本+2MDD] %'] = ((1 + df_dayProfit['當日報酬 (多-含成本)'].sum() /
                                             (2*long_MDD + day_cash)) ** (252/len(df_dayProfit)) - 1) * 100
    df_analysis['年化報酬 (空)[投資成本+2MDD] %'] = ((1 + df_dayProfit['當日報酬 (空-含成本)'].sum() /
                                             (2*short_MDD + day_cash)) ** (252/len(df_dayProfit)) - 1) * 100

    # df_analysis['年化報酬 (空-含成本) (2 MDD)'] = (1 + df_dayProfit['當日報酬 (空-含成本)'].sum() / (2*MDD)) ** (252/len(df_dayProfit)) - 1
    df_analysis['~~~~~~~~~~~~~~~~~~~~~~'] = blank
    df_analysis['平均單筆獲利 (多-含成本)'] = df_dayProfit['當日報酬 (多-含成本)'].sum() / \
        df_analysis['總交易次數']
    df_analysis['平均單日獲利 (多-含成本)'] = df_dayProfit['當日報酬 (多-含成本)'].sum() / \
        len(df_dayProfit)
    df_analysis['平均單筆獲利 (空-含成本)'] = df_dayProfit['當日報酬 (空-含成本)'].sum() / \
        df_analysis['總交易次數']
    df_analysis['平均單日獲利 (空-含成本)'] = df_dayProfit['當日報酬 (空-含成本)'].sum() / \
        len(df_dayProfit)

    df_analysis['~~~~~~'] = blank
    
    df_analysis['平均單筆獲利 (多-無成本)'] = df_dayProfit['當日報酬 (多-無成本)'].sum() / \
        df_analysis['總交易次數']
    df_analysis['平均單日獲利 (多-無成本)'] = df_dayProfit['當日報酬 (多-無成本)'].sum() / \
        len(df_dayProfit)
    df_analysis['平均單筆獲利 (空-無成本)'] = df_dayProfit['當日報酬 (空-無成本)'].sum() / \
        df_analysis['總交易次數']
    df_analysis['平均單日獲利 (空-無成本)'] = df_dayProfit['當日報酬 (空-無成本)'].sum() / \
        len(df_dayProfit)
    
    
    df_analysis['~~~~~~~~~~~~~~'] = blank
    
    winRate_long = len(df_target[df_target['報酬 (多-含成本)']
                                 > 0]) / len(df_target)
    winRate_short = len(df_target[df_target['報酬 (空-含成本)']
                                  > 0]) / len(df_target)

    df_analysis['均筆賺 (多-含成本)'] = df_target.loc[df_target['報酬 (多-含成本)'] > 0, '報酬 (多-含成本)'].sum() / \
        len(df_target.loc[df_target['報酬 (多-含成本)'] > 0, '報酬 (多-含成本)'])
    df_analysis['均筆賠 (多-含成本)'] = df_target.loc[df_target['報酬 (多-含成本)'] <= 0, '報酬 (多-含成本)'].sum() / \
        len(df_target.loc[df_target['報酬 (多-含成本)'] <= 0, '報酬 (多-含成本)'])
    df_analysis['賺賠比 (多-含成本)'] = abs(df_analysis['均筆賺 (多-含成本)'] /
                                     df_analysis['均筆賠 (多-含成本)'])
    df_analysis['凱利值 (多-含成本)'] = (winRate_long *
                                  (df_analysis['賺賠比 (多-含成本)'] + 1) - 1) / df_analysis['賺賠比 (多-含成本)']
    # len(df_dayProfit)

    df_analysis['均筆賺 (空-含成本)'] = df_target.loc[df_target['報酬 (空-含成本)'] > 0, '報酬 (空-含成本)'].sum() / \
        len(df_target.loc[df_target['報酬 (空-含成本)'] > 0, '報酬 (空-含成本)'])
    df_analysis['均筆賠 (空-含成本)'] = df_target.loc[df_target['報酬 (空-含成本)'] <= 0, '報酬 (空-含成本)'].sum() / \
        len(df_target.loc[df_target['報酬 (空-含成本)'] <= 0, '報酬 (空-含成本)'])
    df_analysis['賺賠比 (空-含成本)'] = abs(df_analysis['均筆賺 (空-含成本)'] /
                                     df_analysis['均筆賠 (空-含成本)'])
    df_analysis['凱利值 (空-含成本)'] = (winRate_short *
                                  (df_analysis['賺賠比 (空-含成本)'] + 1) - 1) / df_analysis['賺賠比 (空-含成本)']

    df_analysis['~~~~~~~~~~~~~~~'] = blank
    # 勝率
    df_analysis['勝率 (多-含成本)'] = winRate_long * 100
    df_analysis['勝率 (空-含成本)'] = winRate_short * 100
    
    df_analysis['勝率 (多-無成本)'] = len(df_target[df_target['報酬 (多-無成本)'] > 0]) / len(df_target) * 100
    df_analysis['勝率 (空-無成本)'] = len(df_target[df_target['報酬 (空-無成本)'] > 0]) / len(df_target) * 100

    df_analysis['總停損 (多-無成本)'] = df_dayProfit['當日停損 (多-無成本)'].sum()
    df_analysis['總停損 (空-無成本)'] = df_dayProfit['當日停損 (空-無成本)'].sum()
    # df_analysis['總停損 (多-含成本)'] = df_target['停損 (多-含成本)'].sum()
    # df_analysis['總停損 (空-含成本)'] = df_target['停損 (空-含成本)'].sum()
    # df_analysis = pd.to_numeric(df_analysis)

    def convertFn(value):
        if type(value) == str:
            return value
        return dt.dRound(value)
    df_analysis = df_analysis.map(convertFn).to_frame()
    df_analysis = df_analysis.T
    df_analysis.columns = [
        col if '~' not in col else '' for col in df_analysis.columns]
    df_analysis.columns = [
        col[:-1] if '%' in col else col for col in df_analysis.columns]
    df_analysis = df_analysis.T
    return df_analysis


def getSharpe(returns, nDays=252):
    return dt.dRound(
        (returns.mean() / returns.std()) * np.sqrt(nDays))


def metrics(returns, benchmark=None, rf=0, ):
    blank = ['']
    df_returns = pd.DataFrame({"returns": returns})

    drawDown = returns.cumsum() - returns.cumsum().cummax()
    metrics = pd.DataFrame()

    s_start = {'returns': df_returns['returns'].index.strftime('%Y-%m-%d')[0]}
    s_end = {'returns': df_returns['returns'].index.strftime('%Y-%m-%d')[-1]}
    s_rf = {'returns': rf}

    metrics['開始日期'] = pd.Series(s_start)
    metrics['結束日期'] = pd.Series(s_end)
    metrics['零風險利率%'] = pd.Series(s_rf)
    # metrics['存在市場率'] = '%.3f' % (len(returns) / len(set(df['年月日']))* 100)
    metrics['策略總報酬'] = '%.3f' % returns.sum()

    metrics['~~~~~~~~~~~~~~'] = blank

    metrics['Sharpe'] = getSharpe(returns)

    # metrics['Sortino'] = _stats.sortino(df, rf)
    # metrics['Sortino/√2'] = metrics['Sortino'] / _sqrt(2)

    metrics['~~~~~~~~'] = blank
    metrics['Max Drawdown %'] = '%.3f' % drawDown.min()
    metrics['Longest DD Days'] = blank
    metrics['Skew'] = '%.3f' % returns.skew()
    metrics['Kurtosis'] = '%.3f' % returns.kurtosis()

    metrics['~~~~~~~~~~'] = blank
    metrics['風報比'] = '%.3f' % (returns.sum() / abs(drawDown.min()))
    metrics['獲利因子'] = '%.3f' % (
        abs(returns[returns >= 0].sum() / returns[returns < 0].sum()))
    metrics['~~~~~~~'] = blank

    # best/worst
    metrics['Best Day %'] = '%.3f' % (returns.max())
    metrics['Worst Day %'] = '%.3f' % (returns.min())
    metrics['Best Month %'] = '%.3f' % (
        returns.groupby(returns.index.month).sum().max())
    metrics['Worst Month %'] = '%.3f' % (
        returns.groupby(returns.index.month).sum().min())
    metrics['Best Year %'] = '%.3f' % (
        returns.groupby(returns.index.year).sum().max())
    metrics['Worst Year %'] = '%.3f' % (
        returns.groupby(returns.index.year).sum().min())

    metrics.columns = [
        col if '~' not in col else '' for col in metrics.columns]
    metrics.columns = [
        col[:-1] if '%' in col else col for col in metrics.columns]
    metrics = metrics.T
    return metrics


def getDrawDownDetail(returns):
    drawdown = returns.cumsum() - returns.cumsum().cummax()
    no_dd = drawdown == 0

    # extract dd start dates
    starts = ~no_dd & no_dd.shift(1)
    starts = list(starts[starts].index)

    # extract end dates
    ends = no_dd & (~no_dd).shift(1)
    ends = list(ends[ends].index)

    # # no drawdown :)
    if not starts:
        return pd.DataFrame(
            index=[], columns=('start', 'valley', 'end', 'days',
                               'max drawdown'))

    # drawdown series begins in a drawdown
    if ends and starts[0] > ends[0]:
        starts.insert(0, drawdown.index[0])

    # series ends in a drawdown fill with last date
    if not ends or starts[-1] > ends[-1]:
        ends.append(drawdown.index[-1])

    data = []
    for i, _ in enumerate(starts):
        dd = drawdown[starts[i]:ends[i]]
        data.append((starts[i], dd.idxmin(), ends[i],
                     (ends[i] - starts[i]).days, (dd.idxmin() - starts[i]).days, dd.min()))

    df = pd.DataFrame(data=data,
                      columns=('start', 'valley', 'end', 'days', 'DD Longest Day',
                               'max drawdown'))
    df['days'] = df['days'].astype(int)
    df['max drawdown'] = df['max drawdown'].apply(dt.dRound)

    df['start'] = df['start'].dt.strftime('%Y-%m-%d')
    df['end'] = df['end'].dt.strftime('%Y-%m-%d')
    df['valley'] = df['valley'].dt.strftime('%Y-%m-%d')
    return df


def getBestAndWorstProfit(returns):
    df = returns.nlargest(10).append(returns.nsmallest(
        10)).sort_values(ascending=False).to_frame()
    return df


def _embed_figure(figfile, figfmt):
    figbytes = figfile.getvalue()
    if figfmt == 'svg':
        return figbytes.decode()
    data_uri = _b64encode(figbytes).decode()
    return '<img src="data:image/{};base64,{}" />'.format(figfmt, data_uri)


def _html_table(obj, showindex="default"):
    obj = tabulate(obj, headers="keys", tablefmt='html',
                   floatfmt=".2f", showindex=showindex)
    obj = obj.replace(' style="text-align: right;"', '')
    obj = obj.replace(' style="text-align: left;"', '')
    obj = obj.replace(' style="text-align: center;"', '')
    obj = regex.sub('<td> +', '<td>', obj)
    obj = regex.sub(' +</td>', '</td>', obj)
    obj = regex.sub('<th> +', '<th>', obj)
    obj = regex.sub(' +</th>', '</th>', obj)
    return obj


def generateHTML(df_target, types=(1, 1), output='./test.html', title='績效報表', figfmt='svg', condition="", df_benchmark=None):
    tpl = ""
    with open(dt.getRootPath() + '\\util\\daytrade.html') as f:
        tpl = f.read()
        f.close()
    df_dayProfit = getDayProfit(df_target)
    # long_returns = df_dayProfit['當日報酬 (多-含成本)']
    # short_returns = df_dayProfit['當日報酬 (空-含成本)']

    tradeType = '多' if types[0] == 1 else '空'
    hasFee = '含' if types[1] == 1 else '無'
    title = '報酬 ({}-{}成本)'.format(tradeType, hasFee)
    returns = df_dayProfit['當日報酬 ({}-{}成本)'.format(tradeType, hasFee)]
    returns_cum = returns.cumsum()

    date_range = returns.index.strftime('%Y{}%m{}%d{}'.format('年', '月', '日'))
    tpl = tpl.replace('{{date_range}}', date_range[0] + ' - ' + date_range[-1])
    tpl = tpl.replace('{{title}}', title)

    tpl = tpl.replace('{{condition}}', condition)
    mtrx = metrics(returns)
    mtrx.index.name = 'Metric'
    tpl = tpl.replace('{{metrics}}', _html_table(mtrx))
    tpl = tpl.replace('<tr><td></td><td></td><td></td></tr>',
                      '<tr><td colspan="3"><hr></td></tr>')
    tpl = tpl.replace('<tr><td></td><td></td></tr>',
                      '<tr><td colspan="2"><hr></td></tr>')

    df_monthProfit = getMonthProfit(df_dayProfit)

    month_profit = df_monthProfit.copy()
    month_profit.index.name = 'date'
    month_profit.index = month_profit.index.strftime('%Y-%m-%d')
    month_profit = month_profit[[
        '總交易次數', '交易天數', '當月報酬 (多-含成本)', '當月報酬 (空-含成本)']]
    tpl = tpl.replace('{{month_profit}}', _html_table(month_profit))

    year_profit = getYearProfit(df_monthProfit)
    year_profit.index.name = 'date'
    # year_profit.index = year_profit.index.strftime('%Y-%m-%d')
    df_yearGroup = returns.groupby(returns.index.strftime('%Y'))
    year_profit = year_profit[[
        '總交易次數', '交易天數', '當年報酬 (多-含成本)', '當年報酬 (空-含成本)']]
    year_profit['Sharpe'] = (
        (df_yearGroup.mean() / df_yearGroup.std()) * np.sqrt(252)).apply(dt.dRound)
    tpl = tpl.replace('{{year_profit}}', _html_table(year_profit))

    best_and_worst_trade = getBestAndWorstProfit(df_target[title])
    best_and_worst_trade = best_and_worst_trade.reset_index('證券代碼')
    best_and_worst_trade.index = best_and_worst_trade.index.strftime(
        '%Y-%m-%d')
    tpl = tpl.replace('{{best_and_worst_trade}}',
                      _html_table(best_and_worst_trade))

    best_and_worst_day = getBestAndWorstProfit(returns)
    best_and_worst_day.index = best_and_worst_day.index.strftime('%Y-%m-%d')
    tpl = tpl.replace('{{best_and_worst_day}}',
                      _html_table(best_and_worst_day))
    # tpl = tpl.replace('<tr><td></td><td></td><td></td></tr>',
    #                     '<tr><td colspan="3"><hr></td></tr>')
    # tpl = tpl.replace('<tr><td></td><td></td></tr>',
    #                     '<tr><td colspan="2"><hr></td></tr>')
    total_analyze = getTotalAnalyze(df_target)
    # total_analyze = total_analyze
    total_analyze.index.name = 'Metric'
    total_analyze.columns = ['數值']
    tpl = tpl.replace('{{total_analyze}}', _html_table(total_analyze))
    tpl = tpl.replace('<tr><td></td><td></td><td></td></tr>',
                      '<tr><td colspan="3"><hr></td></tr>')
    tpl = tpl.replace('<tr><td></td><td></td></tr>',
                      '<tr><td colspan="2"><hr></td></tr>')

    dd_info = getDrawDownDetail(returns).set_index('start').sort_values(
        by='max drawdown', ascending=True)[:10]
    tpl = tpl.replace('{{dd_info}}', _html_table(dd_info))

    # 畫圖
    figfile = io.BytesIO()
    dt.plotCumProAndDD(returns, title=title, benchmark=df_benchmark, savefig={
                       'fname': figfile, 'format': figfmt}, show=False, figsize=(8, 5))
    tpl = tpl.replace('{{returns}}', _embed_figure(figfile, figfmt))

    figfile = io.BytesIO()
    dt.plotDayProfit(returns, title=title, savefig={
                     'fname': figfile, 'format': figfmt}, show=False, figsize=(8, 5))
    tpl = tpl.replace('{{returns_plot}}', _embed_figure(figfile, figfmt))

    figfile = io.BytesIO()
    dt.plot_histogram(df_target[title], title=title, savefig={
                      'fname': figfile, 'format': figfmt}, show=False, figsize=(8, 5), bins=40)
    tpl = tpl.replace('{{monthly_dist}}', _embed_figure(figfile, figfmt))

    figfile = io.BytesIO()

    dt.plot_monthlyTradeTimes_heatmap(df_dayProfit['總交易次數'], title='月交易次數', savefig={
                                      'fname': figfile, 'format': figfmt}, show=False, figsize=(8, 5))
    tpl = tpl.replace('{{trade_heatmap}}', _embed_figure(figfile, figfmt))

    figfile = io.BytesIO()

    dt.plot_monthlyProfit_heatmap(returns, title='月' + title, savefig={
        'fname': figfile, 'format': figfmt}, show=False, figsize=(8, 5))
    tpl = tpl.replace('{{monthly_heatmap}}', _embed_figure(figfile, figfmt))

    figfile = io.BytesIO()
    with open(output, 'w', encoding='utf-8') as f:
        f.write(tpl)


class Backtest:
    def __init__(self, df, dayCash=1_000_000, df_commission=pd.DataFrame(), stockCash=0):
        self.df = df.copy()
        self.df_commission = df_commission
        self.dayCash = dayCash
        self.stockCash = stockCash

    def run(self, tradeColumn, stopPrice={'做多停損': -np.inf, '做空停損': np.inf}, df_benchmark=None):
        df = self.df
        df['交易次數'] = 1
        
        df_trade = pd.DataFrame()
        df_trade['進場價'] = tradeColumn['進場']
        df_trade['出場價'] = tradeColumn['出場']
        df_trade['做多停損條件'] = df['最低價(元)'] <= stopPrice['做多停損']
        df_trade['做空停損條件'] = df['最高價(元)'] >= stopPrice['做空停損']

        def add_mkt_return(grp):
            grp['當日交易次數'] = grp['交易次數'].sum()
            return grp
        self.df = df = df.groupby('年月日').apply(add_mkt_return)
        
        df['每日投資成本'] = self.dayCash if self.dayCash != 0 else 0
        df['每隻投入成本'] = (self.dayCash / df['當日交易次數']) if self.stockCash == 0 else self.stockCash
        df['進場張數'] = (df['每隻投入成本'] / (df_trade['進場價'] * 1000)
                      ).apply(lambda value: value if math.floor(value) == 0 else math.floor(value))
        df['報酬 (多-無成本)'] = 0
        df['報酬 (空-無成本)'] = 0
        df['停損 (多-無成本)'] = 0
        df['停損 (空-無成本)'] = 0

        df['成本'] = self.df_commission.sum(axis=1) * df['進場張數'] * 1000

        # 報酬 (多-無成本)
        df['報酬 (多-無成本)'] = np.where(df_trade['做多停損條件'],
                                    (stopPrice['做多停損'] - df_trade['進場價']
                                     ) * df['進場張數'] * 1000,
                                    (df_trade['出場價'] - df_trade['進場價']
                                     ) * df['進場張數'] * 1000,
                                    )
        df['報酬 (空-無成本)'] = np.where(df_trade['做空停損條件'],
                                    -1 *
                                    (stopPrice['做空停損'] - df_trade['進場價']
                                     ) * df['進場張數'] * 1000,
                                    -1 *
                                    (df_trade['出場價'] - df_trade['進場價']
                                     ) * df['進場張數'] * 1000,
                                    )
        df.loc[df_trade['做多停損條件'],
               '停損 (多-無成本)'] = (stopPrice['做多停損'] - df_trade['進場價']) * df['進場張數'] * 1000
        df.loc[df_trade['做空停損條件'], '停損 (空-無成本)'] = -1 * (
            stopPrice['做空停損'] - df_trade['進場價']) * df['進場張數'] * 1000

        df['報酬 (多-含成本)'] = df['報酬 (多-無成本)'] - df['成本']
        df['報酬 (多-含成本)'] = df['報酬 (多-含成本)'].apply(lambda x: '%.6f' % x)
        df['報酬 (空-含成本)'] = df['報酬 (空-無成本)'] - df['成本']


        df['報酬 (多-無成本)'] = pd.to_numeric(df['報酬 (多-無成本)'])
        df['報酬 (空-無成本)'] = pd.to_numeric(df['報酬 (空-無成本)'])
        df['報酬 (多-含成本)'] = pd.to_numeric(df['報酬 (多-含成本)'], errors='coerce')
        df['報酬 (空-含成本)'] = pd.to_numeric(df['報酬 (空-含成本)'], errors='coerce')
        return df

    def generateHTML(self, types=(-1, 1), output='./報表.html', df_benchmark=None, title='報表'):
        generateHTML(self.df, types=types, output=output, df_benchmark=df_benchmark, title=title)
# def oneFactorOptimize(df_target, conditionFn, target, linspace):
#     df_final = pd.DataFrame()
#     for i in linspace:
#         try:
#             df_profit = df_target
#             df_profit = conditionFn(df_profit, i)
#             df_profit = attachProfit(df_profit, day_cash, discount_fee, slippage_fee )
#             df_final.loc[i, target] = (dt.getTotalAnalyze(df_profit)).loc[target, 0]
#         except:
#             print('fail', i)
#     return df_final

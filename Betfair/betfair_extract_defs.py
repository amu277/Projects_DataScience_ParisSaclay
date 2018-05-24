import json
import os
from os import path
import glob
import pandas as pd
import numpy as np
import timeit

#### NOTES ####
# Keys per file:

# pt = published Time (in millis since epoch) (int)
# mc = market change, changes to market prices, runners or market definition - can be disregarded
# op = operation type (string)
# clk = sequence token - *can be disregarded as part of this data set* (string)

# https://historicdata.betfair.com/Betfair-Historical-Data-Feed-Specification.pdf

# when 'turnInPlayEnabled': True
# status: active, inplay: false
# status: active, inplay: true <- choose one before ltp

# store by team, date, country code
# ltp for each of the two teams and the draw -> odds conversion

# Get event and market information
def get_event_info(mkinfo, mkl):
    for i in range(0, mkl):
        mlist = mkinfo[i]['mc']
        keys = ['eventId', 'eventName', 'marketTime', 'countryCode', 'timezone', 'settledTime']
        rkeys = ['id', 'name', 'status']
        for d in mlist:  # gets id, marketDefinition as keys if exists
            md = 'marketDefinition'
            mid = (d.get('id'))
            if md in str(d.keys()):
                if d['marketDefinition'].get('status') == 'CLOSED':
                    mklist = [d['marketDefinition'].get(key) for key in keys]  # general market and event info
                    tm1 = [d['marketDefinition'].get('runners')[0].get(key) for key in rkeys]  # first team
                    tm2 = [d['marketDefinition'].get('runners')[1].get(key) for key in rkeys]  # second team
                    draw = [d['marketDefinition'].get('runners')[2].get(key) for key in rkeys]  # draw

    return (mid, mklist, tm1, tm2, draw)


# Get indexes of where market is inPlay = True/False
def get_inplay_indices(mkinfo, mkl):
    list_no = []  # inPlay = True indices
    list_nof = []  # inPlay = False indices

    for j in range(0, mkl):
        nlist = mkinfo[j]['mc']
        for d in nlist:
            md = 'marketDefinition'
            if md in str(d.keys()):
                # append indexes where inPlay status is true
                if ((d['marketDefinition'].get('status') == 'OPEN') and (
                            d['marketDefinition'].get('turnInPlayEnabled') == True) and (
                            d['marketDefinition'].get('inPlay') == True) and (
                            d['marketDefinition'].get('runners')[0].get('status') == 'ACTIVE')):
                    list_no.append(j)
                # append indexes where inPlay status is false
                if ((d['marketDefinition'].get('status') == 'OPEN') and (
                            d['marketDefinition'].get('inPlay') == False) and (
                            d['marketDefinition'].get('runners')[0].get('status') == 'ACTIVE')):
                    list_nof.append(j)

    return (list_no, list_nof)


# Get ltps between last inPlay = False index and first inPlay = True
def get_final_ltps(list_no, list_nof, mkinfo):

    idx_fp = min(list_no)  # first index where inPlay is True (+1 to include in range)
    idx_lp = max(list_nof)  # last index where inPlay is False

    # Get all ltps between indexes
    rdf = pd.DataFrame()
    llist = mkinfo[:][(idx_lp + 1):idx_fp]  # get market info between last inPlay:False and first inPlay True
    lenlist = len(llist)

    for k in range(0, lenlist):
        mclist = llist[k].get('mc')  # gets mc list
        rclist = mclist[0].get('rc')  # gets rc list

        if rclist is None:
            pass

        else:

            # Select runner_id, and ltp
            for idx in rclist:
                akeys = ['id', 'ltp']
                ltpinfo = ([idx.get(key) for key in akeys])  # extracts id (W/L/D) and associated trading price

                flat = [ltpinfo[0], ltpinfo[1]]
                rdf = rdf.append([flat], ignore_index=True)

    # Speed up processing
    if len(rdf) > 100:
        rdf = rdf.tail(100)

    try:
        # Get final ltps
        ltps = pd.DataFrame()
        rdf['idx'] = rdf.index
        ndf = rdf.groupby(rdf[0]).max()

        for idx in ndf.idx:
            rd = rdf.loc[rdf.idx == idx]
            rdx = [rd.iloc[0][0], rd.iloc[0][1], rd.iloc[0]['idx']]
            ltps = ltps.append([rdx], ignore_index=False)

    except:
        pass

    return (ltps, rdf, idx_fp, idx_lp)


def get_market_df(mid, ltps, mklist, tm1, tm2, draw, idx_lp, idx_fp, mkl, file_var, filename):
    market_df = pd.DataFrame()

    # Append winner team info

    if len(ltps.loc[ltps[0] == tm1[0], 1]) == 0:
        pass

    else:

        markif_win = pd.Series(
            [mid, mklist[0], mklist[1], mklist[2], mklist[3], mklist[4], mklist[5], tm1[0], tm1[1], tm1[2],
             ltps.loc[ltps[0] == tm1[0], 0].item(), ltps.loc[ltps[0] == tm1[0], 1].item(), [idx_lp, idx_fp], mkl,
             file_var, filename],
            ['market_id', 'event_id', 'event_name', 'market_time', 'country_code', 'timezone', 'suspend_time',
             'team_id', 'team_name', 'team_status', 'result_id', 'ltp', 'idx_in_file', 'file_len', 'file_no',
             'file_name'])
        market_df = market_df.append([markif_win], ignore_index=True)

    # Append loser team info

    if len(ltps.loc[ltps[0] == tm2[0], 1]) == 0:
        pass

    else:

        markif_lost = pd.Series(
            [mid, mklist[0], mklist[1], mklist[2], mklist[3], mklist[4], mklist[5], tm2[0], tm2[1], tm2[2],
             ltps.loc[ltps[0] == tm2[0], 0].item(), ltps.loc[ltps[0] == tm2[0], 1].item(), [idx_lp, idx_fp], mkl,
             file_var, filename],
            ['market_id', 'event_id', 'event_name', 'market_time', 'country_code', 'timezone', 'suspend_time',
             'team_id', 'team_name', 'team_status', 'result_id', 'ltp', 'idx_in_file', 'file_len', 'file_no',
             'file_name'])
        market_df = market_df.append([markif_lost], ignore_index=True)

    # Append draw info

    # Check if draw exists
    if len(ltps.loc[ltps[0] == draw[0], 1]) == 0:
        pass

    else:

        markif_draw = pd.Series(
            [mid, mklist[0], mklist[1], mklist[2], mklist[3], mklist[4], mklist[5], draw[0], draw[1], draw[2],
             ltps.loc[ltps[0] == draw[0], 0].item(), ltps.loc[ltps[0] == draw[0], 1].item(), [idx_lp, idx_fp], mkl,
             file_var, filename],
            ['market_id', 'event_id', 'event_name', 'market_time', 'country_code', 'timezone', 'suspend_time',
             'team_id', 'team_name', 'team_status', 'result_id', 'ltp', 'idx_in_file', 'file_len', 'file_no',
             'file_name'])
        market_df = market_df.append([markif_draw], ignore_index=True)

    market_df = market_df[market_df.team_id == market_df.result_id]  # drop rows with mismatched ids
    market_df['prob'] = 1. / market_df.ltp

    return market_df
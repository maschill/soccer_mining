from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options

import os
import json
from urllib.request import urlopen, Request

import sys
import time

from unidecode import unidecode

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import sqlite3

from datetime import datetime
from difflib import SequenceMatcher
import copy

from fuzzywuzzy import fuzz, process

# utilities

#convert string of type x% to a float
def perc2float(string):
    s =string.strip()
    if s[-1]=='%':
        return float(s.strip("%"))/100
    return string

#converts date string of type dd-mon-YY to datetime
def str2date(string):
    helper = { 'Sep':9, 'Aug':8, 'Jul':7, 'Jun':6, 'May':5, 'Apr':4, 'Mar':3, 'Feb':2, 'Jan':1, 'Oct':10, 'Nov':11, 'Dec': 12 }
    s = string.split('-')
    return(datetime(int(''.join(['20',s[2]])), helper[s[1]], int(s[0])))

#converts date string of type dd-mon-YY to datetime
def str2date2(string):
    helper = { 'Sep':9, 'Aug':8, 'Jul':7, 'Jun':6, 'May':5, 'Apr':4, 'Mar':3, 'Feb':2, 'Jan':1, 'Oct':10, 'Nov':11, 'Dec': 12 }
    s = string.split(' ')
    return(datetime(int(s[2]), helper[s[1]], int(s[0])))

#
def dist(string1, string2):
    match = SequenceMatcher(None, string1, string2).find_longest_match(0,len(string1), 0, len(string2))
    return match.size

#
def map_names(list1, list2):
    if len(list1) != len(list2):
        print('lists do not have equal length')
        return None
    list12 = [x[0] for x in list1]
    indices = [np.argmax([dist(s2, s) for s2 in list12]) for s in list2]
    mapping = {}
    for i,ix in enumerate(indices):
        mapping[list2[i]] = list1[ix][1]
    return mapping

def find_patch_id(date):
    dates2patch = {datetime(2017,8,28): 158835,
                datetime(2017,2,22): 158647,
                datetime(2016,8,25): 158466,
                datetime(2016,2,19): 158278,
                datetime(2015,8,28): 158103,
                datetime(2015,2,20): 157914,
                datetime(2014,8,29): 157739,
                datetime(2014,2,21): 157550,
                datetime(2013,8,31): 157376,
                datetime(2013,2,22): 157186,
                datetime(2012,8,31): 157011,
                datetime(2012,2,22): 156820,
                datetime(2011,8,30): 156644,
                datetime(2011,2,22): 156455,
                datetime(2010,8,30): 156279,
                datetime(2010,2,22): 156090,
                datetime(2009,8,30): 155914,
                datetime(2009,2,22): 155725,
                datetime(2008,8,30): 155549}


    for d in dates2patch.keys():
        if date < d:
            continue
        else:
            return dates2patch[d]
    print('none found')
    return None


## players: list of players, first half of team 1, second half of team 2
## teams: teamIDs as seen in database
## database: name of database file
## c: sqlite3 cursor.
## Either c or database must have a value.
def player_name2playerID(players, teams, database=None, c=None):
    if not c:
        con = sqlite3.connect(database)
        c = con.cursor()
    query1 = [x for x in c.execute('SELECT name,playerID FROM player WHERE teamID IN ('+str(teams[0])+')')]
    query2 = [x for x in c.execute('SELECT name,playerID FROM player WHERE teamID IN ('+str(teams[1])+')')]
    if not c:
        con.close()
    playerIDs = []

    if len(players ) != 36:
        print(len(players))
        print('not the right amount of players provided, sorry')
        return None

    for player in players[:18]:
        if player == '0':
            playerIDs.append(0)
            print(0)
            continue

        playerIDs.append(query1[np.argmax([dist(player, q[0]) for q in query1])][1])

    for player in players[18:]:
        if player == '0':
            playerIDs.append(0)
            print(0)
            continue
        playerIDs.append(query2[np.argmax([dist(player, q[0]) for q in query2])][1])

    if len(players) != len(set(playerIDs)):
        print('player was matched multiple times')

    return playerIDs

## input player1: (name, playerID, jerseynumber)
##      player2: (name2, jerseynumber)
def match_player(player1, player2):
    result = []

    for i in range(2):
        jn_dict = {jn:pid for name, pid, jn in player1[i]}

        for name2, pid2  in player2[i]:
            try:
                match = jn_dict.get(int(pid2), -1)
                result.append(match)
            except ValueError:
                result.append(-1)
        result.extend([-1]*((i+1)*18-len(result)))
    return result

def match_club_names(club1_raw, club2_raw, scorer=fuzz.ratio):
    # dict of club1 key = name, value = id
    club1_dict = {club: cid for club, cid in club1_raw}
    # list of club2 names
    club2 = copy.deepcopy(club2_raw)
    # list of club1 names
    club1 = [club for club, _ in club1_raw]
    # dict contains counts of club1 name matches
    club1_match_count = {c1: 0 for c1, _ in club1_raw}
    # key = club2 name, value = club1 name
    match_dict = {}

    for c2 in club2:
        result = process.extractOne(c2, club1, scorer=scorer)
        club1_match_count[result[0]] += 1
        match_dict[c2] = result

    # if club1 name was matched once add match to result dict
    result_dict = {}

    for c2, (c1, _) in match_dict.items():
        if club1_match_count[c1] == 1:
            result_dict[c2] = (c1, club1_dict[c1])
            club1.remove(c1)
            club2.remove(c2)

    # repeatedly add match with highest score to result dict
    n = len(club1)

    for i in range(n):
        conflict_list = []

        for c2 in club2:
            result = process.extractOne(c2, club1, scorer=scorer)
            conflict_list.append((result[1], result[0], c2))

        conflict_list.sort(key=lambda tup: tup[0], reverse=True)
        _, c1_x, c2_x = conflict_list[0]

        result_dict[c2_x] = (c1_x, club1_dict[c1_x])
        club1.remove(c1_x)
        club2.remove(c2_x)

    return result_dict


if __name__=='__main__':
    #adblock with add_extension
    chop = webdriver.ChromeOptions()
    chop.add_extension('/home/mace/Downloads/Adblock-Plus_v1.13.3.crx')
    driver = webdriver.Chrome('/home/mace/Downloads/chromedriver', chrome_options=chop)
    driver2 = webdriver.Chrome('/home/mace/Downloads/chromedriver', chrome_options=chop)
    database = 'soccer.sqlite'
    ligaID = 19
    driver.get(url_base)
    con = sqlite3.connect(database)
    c = con.cursor()

    for _ in range(11):
        matches = driver.find_elements_by_css_selector('.score-time.score')
        for match in matches:
            url = match.find_element_by_tag_name('a').get_attribute('href')
            url_ = url.split('/')

            print(url)
            date = datetime(*[int(x) for x in url_[4:7]])
            patch = find_patch_id(date)
            matchID = url_[11]
            #todo map names team_mapping = map_names(stuffstuff)
            homeID, awayID = [x for x in [' '.join(url_[9].split('-')),' '.join(url_[10].split('-'))]]
            driver2.get(url)
            time.sleep(1)
            names = []
            for starting in driver2.find_elements_by_css_selector(".playerstats.lineups.table"):
                for player in starting.find_elements_by_css_selector('.player.large-link'):
                    names.append( ' '.join(player.find_element_by_tag_name('a').get_attribute('href').split('/')[4].split('-')))

            #  player_name2playerID(names, [homeID, awayID], c=c )
            # match_data = player_name2playerID(names, [homeID, awayID], c=c )

            print(matchID, ligaID, date, homeID, awayID, url, names)


            #TODO football-data import

            #TODO DB Export
            print('=============Match Done============')
        driver.find_element_by_css_selector('.previous').click()


    driver.quit()
    driver2.quit()
    con.close()

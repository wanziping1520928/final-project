import pandas as pd
import requests
response = requests.get('https://api.opendota.com/api/heroes')
data = response.json()
heroname = [hero['localized_name']for hero in data[:-1]]
# print(heroname)

match_data = pd.read_csv('match_data')

df_counter_rate= pd.DataFrame(columns=[heroname],index=[heroname],dtype=float)
# print(df_log5)

winrates = {}
for hero in heroname:
    count = 0
    win_count =0
    for index,row in match_data.iterrows():
        radiant_picks = row['radiant_pick']
        dire_picks = row['dire_pick']
        radiant_win = row['radiant_win']
        if hero in radiant_picks:
            count +=1
            if radiant_win:
                win_count +=1
        if hero in dire_picks:
            count +=1
            if not radiant_win:
                win_count += 1
    if count != 0:
        winrate = round(win_count/count,2)
    else:
        winrate = 0
    winrates[hero] = winrate

def actual_winrate_ab(heroA,heroB):
    count = 0
    win_count =0
    for index,row in match_data.iterrows():
        radiant_picks = row['radiant_pick']
        dire_picks = row['dire_pick']
        radiant_win = row['radiant_win']
        if heroA in radiant_picks and heroB in dire_picks:
            count +=1
            if radiant_win:
                win_count +=1
        if heroA in dire_picks and heroB in radiant_picks:
            count +=1
            if not radiant_win:
                win_count += 1
    if count != 0:
        winrate = round(win_count/count,2)
    else:
        winrate = 0
    return winrate

def log5(pa,pb):
    pab = (pa-pa*pb)/(pa+pb-2*pa*pb)
    return pab

for heroA in heroname:
    for heroB in heroname:
        pa = winrates[heroA]
        pb = winrates[heroB]
        pre_pab = log5(pa,pb)
        pab = actual_winrate_ab(heroA,heroB)
        if heroA == heroB:
            counter_rate = 0
        else:
            counter_rate = round(pab-pre_pab,2)
        print(heroA,heroB,counter_rate)
        df_counter_rate.loc[heroA,heroB] = counter_rate


print(df_counter_rate)
df_counter_rate.to_csv('counter_rate.csv')

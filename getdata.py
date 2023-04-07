import time
import json
import requests
import pandas as pd

max_match_id = 6818576227

def get_match_id(max_match_id):
    url = f'https://api.opendota.com/api/proMatches'
    params = {'less_than_match_id': max_match_id}
    response = requests.get(url, params)
    matches = response.json()
    match_ids = [match['match_id'] for match in matches]
    # print(match_ids)
    return match_ids


response = requests.get('https://api.opendota.com/api/heroes')
data = response.json()
heroname = [[hero['id'], hero['localized_name']] for hero in data]
def id_to_name(heroid):
    for i in heroname:
        if i[0] == heroid:
            return i[1]

def getmatchs(id):
    url = f'https://api.opendota.com/api/matches/{id}?api_key=cf9951a8-f50c-460a-93f4-62ee76911d2e'
    getmatchs_tag = 0
    time.sleep(0.05)
    while getmatchs_tag == 0:
        try:
            response = requests.get(url)
            getmatchs_tag = 1
            if response.text:
                match_data = json.loads(response.text)
                return match_data
            else:
                getmatchs(id)
        except:
            getmatchs(id)


response = requests.get('https://api.opendota.com/api/heroes')
data = response.json()
heroname = [[hero['id'],hero['localized_name']]for hero in data]
df_matchs = pd.DataFrame(columns=['match_id','radiant_pick','dire_pick','radiant_pick_order','dire_pick_order','radiant_win'])
k=0
while df_matchs.shape[0]<20000:
    match_id_list = get_match_id(max_match_id)
    for j in range(0,100):
        k += 1
        match_id = match_id_list[j]
        radiant_pick = []
        radiant_pick_order = []
        dire_pick = []
        dire_pick_order = []
        if 'picks_bans' in getmatchs(match_id) and getmatchs(match_id)['picks_bans']:
            print(match_id)
            match_data = getmatchs(match_id)
            # print(match_data)
            picks = match_data['picks_bans']
            # print(picks)
            # print(match_data['radiant_win'])
            for i in picks:
                if i['is_pick'] is True and i['team']==0:
                    radiant_pick.append(id_to_name(i['hero_id']))
                    radiant_pick_order.append(i['order'])
                elif i['is_pick'] is True and i['team'] == 1:
                    dire_pick.append(id_to_name(i['hero_id']))
                    dire_pick_order.append(i['order'])
        df_matchs.loc[k, 'radiant_win'] = match_data['radiant_win']
        df_matchs.loc[k, 'match_id'] = match_data['match_id']
        df_matchs.loc[k, 'radiant_pick'] = radiant_pick
        df_matchs.loc[k, 'radiant_pick_order'] = radiant_pick_order
        df_matchs.loc[k, 'dire_pick'] = dire_pick
        df_matchs.loc[k, 'dire_pick_order'] = dire_pick_order
        df_matchs.to_csv('matchdata2',index=False)
    max_match_id = get_match_id(max_match_id)[-1]
    print(max_match_id)
    print(df_matchs.shape[0])
print(df_matchs)

# matchs = getmatchs(max_match_id)
# print(matchs)
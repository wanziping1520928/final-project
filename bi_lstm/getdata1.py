import json
import requests
import time


max_match_id = 7136914606
target_match_num = 50000
lowest_mmr = 5000

# url = "https://api.opendota.com/api/matches/5075751801?api_key=cf9951a8-f50c-460a-93f4-62ee76911d2e"
base_url = 'https://api.opendota.com/api/publicMatches?less_than_match_id='
session = requests.Session()
session.headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
    'Authorization': 'Bearer cf9951a8-f50c-460a-93f4-62ee76911d2e'}


def crawl(input_url):
    time.sleep(0.1)
    crawl_tag = 0
    while crawl_tag==0:
        try:
            session.get("http://www.opendota.com/")
            content = session.get(input_url)
            crawl_tag = 1
        except:
            print("Poor internet connection. We'll have another try.")
    json_content = json.loads(content.text)
    #print(json_content)
    return json_content


match_list = []
recurrent_times = 0
write_tag = 0
with open('matches_list_ranking.csv','w') as fout:
    fout.write('gameID, time, radiant_team, dire_team, radiant_win\n')
    while(len(match_list)<target_match_num):
        json_content = crawl(base_url + str ( max_match_id ))
        for i in range(len(json_content)):
            match_id = json_content[i]['match_id']
            radiant_win = json_content[i]['radiant_win']
            start_time = json_content[i]['start_time']
            avg_mmr = json_content[i]['avg_mmr']
            if avg_mmr==None:
                avg_mmr = 0
            lobby_type = json_content[i]['lobby_type']
            game_mode = json_content[i]['game_mode']
            radiant_team = json_content[i]['radiant_team']
            dire_team = json_content[i]['dire_team']
            duration = json_content[i]['duration']
            if int(avg_mmr)<lowest_mmr:
                continue
            if int(duration)<1200:
                continue
            if int(lobby_type)!=7 or (int(game_mode)!=3 and int(game_mode)!=22):
                continue
            x = time.localtime(int(start_time))
            game_time = time.strftime('%Y-%m-%d %H:%M:%S',x)
            one_game = [game_time,radiant_team,dire_team,radiant_win,match_id]
            match_list.append(one_game)
        max_match_id = json_content[-1]['match_id']
        recurrent_times += 1
        print(recurrent_times,len(match_list),max_match_id)
        if len(match_list)>target_match_num:
            match_list = match_list[:target_match_num]
        if write_tag<len(match_list):
            for i in range(len(match_list))[write_tag:]:
                fout.write(str(match_list[i][4])+', '+match_list[i][0]+', '+match_list[i][1]+', '+\
                    match_list[i][2]+', '+str(match_list[i][3])+'\n')
            write_tag = len(match_list)
import streamlit as st
import numpy as np
import pandas as pd
import os
import datetime
from zoneinfo import ZoneInfo


from riotwatcher import LolWatcher, RiotWatcher, ApiError

RIOT_API_KEY = os.environ.get("RIOT_API_KEY")


lol_watcher = LolWatcher(RIOT_API_KEY)

riot_watcher = RiotWatcher(RIOT_API_KEY)

SPACES_KEY = os.environ.get("SPACES_KEY")
SPACES_SECRET = os.environ.get("SPACES_SECRET")

import os, boto3
from smart_open import open

session = boto3.session.Session()
client = session.client(
    "s3",
    region_name="nyc3",
    endpoint_url="https://nyc3.digitaloceanspaces.com",
    aws_access_key_id=SPACES_KEY,
    aws_secret_access_key=SPACES_SECRET,
)
transport_params = {"client": client}


my_region = "na1"

st.set_page_config(layout="wide")
st.title("LoL Challenge")

teams = [
    {
        "id": 1,
        "name": "BANNED RIP",
        "members": [
            {"realname": "Benny", "name": "festivus", "tag": "feats"},
            {"realname": "EJ", "name": "TonkaTank420", "tag": "Tank1"},
        ],
    },
    {
        "id": 2,
        "name": "cc treeshes 👫",
        "members": [
            {"realname": "Triston", "name": "Shadow Daddy", "tag": "Bro"},
            {"realname": "Calvin", "name": "Void Mommy", "tag": "Sis"},
        ],
    },
    {
        "id": 3,
        "name": "Fat Boy Sam™ 🎅",
        "members": [
            {"realname": "Purgance", "name": "SAM IS BIG BODY", "tag": "TRASH"},
            {"realname": "Dillon", "name": "SAMiSFaTTY", "tag": "4904"},
        ],
    },
    {
        "id": 4,
        "name": "Little Man Vods 🥈",
        "members": [
            {"realname": "Sam", "name": "PikeHigh2014", "tag": "317"},
            {"realname": "Brandon", "name": "StephenUllrich", "tag": "LOVER"},
        ],
    },
    {
        "id": 5,
        "name": "Pike Bros On Top 🐟",
        "members": [
            {"realname": "Downzee", "name": "Wanariceciab", "tag": "7336"},
            {"realname": "Snivel", "name": "ASPOnTop", "tag": "Indy"},
        ],
    },
    {
        "id": 6,
        "name": "Knights who say Ni 🛡️",
        "members": [
            {"realname": "Raythar", "name": "KnockKnock", "tag": "Jorts"},
            {"realname": "Braveclue", "name": "WhosThere", "tag": "Jorts"},
        ],
    },
    {
        "id": 7,
        "name": "Femboy Poop Plug 👯‍♂️",
        "members": [
            {"realname": "Surge", "name": "JSurge70onTwitch", "tag": "Stink"},
            {"realname": "BP", "name": "SurgePoopPlug", "tag": "Fem"},
        ],
    },
    {
        "id": 7,
        "name": "Bouncing in the Bounce House 🏰",
        "members": [
            {"realname": "Benny", "name": "training season", "tag": "dua", "starttime" :  1738290600},
            {"realname": "EJ", "name": "Time2Fish", "tag": "FishU", "starttime" :  1738290600},
        ],
    },
]


def get_match_by_id(match_id):
    import json

    try:
        with open(
            f"s3://lol-challenge/matches/{match_id}.json",
            "r",
            transport_params=transport_params,
        ) as m:
            match = json.load(m)
    except:
        match = lol_watcher.match.by_id(my_region, match_id)
        with open(
            f"s3://lol-challenge/matches/{match_id}.json",
            "w",
            transport_params=transport_params,
        ) as m:
            json.dump(match, m)
    return match


@st.cache_data
def get_account(name, tag):
    return riot_watcher.account.by_riot_id("AMERICAS", name, tag)


@st.cache_data
def get_matches(puuid, timestamp, starttime = 1737504000):
    match_ids = lol_watcher.match.matchlist_by_puuid(
        my_region, puuid, count=100, start_time=starttime
    )
    matches = []
    for match_id in match_ids:
        match_rec = get_match_by_id(match_id)
        queue_id = match_rec["info"]["queueId"]
        if queue_id == 420:
            matches.append(match_rec)
    return matches


@st.cache_data
def get_summoner(acc):
    return lol_watcher.summoner.by_puuid(my_region, acc["puuid"])


def get_league_data(summoner):
    return lol_watcher.league.by_summoner(my_region, summoner["id"])


@st.cache_data
def get_league_data_by_summoner_id(summoner_id):
    import json

    try:
        with open(
            f"s3://lol-challenge/leagues/{summoner_id}.json",
            "r",
            transport_params=transport_params,
        ) as m:
            league_data = json.load(m)
        return league_data
    except:
        league_data = lol_watcher.league.by_summoner(my_region, summoner_id)
        if not league_data:
            return []
        with open(
            f"s3://lol-challenge/leagues/{summoner_id}.json",
            "w",
            transport_params=transport_params,
        ) as m:
            json.dump(league_data, m)
        return league_data


@st.cache_data
def get_avg_rank(match_dto):
    team_lps = []
    for participant in match_dto["info"]["participants"]:
        summoner_id = participant["summonerId"]
        import time

        # wait .02 seconds

        league_data = get_league_data_by_summoner_id(summoner_id)
        for stats in league_data:
            if stats["queueType"] == "RANKED_SOLO_5x5":
                tier = stats["tier"]
                rank = stats["rank"]
                points = stats["leaguePoints"]
                team_lps.append(RANK_MAPPING[tier] + TIER_MAPPING[rank] + points)
    if not team_lps:
        return None
    return int(np.mean(team_lps))


def get_rank_string_from_lp(lp):
    sorted_ranks = sorted(RANK_MAPPING.items(), key=lambda x: x[1])
    for k, v in sorted_ranks:
        if lp > v and lp < v + 400:

            remaining_lp = lp - v
            print(k, v, remaining_lp)
            tier = k
            sorted_tiers = sorted(TIER_MAPPING.items(), key=lambda x: x[1])
            for k2, v2 in sorted_tiers:
                print(k2, v2)
                if remaining_lp > v2 and remaining_lp < v2 + 100:
                    print("found")
                    return f"{tier} {k2} {remaining_lp - v2}LP"
    return "UNRANKED"


RANK_MAPPING = {
    "IRON": 0,
    "BRONZE": 400,
    "SILVER": 800,
    "GOLD": 1200,
    "PLATINUM": 1600,
    "EMERALD": 2000,
    "DIAMOND": 2400,
}

TIER_MAPPING = {
    "IV": 0,
    "III": 100,
    "II": 200,
    "I": 300,
}


def get_ranked_stats(acc):
    summoner = get_summoner(acc)
    leaguedata = get_league_data(summoner)
    for stats in leaguedata:
        if stats["queueType"] == "RANKED_SOLO_5x5":
            return stats
    return None


@st.cache_data
def get_rank_string(acc, timestamp):
    ranked_stats = get_ranked_stats(acc)
    if not ranked_stats:
        return "UNRANKED"
    tier = ranked_stats["tier"]
    rank = ranked_stats["rank"]
    points = ranked_stats["leaguePoints"]
    return f"{tier} {rank} {points}LP"


def get_lp(acc):
    ranked_stats = get_ranked_stats(acc)
    if not ranked_stats:
        return 0
    tier = ranked_stats["tier"]
    rank = ranked_stats["rank"]
    points = ranked_stats["leaguePoints"]
    return RANK_MAPPING[tier] + TIER_MAPPING[rank] + points


def get_team(match_dto, puuid):
    for participant in match_dto["info"]["participants"]:
        if participant["puuid"] == puuid:
            return participant["teamId"]
    return None


def get_kda_player_stats(match_dto, puuid):
    for participant in match_dto["info"]["participants"]:
        if participant["puuid"] == puuid:
            kills = participant["kills"]
            deaths = participant["deaths"]
            assists = participant["assists"]
            total_dmg_dealt_to_champions = participant["totalDamageDealtToChampions"]
            enemy_missing_pings = participant["enemyMissingPings"]
            
            gold_per_minute = participant["challenges"]["goldPerMinute"]
            return (
                kills,
                deaths,
                assists,
                total_dmg_dealt_to_champions,
                enemy_missing_pings,
                
                gold_per_minute,
            )
    return None


def get_team_kills(match_dto, team_id):
    for team in match_dto["info"]["teams"]:
        if team["teamId"] == team_id:
            return team["objectives"]["champion"]["kills"]
    return None


def get_winning_team(match_dto):
    for team in match_dto["info"]["teams"]:
        if team["win"]:
            return team["teamId"]
    return None


def get_fullname(name, tag):
    return f"{name}#{tag}"


def get_opgg(name, tag):
    import urllib.parse

    s = f"https://op.gg/summoners/na/{name}-{tag}"
    # urlencode string
    return urllib.parse.quote(s, safe=":/")


def get_link(team_member):
    name = team_member["name"]
    tag = team_member["tag"]
    fullname = get_fullname(name, tag)
    opgg = get_opgg(name, tag)
    s = f"[{fullname}]({opgg})"
    return s


def get_duration_seconds(dto):
    ts = dto["info"]["gameEndTimestamp"]
    ts2 = dto["info"]["gameStartTimestamp"]
    duration_seconds = (ts - ts2) / 1000
    return duration_seconds


def round_to_nearest_10_seconds(dt):
    return dt + datetime.timedelta(seconds=-(dt.second % 10))


@st.cache_data
def get_data(timestamp):
    data = []
    for team in teams:
        # st.markdown(
        #     f"""{team["name"]} {get_link(team["members"][0])},
        #     {get_link(team["members"][1])}
        #     """
        # )
        wins = []
        losses = []
        remakes = []
        teamdata = []
        avg_team_rank = 0
        avg_team_rank_str = ""
        member_counter = 0
        avg_ranks = []
        team_description = team['name']
        for member in team["members"]:
            fullname = get_fullname(member["name"], member["tag"])
            realname = member["realname"]
            op_gg = get_opgg(member["name"], member["tag"])
            account = get_account(member["name"], member["tag"])
            dt = datetime.datetime.now()
            dtr = round_to_nearest_10_seconds(dt)
            now_time_str = dtr.strftime("%Y-%m-%dT%H:%M:%S")
            starttime = member.get("starttime", 1737504000)
            matches = get_matches(account["puuid"], now_time_str, starttime)
            win = 0
            loss = 0
            remake = 0
            durations = []
            kills = 0
            deaths = 0
            assists = 0
            sum_total_enemy_missing_pings = 0
            kdas = []
            kds = []
            kps = []
            dmgs = []
            all_total_enemy_missing_pings = []
            gold_per_minutes = []
            counter = 0
            for dto in matches:
                duration_seconds = get_duration_seconds(dto)
                if duration_seconds > 210:
                    durations.append(duration_seconds)
                    team = get_team(dto, account["puuid"])
                    team_kills = get_team_kills(dto, team)
                    winning_team = get_winning_team(dto)
                    print(winning_team)
                    if team == winning_team:
                        win += 1
                    else:
                        loss += 1
                    (
                        kill,
                        death,
                        assist,
                        total_dmg_dealt_to_champions,
                        enemy_missing_pings,
                        gold_per_minute,
                    ) = get_kda_player_stats(dto, account["puuid"])
                    kills += kill
                    deaths += death
                    assists += assist
                    sum_total_enemy_missing_pings += enemy_missing_pings
                    if death > 0:
                        kda = (kill + assist) / death
                        kd = kill / death
                    else:
                        kda = kill + assist
                        kd = kill
                    kp = (kill + assist) / max(team_kills, 1)
                    dmgs.append(total_dmg_dealt_to_champions)
                    all_total_enemy_missing_pings.append(enemy_missing_pings)
                    kds.append(kd)
                    kdas.append(kda)
                    kps.append(kp)

                    gold_per_minutes.append(gold_per_minute)
                    if counter < 5 and member_counter == 0:
                        avg_rank = get_avg_rank(dto)
                        if avg_rank:
                            avg_ranks.append(avg_rank)
                    counter += 1
                else:
                    remake += 1
            member_counter += 1
            wins.append(win)
            losses.append(loss)
            remakes.append(remake)
            avg_kda = np.mean(kdas)
            avg_kd = np.mean(kds)
            avg_kp = np.mean(kps)
            avg_dmg = int(np.mean(dmgs))
            avg_missing_pings = int(np.mean(all_total_enemy_missing_pings))

            avg_gold_per_minute = np.mean(gold_per_minutes)
            wr = win / (win + loss) if win + loss > 0 else 0
            print(avg_ranks)
            avg_team_rank = np.mean(avg_ranks)
            if len(avg_ranks) == 0:
                avg_team_rank_str = "UNRANKED"
            else:
                print(avg_team_rank)
                print(int(avg_team_rank))
                avg_team_rank_str = get_rank_string_from_lp(int(avg_team_rank))

            if len(durations) == 0:
                avg_duration = 0
            else:
                avg_duration = np.mean(durations)
            avg_duration_str = str(datetime.timedelta(seconds=int(avg_duration)))
            teamdata.append(
                {
                    "name": realname,
                    "opgg": op_gg,
                    "wins": win,
                    "losses": loss,
                    "winrate": wr,
                    "rank": get_rank_string(account, now_time_str),
                    "lp": get_lp(account),
                    "remakes": remake,
                    "avg_duration": avg_duration_str,
                    "avg_kda": avg_kda,
                    "avg_kd": avg_kd,
                    "avg_kp": avg_kp,
                    "avg_dmg": avg_dmg,
                    "kills": kills,
                    "deaths": deaths,
                    "assists": assists,
                    "enemy_missing_pings": sum_total_enemy_missing_pings,
                    "avg_missing_pings": avg_missing_pings,
                    "avg_gold_per_minute": avg_gold_per_minute,
                }
            )
        if matches:
            last_match = matches[0]
            ts = last_match["info"]["gameEndTimestamp"]
            est_time = (
                datetime.datetime.fromtimestamp(ts / 1000.0, tz=datetime.timezone.utc)
                .astimezone(tz=ZoneInfo("US/Eastern"))
                .strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            est_time = "No matches"
        avg_lp = sum([d["lp"] for d in teamdata]) / len(teamdata)
        avg_rank_string = get_rank_string_from_lp(int(avg_lp))
        
        print(avg_team_rank_str)
        for d in teamdata:
            d["avg_lp"] = avg_lp
            d["avg_rank"] = avg_rank_string
            d["team"] = team_description
            d["last_match"] = est_time
            d["avg_game_rank"] = avg_team_rank_str
        data.extend(teamdata)
        mean_wins = np.mean(wins)
        mean_losses = np.mean(losses)

        if mean_wins + mean_losses == 0:
            winrate = 0
        else:
            winrate = mean_wins / (mean_wins + mean_losses)
        # st.text(f"Wins: {mean_wins}")
        # st.text(f"Losses: {mean_losses}")
        # st.text(f"Winrate: {winrate}")

    df2 = pd.DataFrame(data).sort_values("avg_lp", ascending=False)
    df2 = df2[
        [
            "team",
            "name",
            "opgg",
            "rank",
            "lp",
            "avg_lp",
            "avg_rank",
            "wins",
            "losses",
            "winrate",
            "remakes",
            "avg_duration",
            "avg_kda",
            "avg_kd",
            "avg_kp",
            "avg_dmg",
            "kills",
            "deaths",
            "assists",
            "enemy_missing_pings",
            "avg_missing_pings",
            "avg_gold_per_minute",
            "last_match",
            "avg_game_rank",
        ]
    ]
    return df2


import time

dt = datetime.datetime.now()
dtr = round_to_nearest_10_seconds(dt)
now_time_str = dtr.strftime("%Y-%m-%dT%H")

t = st.empty()
try:
    df = pd.read_csv(
        open("s3://lol-challenge/data.csv", "r", transport_params=transport_params)
    )
    t.dataframe(df, column_config={"opgg": st.column_config.LinkColumn()})
except:
    pass


def refresh_data():
    dt = datetime.datetime.now()
    ts = dt.astimezone(tz=ZoneInfo("US/Eastern")).strftime("%Y-%m-%dT%H:%M:%S")
    df1 = get_data(ts)
    df1["updated_at"] = ts
    t = st.empty()
    t.dataframe(df1, column_config={"opgg": st.column_config.LinkColumn()})
    df1.to_csv(
        open("s3://lol-challenge/data.csv", "wb", transport_params=transport_params),
        index=False,
    )


st.button("Refresh Data", on_click=refresh_data)

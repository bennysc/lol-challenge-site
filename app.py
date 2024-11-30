import streamlit as st
import numpy as np
import pandas as pd
import os


from riotwatcher import LolWatcher, RiotWatcher, ApiError


lol_watcher = LolWatcher(os.environ["RIOT_API_KEY"])

riot_watcher = RiotWatcher(os.environ["RIOT_API_KEY"])


my_region = "na1"


st.title("LoL Challenge")

teams = [
    {
        "id": 1,
        "name": "blue",
        "members": [
            {"realname":"Benny","name": "gorp", "tag": "grubs"},
            {"realname":"Surge","name": "Big Beak", "tag": "beak"},
        ],
    },
    {
        "id": 2,
        "name": "red",
        "members": [
            {"realname":"Draco","name": "Rank1Azir", "tag": "8182"},
            {"realname":"BP","name": "AnaSecretAdmirer", "tag": "885"},
        ],
    },
    {
        "id": 3,
        "name": "green",
        "members": [
            {"realname":"Calvin","name": "BennyGoatBAAAH", "tag": "NA3"},
            {"realname":"Sam","name": "CantHandicapMe", "tag": "CALVN"},
        ],
    },
    {
        "id": 4,
        "name": "yellow",
        "members": [
            {"realname":"EJ","name": "Monoler", "tag": "NA1"},
            {"realname":"Edwin","name": "Mitooma", "tag": "NA1"},
        ],
    },
    {
        "id": 5,
        "name": "purple",
        "members": [
            {"realname":"Raythar","name": "BennysBonita", "tag": "Benny"},
            {"realname":"Snivel","name": "Odegurd", "tag": "NA1"},
        ],
    },
]


@st.cache_data
def get_account(name, tag):
    return riot_watcher.account.by_riot_id("AMERICAS", name, tag)


@st.cache_data
def get_matches(puuid):
    match_ids = lol_watcher.match.matchlist_by_puuid(
        my_region, puuid, count=100, queue=420
    )
    matches = []
    for match_id in match_ids:
        matches.append(lol_watcher.match.by_id(my_region, match_id))
    return matches

@st.cache_data
def get_summoner(acc):
    return lol_watcher.summoner.by_puuid(my_region, acc['puuid'])

@st.cache_data
def get_league_data(summoner):
    return lol_watcher.league.by_summoner(my_region, summoner['id'])


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
        if stats['queueType'] == "RANKED_SOLO_5x5":
            return stats
    return None


@st.cache_data
def get_rank_string(acc):
    ranked_stats = get_ranked_stats(acc)
    if not ranked_stats:
        return "UNRANKED"
    tier = ranked_stats['tier']
    rank = ranked_stats['rank']
    points = ranked_stats['leaguePoints']
    return f"{tier} {rank} {points}LP"

def get_lp(acc):
    ranked_stats = get_ranked_stats(acc)
    if not ranked_stats:
        return 0
    tier = ranked_stats['tier']
    rank = ranked_stats['rank']
    points = ranked_stats['leaguePoints']
    return RANK_MAPPING[tier] + TIER_MAPPING[rank] + points


def get_team(match_dto, puuid):
    for participant in match_dto["info"]["participants"]:
        if participant["puuid"] == puuid:
            return participant["teamId"]
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


data = []
for team in teams:
    # st.markdown(
    #     f"""{team["name"]} {get_link(team["members"][0])}, 
    #     {get_link(team["members"][1])}
    #     """
    # )
    wins = []
    losses = []
    teamdata = []
    for member in team["members"]:
        fullname = get_fullname(member["name"], member["tag"])
        realname = member["realname"]
        op_gg = get_opgg(member["name"], member["tag"])
        account = get_account(member["name"], member["tag"])
        matches = get_matches(account["puuid"])
        win = 0
        loss = 0
        for dto in matches:
            team = get_team(dto, account["puuid"])
            winning_team = get_winning_team(dto)
            if team == winning_team:
                win += 1
            else:
                loss += 1
        wins.append(win)
        losses.append(loss)
        wr = win / (win + loss) if win + loss > 0 else 0
        teamdata.append(
            {
                "name": realname,
                "opgg": op_gg,
                "wins": win,
                "losses": loss,
                "winrate": wr,
                "rank": get_rank_string(account),
                "lp":get_lp(account)  
            }
        )
    avg_lp = sum([d["lp"] for d in teamdata]) / len(teamdata)
    team_description = f"{teamdata[0]['name']} and {teamdata[1]['name']}"
    for d in teamdata:
        d["avg_lp"] = avg_lp
        d["team"] = team_description
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

df = pd.DataFrame(data).sort_values("avg_lp", ascending=False)

st.dataframe(
    df,
    column_config={
        "account_data": {"max_width": 200},
        "opgg": st.column_config.LinkColumn(),
    },
)


def clear_cache():
    st.cache_data.clear()
    st.write("Cache cleared")


st.button("Clear cache", on_click=clear_cache)

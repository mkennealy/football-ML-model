import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import ssl
import numpy as np

# use the ssl module to "workaround" the certification (to avoid SSL: CERTIFICATE_VERIFY_FAILED error)
ssl._create_default_https_context = ssl._create_unverified_context


def get_football_uk_data():
    """
    reads in csv files downloaded from football-data.co.uk for premier league matches from 2008 to 2022
    concatenates data from each season together
    """

    df_all_seasons = pd.DataFrame()

    seasons = []
    for x in range (8,22): #create list of season start and end dates
        season_start = str(x)
        season_end = str(x + 1)
        if len(season_start) == 1:
            season_start = "0"+season_start
        if len(season_end) == 1:
            season_end = "0"+season_end
        
        season=season_start+season_end
        seasons.append(season)

    # N.B. the number of columns in each dataset varies, ranging from 62 columns in 2016/7 season to 106 in 2020/1
    for season in seasons:
        df_season = pd.read_csv(r"C:\Users\Spike\Documents\Football Prediction\Premier League Matches\{}.csv".format(season))
        df_season["Season"] = season
        df_all_seasons = pd.concat(objs = [df_all_seasons,df_season],join='outer',axis=0)
    
    return df_all_seasons

def get_fifa_data():
    """
    webscrapes data from https://www.fifaindex.com for team statistics (Attck, Midfield, Defense, and Overall) in each season 
    """

    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(u'https://www.fifaindex.com/teams/')
    soup = BeautifulSoup(driver.page_source, 'html5lib') #parse the  HTML

    premier_league_urls = []

    for link in soup.findAll('a'):
        link = str(link.get('href'))
        if "/teams/fifa" in link and "wc" not in link and "fifa22" not in link: # 2022 data uses different url format
            premier_league_url = link + "?league=13&order=desc"
            premier_league_urls.append(premier_league_url)
    premier_league_urls.insert(0,"/teams/?league=13&order=desc") 

    df_team_scores = pd.DataFrame()
    for ix, premier_league_url in enumerate(premier_league_urls):
        dfs = pd.read_html(u"https://www.fifaindex.com{}".format(premier_league_url)) #read table on webpage
        df = dfs[0]
        df = df[["Name","League","ATT","MID","DEF","OVR"]]
        df = df[pd.isnull(df["Name"])==False]
        df["Year"] = 2022-ix
        df_team_scores = df_team_scores.append(df)

    #need to make the name of the teams abbreviated to be consistent with football-data.co.uk
    name_changes = {}
    for name in df_team_scores["Name"].unique():
        name_changes[name] = name
    name_changes['Manchester City'] = "Man City"
    name_changes['Manchester United'] = "Man United"
    name_changes['Tottenham Hotspur'] = "Tottenham"
    name_changes['Leicester City'] = "Leicester"
    name_changes['West Ham United'] = "West Ham"
    name_changes['Wolverhampton Wanderers'] = "Wolves"
    name_changes['Newcastle United'] = "Newcastle"
    name_changes['Leeds United'] = "Leeds"
    name_changes['Brighton & Hove Albion'] = "Brighton"
    name_changes['Norwich City'] = "Norwich"
    name_changes['West Bromwich Albion'] = "West Brom"
    name_changes['Sheffield United'] = "Sheffield United"
    name_changes['Sheffield'] = "Sheffield United"
    name_changes['AFC Bournemouth'] = "Bournemouth"
    name_changes['Huddersfield Town'] = "Huddersfield"
    name_changes['Cardiff City'] = "Cardiff"
    name_changes['Stoke City'] = "Stoke"
    name_changes['Swansea City'] = "Swansea"
    name_changes['Hull City'] = "Hull"
    name_changes['Queens Park Rangers'] = "QPR"
    name_changes['Wigan Athletic'] = "Wigan"
    name_changes['Blackburn Rovers'] = "Blackburn"
    name_changes['Bolton Wanderers'] = "Bolton"
    name_changes['Birmingham City'] = "Birmingham"
    name_changes['West Bromwich'] = "West Brom"
    name_changes['Bolton Wanderers'] = "Bolton"
    name_changes['Chelsea FC'] = "Chelsea"
    name_changes['Arsenal FC'] = "Arsenal"
    name_changes['Reading FC'] = "Reading"

    df_team_scores.Name = df_team_scores.Name.apply(lambda x:name_changes[x])

    return df_team_scores


def merge_in_fifa_data(df,fifa_data):
    """
    merges data from football-data.co.uk with the fifa data
    """
    
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True) 
    df["Year"] = df["Date"].apply(lambda x:x.year) #create Year field to merge on 
    df = df.merge(fifa_data[["Name","Year","ATT","MID","DEF","OVR"]].add_suffix("_Home"),left_on = ["HomeTeam","Year"],right_on = ["Name_Home","Year_Home"],how = "left")
    df = df.merge(fifa_data[["Name","Year","ATT","MID","DEF","OVR"]].add_suffix("_Away"),left_on = ["AwayTeam","Year"],right_on = ["Name_Away","Year_Away"],how = "left")
    
    return df

def get_data():
    """
    uses the functions above and returns the full raw dataset
    """

    football_uk_data = get_football_uk_data()
    fifa_data = get_fifa_data()
    merged_data = merge_in_fifa_data(football_uk_data,fifa_data)

    return merged_data

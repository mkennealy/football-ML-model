import pandas as pd
import numpy as np

def clean_data(df):
    '''
    cleans features in the raw dataset
    '''
    #convert numerical variables into numeric datatype
    for col in df.columns:
        if col not in ['Div', 'Date', 'HomeTeam', 'AwayTeam','Teams','Referee','FTR','HTR',"Season"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    #keep columns with fewer than 1000 rows of missing data 
    vars_to_keep = []
    for k,v in dict(df.isna().sum()).items():
        if v <1000:
            vars_to_keep.append(k)
    df = df[vars_to_keep]

    #remove rows that are missing a full time result (erroneous data)
    df=df[df["FTR"].isin(["H","A","D"])] 
    return df

def feature_engineering(df):
    '''
    creates new predictive features based on existing ones
    '''

    df = clean_data(df)

    ####### Lagged Avg. Goal features ############
    #This includes:
    #   - Average number of goals scored in each match by home/away team in previous 2 seasons (when it played at home, when it played away, and an average of the two)
    #   - Total number of goals scored by home/away team in previous 2 seasons (when it played at home, when it played away, and an average of the two)
    #   - The difference in these metrics for the Home/Away Team for each match

    grp_goals_home = df.groupby(["Season","HomeTeam"]).agg({'FTHG': ['mean', 'count','sum']})
    grp_goals_home.columns = grp_goals_home.columns.to_flat_index().str.join('_')
    grp_goals_home.reset_index(inplace=True)

    grp_goals_away = df.groupby(["Season","AwayTeam"]).agg({'FTAG': ['mean', 'count','sum']})
    grp_goals_away.columns = grp_goals_away.columns.to_flat_index().str.join('_')
    grp_goals_away.reset_index(inplace=True)
    grp_goals_home_away = pd.merge(grp_goals_home,grp_goals_away,left_on =["Season","HomeTeam"],right_on = ["Season","AwayTeam"])

    #each team plays an equal number of matches at home or away so this calculates the average of all matches regardless of if they played at home/away
    grp_goals_home_away["Avg_mean_Goals"] = (grp_goals_home_away["FTHG_mean"] + grp_goals_home_away["FTAG_mean"])/2
    grp_goals_home_away["Avg_sum_Goals"] = (grp_goals_home_away["FTHG_sum"] + grp_goals_home_away["FTAG_sum"])/2
                    
    #previous season average goals per match and total goals in at home/away and overall average  
    grp_goals_home_away_shifted1 = grp_goals_home_away.groupby(["Season","HomeTeam"])[["FTHG_mean","FTAG_mean","Avg_mean_Goals","FTHG_sum","FTAG_sum","Avg_sum_Goals"]].mean().unstack().shift(1).stack()
    grp_goals_home_away_shifted2 = grp_goals_home_away.groupby(["Season","HomeTeam"])[["FTHG_mean","FTAG_mean","Avg_mean_Goals","FTHG_sum","FTAG_sum","Avg_sum_Goals"]].mean().unstack().shift(2).stack()
    grp_goals_home_away_shifted1 = grp_goals_home_away_shifted1.add_suffix("_shift1") # add shift1 label to all column names 
    grp_goals_home_away_shifted2 = grp_goals_home_away_shifted2.add_suffix("_shift2")

    df = df.merge(grp_goals_home_away_shifted1.add_suffix("_H"),left_on=["Season","HomeTeam"],right_on=["Season","HomeTeam"],how="left")
    df = df.merge(grp_goals_home_away_shifted1.add_suffix("_A"),left_on=["Season","AwayTeam"],right_on=["Season","HomeTeam"],how="left")
    df = df.merge(grp_goals_home_away_shifted2.add_suffix("_H"),left_on=["Season","HomeTeam"],right_on=["Season","HomeTeam"],how="left")
    df = df.merge(grp_goals_home_away_shifted2.add_suffix("_A"),left_on=["Season","AwayTeam"],right_on=["Season","HomeTeam"],how="left")
    
    df["Avg_Goal_Diff_shift1"] = df["Avg_mean_Goals_shift1_H"] - df["Avg_mean_Goals_shift1_A"] 
    df["Sum_Goal_Diff_shift1"] = df["Avg_sum_Goals_shift1_H"] - df["Avg_sum_Goals_shift1_A"] 
    df["Avg_Goal_Diff_shift2"] = df["Avg_mean_Goals_shift2_H"] - df["Avg_mean_Goals_shift2_A"]
    df["Sum_Goal_Diff_shift2"] = df["Avg_sum_Goals_shift2_H"] - df["Avg_sum_Goals_shift2_A"]

    ########### FIFA Comparison Features ##############
    # create new features that represent the differences in the EA Sports FIFA scores for the Home and Away teams in each season
    df["Attack_Diff"] = df["ATT_Home"] - df["ATT_Away"]
    df["Defence_Diff"] = df["DEF_Home"] - df["DEF_Away"]
    df["Midfield_Diff"] = df["MID_Home"] - df["MID_Away"]
    df["Overall_Diff"] = df["OVR_Home"] - df["OVR_Away"]

    df["H_Attack_A_Def_Diff"] = df["ATT_Home"] - df["DEF_Away"]
    df["H_Attack_A_Midfield_Diff"] = df["ATT_Home"] - df["MID_Away"]
    df["H_Midfield_A_Def_Diff"] = df["MID_Home"] - df["DEF_Away"]

    df["A_Attack_H_Def_Diff"] = df["ATT_Away"] - df["DEF_Home"]
    df["A_Attack_H_Midfield_Diff"] = df["ATT_Away"] - df["MID_Home"]
    df["A_Midfield_H_Def_Diff"] = df["MID_Away"] - df["DEF_Home"]

    #add feature of combination of teams in match playing at home and away 
    df["Teams"] = df["HomeTeam"] + " vs " + df["AwayTeam"]

    ########## Add features for if home/away team was not in the Premier League in previous season, and season before that, or not in both seasons ###
    group_home = df.groupby(["Season","HomeTeam"]).agg({"FTHG":'count'})
    group_home['FTHG_shift1'] = group_home.unstack().shift(1).stack()
    group_home["TeamNotInPremLeagueLastSeasonFlag"] = 0
    group_home.loc[pd.isnull(group_home.FTHG_shift1)==True,"TeamNotInPremLeagueLastSeasonFlag"] = 1
    group_home['FTHG_shift2'] = group_home.unstack().FTHG.shift(2).stack()
    group_home["TeamNotInPremLeagueLastSeason2Flag"] = 0
    group_home.loc[pd.isnull(group_home.FTHG_shift2)==True,"TeamNotInPremLeagueLastSeason2Flag"] = 1
    
    group_home["TeamNotInPremLeagueBothPrevSeasons"] = np.where(((group_home.TeamNotInPremLeagueLastSeasonFlag)&(group_home.TeamNotInPremLeagueLastSeason2Flag)),
                                                           1,
                                                           0)

    df = pd.merge(df,group_home[["TeamNotInPremLeagueLastSeasonFlag","TeamNotInPremLeagueLastSeason2Flag","TeamNotInPremLeagueBothPrevSeasons"]].add_prefix("Home"),on=["Season","HomeTeam"])
    df = pd.merge(df,group_home[["TeamNotInPremLeagueLastSeasonFlag","TeamNotInPremLeagueLastSeason2Flag","TeamNotInPremLeagueBothPrevSeasons"]].add_prefix("Away"),left_on=["Season","AwayTeam"],right_on=["Season","HomeTeam"])

    #Create feature for combination of teams playing in any order (regardless of the order of who is playing away and who is playing at home )
    list_of_team_combos = list(zip(df["HomeTeam"],df["AwayTeam"])) # Non Unique (order of home and away team still matters)
    team_combos_sets = [frozenset(team_combo) for team_combo in list_of_team_combos]
    unique_team_combos = list(set(team_combos_sets)) #make a set of frozen sets, then convert this to a list

    #Give each team combination a unique ID
    dict_unique_teams_ID = {}
    for i,team in enumerate(unique_team_combos):
        dict_unique_teams_ID[team]=i
        return df
    
    def make_teams_set(df,dict_unique_teams_ID):
        teams_set = frozenset([df["HomeTeam"],df["AwayTeam"]])
        unique_ID = dict_unique_teams_ID[teams_set]
        return unique_ID

    df["TeamsUniqueID"] = df.apply(make_teams_set,axis=1,dict_unique_teams_ID=dict_unique_teams_ID)
    
    #Previous match results with same home and away teams - in either order of home or away
    df["FTRPrevMatchofTeams1"] = df.groupby(["TeamsUniqueID"])["FTR"].shift(1)
    df["FTRPrevMatchofTeams2"] = df.groupby(["TeamsUniqueID"])["FTR"].shift(2)
    df["FTRPrevMatchofTeams3"] = df.groupby(["TeamsUniqueID"])["FTR"].shift(3)

    #Previous half time results with same home and away teams - in either order of home or away
    df["HTRPrevMatchofTeams1"] = df.groupby(["TeamsUniqueID"])["HTR"].shift(1)
    df["HTRPrevMatchofTeams2"] = df.groupby(["TeamsUniqueID"])["HTR"].shift(2)
    df["HTRPrevMatchofTeams3"] = df.groupby(["TeamsUniqueID"])["HTR"].shift(3)

    #Previous match results with same home and away teams - in that order
    df["FTRPrevMatch1"] = df.groupby(["Teams"])["FTR"].shift(1)
    df["FTRPrevMatch2"] = df.groupby(["Teams"])["FTR"].shift(2)
    df["FTRPrevMatch3"] = df.groupby(["Teams"])["FTR"].shift(3)

    df["HRatioShotsToShotsOnTarget"] = df["HST"]/df["HS"]
    df["ARatioShotsToShotsOnTarget"] = df["AST"]/df["AS"]
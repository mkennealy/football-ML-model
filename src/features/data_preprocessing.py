import pandas as pd


def clean_data(df):

    #keep columns with fewer than 1000 rows of missing data 
    reduced_vars = []
    for k,v in dict(df.isna().sum()).items():
        if v <1000:
            reduced_vars.append(k)
    df = df[reduced_vars]


    #convert numerical variables into numeric datatype
    for col in df.columns:
        if col not in ['Div', 'Date', 'HomeTeam', 'AwayTeam','Teams','Referee','FTR','HTR',"Season"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    #remove rows that are missing a full time result (erroneous data)
    df=df[df["FTR"].isin(["H","A","D"])] 
    return df

def feature_engineering(df):
    
    df = clean_data(df)

    #add feature of teams playing in home and away position
    df["Teams"] = df["HomeTeam"] + " vs " + df["AwayTeam"]

    #add features for avg number of goals scored by team in previous 2 seasons
    #calculate average number of goals scored by home team in each season when it plays at Home and Away
    grp_goals_home_ = df.groupby(["Season","HomeTeam"]).agg({'FTHG': ['mean', 'count','sum']})
    grp_goals_home_.columns = grp_goals_home_.columns.to_flat_index().str.join('_')
    grp_goals_home_.reset_index(inplace=True)

    grp_goals_away_ = df.groupby(["Season","AwayTeam"]).agg({'FTAG': ['mean', 'count','sum']})
    grp_goals_away_.columns = grp_goals_away_.columns.to_flat_index().str.join('_')
    grp_goals_away_.reset_index(inplace=True)
    grp_goals_home_away_ = pd.merge(grp_goals_home_,grp_goals_away_,left_on =["Season","HomeTeam"],right_on = ["Season","AwayTeam"])

    #each team plays an equal number of matches at home or away
    grp_goals_home_away_["Avg_mean_Goals"] = (grp_goals_home_away_["FTHG_mean"] + grp_goals_home_away_["FTAG_mean"])/2
    grp_goals_home_away_["Avg_sum_Goals"] = (grp_goals_home_away_["FTHG_sum"] + grp_goals_home_away_["FTAG_sum"])/2
                    
    #previous season average goals per match in home/away position and average (weighted by number of matches)
    grp_goals_home_away_shifted1 = grp_goals_home_away_.groupby(["Season","HomeTeam"])[["FTHG_mean","FTAG_mean","Avg_mean_Goals","FTHG_sum","FTAG_sum","Avg_sum_Goals"]].mean().unstack().shift(1).stack()
    grp_goals_home_away_shifted2 = grp_goals_home_away_.groupby(["Season","HomeTeam"])[["FTHG_mean","FTAG_mean","Avg_mean_Goals","FTHG_sum","FTAG_sum","Avg_sum_Goals"]].mean().unstack().shift(2).stack()
    grp_goals_home_away_shifted1 = grp_goals_home_away_shifted1.add_suffix("_shift1")
    grp_goals_home_away_shifted2 = grp_goals_home_away_shifted2.add_suffix("_shift2")

    df = df.merge(grp_goals_home_away_shifted1.add_suffix("_H"),left_on=["Season","HomeTeam"],right_on=["Season","HomeTeam"],how="left")
    df = df.merge(grp_goals_home_away_shifted2.add_suffix("_H"),left_on=["Season","HomeTeam"],right_on=["Season","HomeTeam"],how="left")
    df = df.merge(grp_goals_home_away_shifted1.add_suffix("_A"),left_on=["Season","AwayTeam"],right_on=["Season","HomeTeam"],how="left")
    df = df.merge(grp_goals_home_away_shifted2.add_suffix("_A"),left_on=["Season","AwayTeam"],right_on=["Season","HomeTeam"],how="left")
    
    df["Avg_Goal_Diff_shift1"] = df["Avg_mean_Goals_shift1_H"] - df_all_seasons_reduced["Avg_mean_Goals_shift1_A"] 
    df["Avg_Goal_Diff_shift2"] = df["Avg_mean_Goals_shift2_H"] - df_all_seasons_reduced["Avg_mean_Goals_shift2_A"]
    df["Sum_Goal_Diff_shift1"] = df["Avg_sum_Goals_shift1_H"] - df_all_seasons_reduced["Avg_sum_Goals_shift1_A"] 
    df["Sum_Goal_Diff_shift2"] = df["Avg_sum_Goals_shift2_H"] - df_all_seasons_reduced["Avg_sum_Goals_shift2_A"]



    #create new features that represent the differences in the EA Sports FIFA scores for the Home and Away teams in each season
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

    return df

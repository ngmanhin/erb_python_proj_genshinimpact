import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import statsmodels.api as sm
from scipy.stats import pearsonr



pio.renderers.default = "browser"
pd.set_option('display.max_rows', 150,'display.max_columns', 120, 'display.width', 300)

'===== 01_Data Cleaning ====='
'===== Raw Data ====='
df1 = pd.read_csv("Marketing\\Genshin charac rev (by banner).csv", index_col=0)
df2 = pd.read_csv("Marketing\\genshin_characters_v1.csv", index_col=0)
df3 = pd.read_csv("Marketing\\Data Booklet.csv", index_col=0)

'=============================== df1 ==============================='
# print('df1 - Genshin charac rev (by banner)',df1.info())
# print(df1)

'===== Revenue & Avg Revenue "strip to float" =====' ### 數字唔要",", 方便計數
df1["revenue"] = df1["revenue"].str.replace(",", "")
df1["avg_revenue"] = df1["avg_revenue"].str.replace(",", "")
### 轉revenue 做 int 同 float, 方便計數
df1["revenue"] = df1["revenue"].astype(float)
df1["avg_revenue"] = df1["avg_revenue"].astype(float)
# print(df1)

'===== Split Revenue "The real Revenue of each Character" =====' ### revenue 除 banner 人數
df1["count"] = df1["5_star_characters"].str.count("&") + 1
df1["revenue"] = df1["revenue"] / df1["count"]
# print(df1["count"])
# print(df1["revenue"])
# print(df1)
# print(df1["5_star_characters"].info())

'===== Split all the character to each row =====' ### mixed banner 角色分行
df1["5_star_characters"] = df1["5_star_characters"].str.split("&")
df1 = df1.explode("5_star_characters")
# print(df1)
# print(df1["5_star_characters"].info())

'===== rename the space-bar =====' ### 角色名, 姓同名, 中間只隔一格space
df1["5_star_characters"] = df1["5_star_characters"].str.replace(" ", "")
df1["5_star_characters"] = df1["5_star_characters"].str.replace(r"([a-z])([A-Z])", r"\1 \2", regex=True)

'===== The 1st Release Characters ====='# 分開首次發銷, 同rerun 數據, 方便process
df1_1st_release = df1[~df1["5_star_characters"].str.contains("Rerun", na=False)]
# print(df1_1st_release.info(verbose=True, show_counts=True))
# print(df1_1st_release)

'===== List of Return Characters ====='
df_1_rerun = df1[df1["5_star_characters"].str.contains("Rerun", na=False)]
# print(df_1_rerun)

'===== The 3rd Return Characters (2 Characters) ====='# Dataframe with 3rd rerun only
df_3_rerun = df_1_rerun[df_1_rerun["5_star_characters"].str.contains("3rd", na=False)]
df_3_rerun['rerun'] = df_3_rerun['rerun'].map({'Y': 1, 'N': 0})
df_3_rerun['mixed'] = df_3_rerun['mixed'].map({'Y': 1, 'N': 0})
print('The 3rd Return Characters')
print(df_3_rerun)
'===== The 2nd Return Characters (9 Characters) ====='# Dataframe with 2nd rerun only
df_2_rerun = df_1_rerun[df_1_rerun["5_star_characters"].str.contains("2nd", na=False)]
df_2_rerun['rerun'] = df_2_rerun['rerun'].map({'Y': 1, 'N': 0})
df_2_rerun['mixed'] = df_2_rerun['mixed'].map({'Y': 1, 'N': 0})
print('The 2nd Return Characters')
print(df_2_rerun)
'===== The 1st Return Characters (15 Characters) ====='# Dataframe with 1st rerun only
df_1_rerun = df_1_rerun[~df_1_rerun["5_star_characters"].str.contains("2nd", na=False)]
df_1_rerun = df_1_rerun[~df_1_rerun["5_star_characters"].str.contains("3rd", na=False)]
df_1_rerun['rerun'] = df_1_rerun['rerun'].map({'Y': 1, 'N': 0})
df_1_rerun['mixed'] = df_1_rerun['mixed'].map({'Y': 1, 'N': 0})
print('The 1st Return Characters')
print(df_1_rerun)

### rRename column: Characters, rerun's revenue
df_1_rerun = df_1_rerun.rename({"rerun": "1st_ReRun", "mixed":"1st_mixed", "start_date":"1st_start_date", "end_date":"1st_end_date", "revenue": "ReRun1_revenue", "banner_days": "ReRun1_banner_days", "avg_revenue": "ReRun1_avg_revenue"},axis=1)
df_1_rerun["5_star_characters"] = df_1_rerun["5_star_characters"].str.replace("(Rerun)", "")
df_1_rerun["5_star_characters"] = df_1_rerun["5_star_characters"].str.replace("Kokomi", "Sangonomiya Kokomi")

df_2_rerun = df_2_rerun.rename({"rerun":"2nd_ReRun", "mixed":"2nd_mixed", "start_date":"2nd_start_date", "end_date":"2nd_end_date", "revenue":"ReRun2_revenue", "banner_days":"ReRun2_banner_days", "avg_revenue":"ReRun2_avg_revenue"},axis=1)
df_2_rerun["5_star_characters"] = df_2_rerun["5_star_characters"].str.replace("(2nd Rerun)", "")
df_2_rerun["5_star_characters"] = df_2_rerun["5_star_characters"].str.replace("Kokomi", "Sangonomiya Kokomi")

df_3_rerun = df_3_rerun.rename({"rerun": "3rd_ReRun", "mixed":"3rd_mixed", "start_date":"3rd_start_date", "end_date":"3rd_end_date", "revenue": "ReRun3_revenue", "banner_days": "ReRun3_banner_days", "avg_revenue": "ReRun3_avg_revenue"},axis=1)
df_3_rerun["5_star_characters"] = df_3_rerun["5_star_characters"].str.replace("(3rd Rerun)", "")

# Keep revenue banner
df_1_rerun = df_1_rerun[["5_star_characters", "1st_ReRun", "1st_start_date", "1st_end_date", "ReRun1_revenue", "1st_mixed", "ReRun1_banner_days", "ReRun1_avg_revenue"]]
df_2_rerun = df_2_rerun[["5_star_characters", "2nd_ReRun", "2nd_start_date", "2nd_end_date", "ReRun2_revenue", "2nd_mixed", "ReRun2_banner_days", "ReRun2_avg_revenue"]]
df_3_rerun = df_3_rerun[["5_star_characters", "3rd_ReRun", "3rd_start_date", "3rd_end_date", "ReRun3_revenue", "3rd_mixed", "ReRun3_banner_days", "ReRun3_avg_revenue"]]

# merge Rerun data
df1_merge = df1_1st_release.merge(df_1_rerun, on= "5_star_characters", how="left")
df1_merge = df1_merge.merge(df_2_rerun, on= "5_star_characters", how="left")
df1_merge = df1_merge.merge(df_3_rerun, on= "5_star_characters", how="left")

df1_merge = df1_merge.rename(columns = {'5_star_characters':'Name'})

df1_merge = df1_merge.fillna(0)
df1_merge['revenue_T'] = df1_merge[['revenue', 'ReRun1_revenue', 'ReRun2_revenue', 'ReRun3_revenue']].sum(axis=1)
df1_merge['banner_days_T'] = df1_merge[['banner_days', 'ReRun1_banner_days', 'ReRun2_banner_days', 'ReRun3_banner_days']].sum(axis=1)
df1_merge['avg_revenue_T'] = df1_merge[['avg_revenue', 'ReRun1_avg_revenue', 'ReRun2_avg_revenue', 'ReRun3_avg_revenue']].sum(axis=1)

'===== ReRun Times ====='
df1_merge['ReRun_Counts'] = df1_merge[['1st_ReRun', '2nd_ReRun','3rd_ReRun']].sum(axis=1)

print(df1_merge)
print(df1_merge.info(verbose=True, show_counts=True))

'===== 1st_ReRun have mixed banner ====='
print('1st_ReRun have mixed banner Character')
print(df1_merge.loc[(df1_merge['1st_ReRun'] == 1) & (df1_merge['1st_mixed'] == 1)])

'===== 2nd_ReRun have mixed banner ====='
print('2nd_ReRun have mixed banner Character')
print(df1_merge.loc[(df1_merge['2nd_ReRun'] == 1) & (df1_merge['2nd_mixed'] == 1)])

'===== 3rd_ReRun have mixed banner ====='
print('3rd_ReRun have mixed banner Character')
print(df1_merge.loc[(df1_merge['3rd_ReRun'] == 1) & (df1_merge['3rd_mixed'] == 1)])
'=============================== df2 ==============================='
# print('df2 - Genshin charac rev (by charac)',df2.info())
df2 = df2.reset_index()
# print(df2.index)
df2 = df2.rename(columns = {'character_name':'Name'})
df2["Name"] = df2["Name"].str.replace(" ", "")
df2["Name"] = df2["Name"].str.replace(r"([a-z])([A-Z])", r"\1 \2", regex=True)
# print(df2.info())
# print(df2.describe())
# print(df2)
'=============================== df3 ==============================='
# print('df3 - Data Booklet',df3.info())
df3["Name"] = df3["Name"].str.replace(" ", "")
df3["Name"] = df3["Name"].str.replace(r"([a-z])([A-Z])", r"\1 \2", regex=True)
df3 = df3.rename(columns = {'Update Playable':'version'})
# print(df3.info())
# print(df3)
# print(df3.isna().sum())
# print(df3.describe())


'=============================== merge df1 + df2 + df3 ==============================='
'====== merge df1 + 2 ======'
df1_2 = pd.merge(df1_merge, df2, how='left', on='Name')
# print(df1_2)
# print(df1_2.info())


'====== merge df1 + 3 ======'
df1_3 = pd.merge(df1_merge, df3, how='left', on='Name')
# print(df1_3)
# print(df1_3.info())


'====== merge df1 + 2 + 3 ======'
df1_2_3 = pd.merge(df1_3, df2, how='left', on='Name')
# print(df1_2_3)
# print(df1_2_3.info(verbose=True, show_counts=True))


'===== merge df=============='
'====== merge df1 + 2 ======'
df1_2 = pd.merge(df1_merge, df2, how='left', on='Name')
# print(df1_2)
# print(df1_2.info())

'====== merge df1 + 3 ======'
df1_3 = pd.merge(df1_merge, df3, how='left', on='Name')
# print(df1_3)
# print(df1_3.info())

'====== merge df1 + 2 + 3 ======'
df1_2_3 = pd.merge(df1_3, df2, how='left', on='Name')
# print(df1_2_3)
# print(df1_2_3.info(verbose=True, show_counts=True))

### DROP COLUMN
df1_2_3 = df1_2_3.drop(columns=['arkhe','voice_en', 'voice_cn', 'birthday', 'region', 'Element'])
# print(df1_2_3.info(verbose=True, show_counts=True))


####Ascension material
df1_2_3_material = df1_2_3[['Name', 'ascension_specialty',
                 'ascension_boss_material', 'ascension_material_0-2', 'ascension_material_2-4', 'ascension_material_4-6',
                 'ascension_gem_0-1', 'ascension_gem_1-3', 'ascension_gem_3-5', 'ascension_gem_5-6']]
# print(df1_2_3_material)
### DROP 突破素材
df1_2_3 = df1_2_3.drop(columns=['ascension_specialty','ascension_boss_material','ascension_material_0-2','ascension_material_2-4','ascension_material_4-6',
                             'ascension_gem_0-1','ascension_gem_1-3','ascension_gem_3-5','ascension_gem_0-1','ascension_gem_5-6',])
# print(df1_2_3.info(verbose=True, show_counts=True))

###  技能素材
df1_2_3_talent = df1_2_3[['Name', 'talent_material_1-2', 'talent_material_2-6', 'talent_material_6-10',
                   'talent_book_1-2', 'talent_book_2-3', 'talent_book_3-4', 'talent_book_4-5', 'talent_book_5-6', 'talent_book_6-7', 'talent_book_7-8', 'talent_book_8-9', 'talent_book_9-10',
                   'talent_weekly']]
print(df1_2_3_talent)
### DROP
df1_2_3 = df1_2_3.drop(columns=['talent_material_1-2', 'talent_material_2-6', 'talent_material_6-10',
                   'talent_book_1-2', 'talent_book_2-3', 'talent_book_3-4', 'talent_book_4-5', 'talent_book_5-6', 'talent_book_6-7', 'talent_book_7-8', 'talent_book_8-9', 'talent_book_9-10',
                   'talent_weekly'])


# print(df1_2_3[['version_name','version','Element','star_rarity','limited','weapon_type','Weapon','start_date','release_date',]])
df1_2_3 = df1_2_3.drop(columns=['weapon_type','release_date'],)


'=============================== final print ==============================='
# print(df1_2_3)
# print(df1_2_3.info(verbose=True, show_counts=True))

df1_2_3_voice = df1_2_3.filter(like="voice").columns.tolist()  ###聲優

'===== ascension bonus ====='
df1_2_3_special = df1_2_3.filter(like="special").columns.tolist()
# print(df1_2_3_special)
df1_2_3_A_status = df1_2_3[['Name', 'ascension_special_stat', 'special_0', 'special_1', 'special_2', 'special_3', 'special_4', 'special_5', 'special_6']]  ####突破BONUS
# print(df1_2_3_A_status)
# print(df1_2_3_A_status['ascension_special_stat'].value_counts())

'===== Copy ====='
dfc = df1_2_3.copy()
# print(dfc)
# print(dfc.info(verbose=True, show_counts=True))

'====== The max revenue character ======'
# print('The best total sales character: ',dfc.loc[df1_2_3.revenue_T == dfc.revenue_T.max()]['Name'])
# print('The best 1st sales character: ',df1_2_3.loc[df1_2_3.revenue == df1_2_3.revenue.max()]['Name'])

'====== The Top Votes character ======'
# print('The best Votes character: ',dfc.loc[df1_2_3.Votes == dfc.Votes.max()]['Name'])

'====== The max hp character ======'
# print('The Max hp character: ',df1_2_3.loc[df1_2_3.hp_90_90 == df1_2_3.hp_90_90.max()]['Name'])
'====== The max atk character ======'
# print('The Max atk character: ',df1_2_3.loc[df1_2_3.atk_90_90 == df1_2_3.atk_90_90.max()]['Name'])
'====== The max def character ======'
# print('The Max def character: ',df1_2_3.loc[df1_2_3.def_90_90 == df1_2_3.def_90_90.max()]['Name'])


'===== Drawings ===== Drawings ===== Drawings ===== Drawings ===== Drawings ===== Drawings ===== Drawings ====='


'===== 02_Votes ====='
Votes = df1_2_3[['Name', 'revenue_T', 'Votes', 'Gender', 'Region', 'Weapon', 'vision', 'voice_jp']]
# print(Votes)

## Region Compare
# print(dfc['Region'].value_counts())
'===== Region (Vote vs Revenue) ====='
# Set up 1 row 2 columns of drawing
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                    subplot_titles=["Region vs Revenue", "Region vs Votes"])

region_order = ["North", "South", "East", "West"]
START_ANGLE = 90  # Set up the angel
# built the drawing with Trace
fig1 = px.pie(dfc, names="Region", values="revenue_T", color="Region",
              category_orders={"Region": region_order})
fig2 = px.pie(dfc, names="Region", values="Votes", color="Region",
              category_orders={"Region": region_order})

# Add Trace
fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)

# Edit the fonts
fig.update_traces(textfont_size=16, sort=False)
fig.update_layout(title_text="Region Performance Comparison",
                  title_font_size=24,
                  legend_font_size=16)

fig.show()

## Weapon Compare
'===== Weapon, Element vs Revenue -Sunburst-01 ====='
# print(dfc['vision'].value_counts())
# print(dfc['Weapon'].value_counts())
'===== Weapon (Vote vs Revenue) -Sunburst-01 ====='
# 1.Set up color map to ensure 2 drawings same color
color_map = {"Sword": "#636EFA",
             "Claymore": "#EF553B",
             "Polearm": "#00CC96",
             "Bow": "#AB63FA",
             "Catalyst": "#FFA15A",}

# 2. Set up the DataFrame location
weapon_order = ["Sword", "Claymore", "Polearm", "Bow", "Catalyst"]
dfc["Weapon"] = pd.Categorical(dfc["Weapon"], categories=weapon_order, ordered=True)
dfc = dfc.sort_values("Weapon")

# 3. To make two Sunburst
fig1 = px.pie if False else px.sunburst(dfc,
                                        path=["Weapon", "vision"],
                                        values="revenue_T",
                                        color="Weapon",
                                        color_discrete_map=color_map,
                                        )

fig2 = px.sunburst(dfc,
                   path=["Weapon", "vision"],
                   values="Votes",
                   color="Weapon",
                   color_discrete_map=color_map,
                   )

# 4. Set 1 row 2columns Subplot
fig = make_subplots(rows=1, cols=2,
                    specs=[[{"type": "domain"}, {"type": "domain"}]],  # 必須指定為 domain
                    subplot_titles=["Weapon vs Revenue", "Weapon vs Votes"],
                    )

# 5. Add 2 Trace to subplot
fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)

# 6. Set up the fonts
fig.update_traces(textfont_size=20,
                  insidetextfont_size=20,
                  sort=False,         # Stop the sort
                  rotation=90,        # make sure the rotation
                  )

fig.update_layout(title_text="Weapon Stats Comparison (Revenue vs Votes)",
                  title_font_size=24,
                  legend_font_size=16,
                  )

fig.show()

'===== Weapon vs Revenue -Boxplot-01 ====='
sns.boxplot(data=dfc, x='Weapon', y='revenue_T',
            hue='Gender',
            palette=["m", 'g'],)
plt.title('Weapon vs Revenue' ,fontsize=20)
plt.xlabel('Weapon',fontsize=15)
plt.ylabel('Revenue',fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

'===== Weapon vs Votes -Boxplot-01 ====='
sns.boxplot(data=dfc, x='Weapon', y='Votes',
            hue='Gender',
            palette=["m", 'g'],)
plt.title('Weapon vs Votes' ,fontsize=20)
plt.xlabel('Weapon',fontsize=15)
plt.ylabel('Votes',fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

### Character Gender
'===== Gender vs Revenue - Sunburst-01 ====='
fig = px.sunburst(
    dfc,
    path=['Gender', 'model', 'Name'],
    values='revenue_T',
    title='Revenue vs Gender with Model Type',)

fig.update_traces(textfont=dict(size=18))
fig.update_layout(title_font_size=20,
                  font=dict(size=16),  )

fig.show()

'===== Gender vs Vote - Sunburst-01 ====='
fig = px.sunburst(
    dfc,
    path=['Gender', 'model', 'Name'],
    values='Votes',
    title='Votes vs Gender with Model Type')

fig.update_traces(textfont=dict(size=18))
fig.update_layout(title_font_size=20,
                  font=dict(size=16),  )

fig.show()

'===== Votes vs Revenue (1st Release / no rerun) -Scatter plot-01 ====='
fig = px.scatter(dfc, x='Votes', y='revenue',
                 color='banner_days', facet_col='Region', facet_row='Gender')
fig.show()
'===== Votes vs Revenue -Scatter plot-02 ====='
X = sm.add_constant(dfc['Votes'])  # Adds the intercept term
y = dfc['revenue_T']
model = sm.OLS(y, X).fit()

r_squared = model.rsquared

g = sns.lmplot(data=dfc, x='Votes', y='revenue_T', hue='Gender')

plt.title('Votes vs Revenue', fontsize=20)
plt.xlabel('Votes', fontsize=15)
plt.ylabel('Revenue', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Display R-squared on the plot
plt.gca().text(
    0.05,
    0.95,
    f'$R^2 = {r_squared:.3f}$',
    transform=plt.gca().transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
)

plt.show()
'===== Votes vs Revenue (Show the Total rerun days, votes & revenue) -Scatter plot-03 ====='
fig = px.scatter(dfc, x='Votes', y='revenue_T',
                 color='Gender',
                 size='banner_days_T', hover_data=['Name'],
                 title='Votes vs Revenue')
fig.update_layout(
    title_font_size=30,          # Main title
    xaxis_title_font_size=20,    # X-axis title
    yaxis_title_font_size=20,    # Y-axis title
    xaxis_tickfont_size=15,      # X-axis tick labels
    yaxis_tickfont_size=15,      # Y-axis tick labels
    legend_font_size=15          # Legend text
)
fig.show()


## Stats have any relationship with Revenue?
'===== 03_Character Stats ====='
Stats = df1_2_3[['Name', 'revenue_T', 'Votes', 'Weapon', 'hp_90_90', 'atk_90_90', 'def_90_90', 'ascension_special_stat']]
# print(Votes)


'===== Status revenue ====='
revenue_90 = dfc[['atk_90_90', 'hp_90_90', 'def_90_90', 'revenue_T']]
# print(revenue_90)
revenue_01 = dfc[['atk_1_20', 'hp_1_20', 'def_1_20', 'revenue_T']]
# print(revenue_01)
'===== Ascension Stat vs Revenue - Sunburst ====='
fig = px.sunburst(dfc, path=['ascension_special_stat', 'Weapon', 'vision',], values='revenue_T',
                  title='Ascension Stat vs Revenue')
fig.show()

'====================== Status Box plot ================================='
df000 = df1_2_3[["Name", "Gender", "revenue", "banner_days", "hp_90_90", "atk_90_90", "def_90_90", "ascension_special_stat"]]
df000["Daily Revenue"] =  round(df000["revenue"] / df000["banner_days"],2)

def ability(a):
  if "Energy Recharge" in a:
      return "Utility"
  elif "HP" in a or "DEF" in a or "Elemental" in a:
      return "Survival"
  else:
      return "Offensive"

df000["Ability"] = df000['ascension_special_stat'].apply(ability)

all = df000.groupby("Ability")
count = all["Name"].count().reset_index(name="Char_Count")
sum   = all["Daily Revenue"].sum().reset_index(name="Total_Daily_Rev")
mean  = all["Daily Revenue"].mean().reset_index(name="Mean_Rev")

summary = pd.merge(count, sum, on="Ability")
summary = pd.merge(summary, mean, on="Ability")
# print(round(summary))

sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 10))
sns.boxplot(data=df000,
          x="Ability",
          y="Daily Revenue",
          hue="Ability")
plt.title("Impact of Skill Types on Daily Revenue", fontsize=30, fontweight="bold")
plt.xlabel("Skill Type", fontsize=30)
plt.ylabel("Daily Revenue", fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.tight_layout()
plt.show()

'====================== HP ====================================='

sns.set_theme()
sns.lmplot(data=df000,
          x="hp_90_90",
          y="Daily Revenue",
          hue="Gender",
          palette={"Male": "blue", "Female": "red"},
          legend=False)

plt.title("Impact of 90lv HP on Daily Revenue", fontsize=15)
plt.xlabel("HP", fontsize=15)
plt.ylabel("Daily Revenue (USD)", fontsize=15)
plt.xlim(9000, 16000)
plt.ylim(0, 1700000)
plt.tight_layout()
plt.show()

Male_HP = df000[df000['Gender'] == 'Male']
Female_HP = df000[df000['Gender'] == 'Female']
r_male, p_male = pearsonr(Male_HP["hp_90_90"], Male_HP["Daily Revenue"])
r_female, p_female = pearsonr(Female_HP["hp_90_90"], Female_HP["Daily Revenue"])
# print("Male_HP:", "r=", round(r_male,2), "p=", round(p_male,2))
# print("Female_HP:", "r=", round(r_female,2), "p=", round(p_female,2))

'====================== ATK ====================================='
sns.set_theme()
sns.lmplot(data=df000,
          x="atk_90_90",
          y="Daily Revenue",
          hue="Gender",
          palette={"Male": "blue", "Female": "red"},
          legend=False)

plt.title("Impact of 90lv ATK on Daily Revenue", fontsize=15)
plt.xlabel("ATK", fontsize=15)
plt.ylabel("Daily Revenue (USD)", fontsize=15)
plt.xlim(100, 350)
plt.ylim(0, 1700000)
plt.tight_layout()
plt.show()

Male_ATK = df000[df000['Gender'] == 'Male']
Female_ATK = df000[df000['Gender'] == 'Female']
r_male_atk, p_male_atk = pearsonr(Male_ATK["atk_90_90"], Male_ATK["Daily Revenue"])
r_female_atk, p_female_atk = pearsonr(Female_ATK["atk_90_90"], Female_ATK["Daily Revenue"])
# print("Male_ATK:", "r=", round(r_male_atk,2), "p=", round(p_male_atk,2))
# print("Female_ATK:", "r=", round(r_female_atk,2), "p=", round(p_female_atk,2))

'====================== DEF ====================================='
from scipy.stats import pearsonr
sns.set_theme()
sns.lmplot(data=df000,
          x="def_90_90",
          y="Daily Revenue",
          hue="Gender",
          palette={"Male": "blue", "Female": "red"},
          legend=False)

plt.title("Impact of 90lv DEF on Daily Revenue", fontsize=15)
plt.xlabel("DEF", fontsize=15)
plt.ylabel("Daily Revenue (USD)", fontsize=15)
plt.xlim(500, 1000)
plt.ylim(0, 1700000)
plt.legend(title="Gender", loc="upper right", fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.show()

Male_DEF = df000[df000['Gender'] == 'Male']
Female_DEF = df000[df000['Gender'] == 'Female']
r_male_def, p_male_def = pearsonr(Male_DEF["def_90_90"], Male_DEF["Daily Revenue"])
r_female_def, p_female_def = pearsonr(Female_DEF["def_90_90"], Female_DEF["Daily Revenue"])
# print("Male_DEF:", "r=", round(r_male_def,2), "p=", round(p_male_def,2))
# print("Female_DEF:", "r=", round(r_female_def,2), "p=", round(p_female_def,2))


'===== Revenue with the Marketing ====='
'===== All Revenue (BY Characters) - grouped bar plot(by ground) ====='
dfc = dfc.sort_values('revenue_T')


fig = px.bar(dfc,
            x='Name',
            y=['revenue', 'ReRun1_revenue', 'ReRun2_revenue', 'ReRun3_revenue'],
            )


fig.update_layout(title=dict(text='All Characters Revenue', font=dict(size=35)), xaxis=dict(
                 title=dict(text='Characters', font=dict(size=30)),  # Title dict ends here
                 tickfont=dict(size=20)                              # Tickfont goes directly under xaxis
                 ),yaxis=dict(title=dict(text='Total Revenue (Million)', font=dict(size=30)),tickfont=dict(size=20)
                 ),legend_title=dict(text='Launched',font=dict(size=30)),legend=dict(font=dict(size=20)),
                 barmode="group")

fig.show()

'===== Total Revenue (BY Characters) - grouped bar plot ====='
dfc = dfc.sort_values('revenue_T')


fig = px.bar(dfc,
            x='Name',
            y=['revenue', 'ReRun1_revenue', 'ReRun2_revenue', 'ReRun3_revenue'],
            )

fig.update_layout(title=dict(text='All Characters Total Revenue', font=dict(size=35)), xaxis=dict(
                 title=dict(text='Characters', font=dict(size=30)),  # Title dict ends here
                 tickfont=dict(size=20)                              # Tickfont goes directly under xaxis
                 ),yaxis=dict(title=dict(text='Total Revenue (Million)', font=dict(size=30)),tickfont=dict(size=20)
                 ),legend_title=dict(text='Launched',font=dict(size=30)),legend=dict(font=dict(size=20)),
                 )

fig.show()

'===== Create a DataFrame sorted by Date for all Revenue ====='
dfd = dfc[["Name","revenue","start_date","banner_days", "ReRun1_revenue","1st_start_date","ReRun1_banner_days",
         "ReRun2_revenue","2nd_start_date","ReRun2_banner_days","ReRun3_revenue","3rd_start_date","ReRun3_banner_days","Votes",]]

df_d = pd.melt(dfc,
             id_vars=["Name", "revenue", "banner_days", "ReRun1_revenue", "ReRun1_banner_days", "ReRun2_revenue", "ReRun2_banner_days", "ReRun3_revenue", "ReRun3_banner_days"],
             value_vars=["start_date", "1st_start_date", "2nd_start_date", "3rd_start_date"],
             var_name="Date_Source",
             value_name="Date",)


# 2. Filter out rows where Date is 0 or NaN (keeping all valid date rows)
df_a = df_d[(df_d["Date"] != 0) & (df_d["Date"].notna())].copy()


# 3. For '3rd_start_date' rows, keep 'ReRun3_revenue' value and set other revenues to 0
mask_3rd = df_a["Date_Source"] == "3rd_start_date"
df_a.loc[mask_3rd, ["revenue", "banner_days", "ReRun1_revenue", "ReRun1_banner_days","ReRun2_revenue", "ReRun2_banner_days"]] = 0
mask_2nd = df_a["Date_Source"] == "2nd_start_date"
df_a.loc[mask_2nd, ["revenue", "banner_days", "ReRun1_revenue", "ReRun1_banner_days", "ReRun3_revenue", "ReRun3_banner_days"]] = 0
mask_1st = df_a["Date_Source"] == "1st_start_date"
df_a.loc[mask_1st, ["revenue", "banner_days", "ReRun2_revenue", "ReRun2_banner_days", "ReRun3_revenue", "ReRun3_banner_days"]] = 0
mask_1release = df_a["Date_Source"] == "start_date"
df_a.loc[mask_1release, ["ReRun1_revenue", "ReRun1_banner_days", "ReRun2_revenue", "ReRun2_banner_days", "ReRun3_revenue", "ReRun3_banner_days"]] = 0


# 4. Convert Date to datetime format safely
df_a["Date"] = pd.to_datetime(df_a["Date"], errors="coerce", dayfirst=True)


# 5. Sort chronologically by Date
df_date = (df_a.dropna(subset=["Date"]).sort_values(by="Date", ascending=True).reset_index(drop=True))
# print(df_date)
# print(df_date[df_date["Name"] == "Venti"])
# print(df_date["Date"].value_counts())


rev_cols = ["revenue", "ReRun1_revenue", "ReRun2_revenue", "ReRun3_revenue"]
df_date["Revenue"] = df_date[rev_cols].sum(axis=1)


# 2. Format Date as a string for clean category spacing on the x-axis
df_date["Date_Str"] = df_date["Date"].dt.strftime("%Y-%m-%d")


'===== Create the bar plot ====='
plt.figure(figsize=(12, 6))
sns.barplot(data=df_date, x="Date_Str", y="Revenue", hue="Date_Source")

plt.xticks(rotation=45, ha="right",fontsize=10)
plt.yticks(fontsize=10)
plt.title("Revenue Timeline by Event Type",fontsize=20)
plt.xlabel("Date",fontsize=20)
plt.ylabel("Revenue",fontsize=20)
plt.legend(title="Event Type",fontsize=10)
plt.tight_layout()
plt.show()

'===== All Revenue - LineChart ====='
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_date, x="Date_Str", y="Revenue", hue="Date_Source", marker="o")

plt.xticks(rotation=45, ha="right", fontsize=13)
plt.yticks(fontsize=13)
plt.title("Revenue Timeline by Event Type", fontsize=20)
plt.xlabel("Date", fontsize=18)
plt.ylabel("Revenue", fontsize=18)
plt.legend(title="Event Type", fontsize=15)
plt.tight_layout()
plt.show()

df_date["Date"] = pd.to_datetime(df_date["Date_Str"])
df_date["Date_Ordinal"] = df_date["Date"].map(pd.Timestamp.toordinal)

'===== All Revenue with Vote - Heatmap ====='
All_Revenue = dfc[["revenue", "banner_days", "ReRun1_revenue", "ReRun1_banner_days", "ReRun2_revenue", "ReRun2_banner_days", "ReRun3_revenue", "ReRun3_banner_days", "Votes", ]]


sns.set_theme()
rCorr = All_Revenue.corr(numeric_only=True)
print(rCorr)
sns.heatmap(rCorr, annot=True, cmap = 'BuPu', lw=1)


plt.tight_layout()
plt.xticks(rotation=45, ha='right', fontsize=11.5)
plt.yticks(fontsize=11.5)
plt.show()

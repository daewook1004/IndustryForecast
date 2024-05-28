import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
import numpy as np 
from statsmodels.stats.proportion import proportions_ztest
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.stats import ranksums

df = pd.read_csv('industry_concat.csv')



region_sum_2022 = df.groupby('Region')['2022_w'].sum()
region_sum_2017 = df.groupby('Region')['2017_w'].sum()
labeling = ['I' if (region_sum_2022[region] - region_sum_2017[region]) > 0 else 'D' for region in df['Region']]
df['labeling'] = labeling
df['Industry Distribution'] = df.apply(lambda row: row['2022_w'] / region_sum_2022[row['Region']], axis=1)

df.to_csv('df_industry.csv', index=False)











####################################################그래프 #############################################################


Region_list = df['Region'].unique().tolist()

for region in Region_list:
    df_graph = df[df['Region'] == region]
    font_path = 'NanumGodic/NanumGothic.ttf'
    # 폰트 설정
    font_prop = font_manager.FontProperties(fname=font_path)
   
    # 그래프 그리기
    plt.figure(figsize=(12, 6))
    bars= plt.bar(df_graph['Industry'], df_graph['Industry Distribution'])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

    plt.xlabel('산업종류', fontproperties=font_prop)
    plt.ylabel('비율', fontproperties=font_prop)
    plt.title(f'{region} 2022년 산업별 비율', fontproperties=font_prop)
    plt.tight_layout()

    filename = f'graphfolder/{region}_2022_industry_distribution.png'
    plt.savefig(filename, dpi=300)
    plt.close()  






############################### W증가 지역과 감소지역의 산업 비율 데이터 프레임##########

industry_2022_D = df[df['labeling'] == 'D'].groupby('Industry')['2022_w'].sum()
industry_2022_I = df[df['labeling'] == 'I'].groupby('Industry')['2022_w'].sum()

# 전체 2022_w의 합 계산
total_2022_D = industry_2022_D.sum()
total_2022_I = industry_2022_I.sum()

# 새로운 데이터프레임 생성
df_industry_distribution = pd.DataFrame({
    'Industry': industry_2022_D.index,
    '2022_D_Distribution': industry_2022_D,
    '2022_I_Distribution': industry_2022_I 
}).reset_index(drop=True)


print(df_industry_distribution)


observed_data = df_industry_distribution[['2022_D_Distribution', '2022_I_Distribution']].values
print(observed_data)
# 카이제곱 검정 수행
chi2_stat, p_val, dof, expected = chi2_contingency(observed_data)
print("Chi2 statistic:", chi2_stat)
print("P-value:", p_val)
print("dof", dof)
print("expected", expected)



df_industry_distribution_graph = pd.DataFrame({
    'Industry': industry_2022_D.index,
    '2022_D_Distribution': industry_2022_D / total_2022_D,
    '2022_I_Distribution': industry_2022_I / total_2022_I
}).reset_index(drop=True)


print(df_industry_distribution_graph)





import matplotlib.pyplot as plt
import numpy as np

# 막대그래프를 그릴 데이터프레임에서 'Industry'를 x축으로, '2022_D_Distribution'와 '2022_I_Distribution'를 y축으로 선택합니다.
industries = df_industry_distribution_graph['Industry']
distribution_2022_D = df_industry_distribution_graph['2022_D_Distribution']
distribution_2022_I = df_industry_distribution_graph['2022_I_Distribution']

# 각 막대그래프의 위치를 계산합니다.
bar_width = 0.35
x = np.arange(len(industries))

# 막대그래프를 그립니다.
plt.figure(figsize=(12, 6))
plt.bar(x - bar_width/2, distribution_2022_D, bar_width, label='2022_Decrease')
plt.bar(x + bar_width/2, distribution_2022_I, bar_width, label='2022_Increase')

# x축 라벨과 y축 라벨, 그리고 범례를 추가합니다.
plt.xlabel('Industry')
plt.ylabel('Distribution')
plt.title('Distribution Comparison by Worker growth')
plt.xticks(x, industries)
plt.legend()

# 그래프를 보여줍니다.
plt.tight_layout()
filename = 'graphfolder/decrease_increase_2022_industry_distribution.png'
plt.savefig(filename, dpi=300)
plt.close()  



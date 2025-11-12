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



# ==========================================
# [1] 데이터 로드 및 전처리
# ==========================================
# 산업별·지역별 근로자 수 데이터 로드
df = pd.read_csv('dataset/industry_concat.csv')

# 2017년과 2022년 지역별 근로자 총합 계산
region_sum_2022 = df.groupby('Region')['2022_w'].sum()
region_sum_2017 = df.groupby('Region')['2017_w'].sum()

# 지역별 근로자 변화량을 기준으로 증가(I) / 감소(D) 라벨 지정
labeling = ['I' if (region_sum_2022[region] - region_sum_2017[region]) > 0 else 'D' for region in df['Region']]
df['labeling'] = labeling

# 각 지역 내에서 산업별 근로자 비율 계산
# Industry Distribution = (해당 산업의 근로자 수) / (해당 지역의 전체 근로자 수)
df['Industry Distribution'] = df.apply(lambda row: row['2022_w'] / region_sum_2022[row['Region']], axis=1)


#### 중간 데이터 저장 
df.to_csv('df_industry.csv', index=False)

# ==========================================
# [2] 지역별 산업 분포 시각화
# ==========================================


Region_list = df['Region'].unique().tolist()

for region in Region_list:
    df_graph = df[df['Region'] == region]
    # 한글 폰트 설정 (NanumGothic.ttf 파일 필요)
    font_path = 'NanumGodic/NanumGothic.ttf'
    # 폰트 설정
    font_prop = font_manager.FontProperties(fname=font_path)
   
    # 지역별 산업 비율 막대그래프 생성
    plt.figure(figsize=(12, 6))
    bars= plt.bar(df_graph['Industry'], df_graph['Industry Distribution'])
    
    # 각 막대 위에 비율값 표시 (소수점 둘째 자리까지)
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







# ==========================================
# [3] 증가(I) 지역 vs 감소(D) 지역 산업 구조 비교
# ==========================================

# 증가(D) 지역과 감소(I) 지역의 산업별 2022년 근로자 수 합산
industry_2022_D = df[df['labeling'] == 'D'].groupby('Industry')['2022_w'].sum()
industry_2022_I = df[df['labeling'] == 'I'].groupby('Industry')['2022_w'].sum()




# 전체 합계 (비율 계산용)
total_2022_D = industry_2022_D.sum()
total_2022_I = industry_2022_I.sum()

# 두 그룹(증가 vs 감소) 간의 산업별 분포 테이블 생성
df_industry_distribution = pd.DataFrame({
    'Industry': industry_2022_D.index,
    '2022_D_Distribution': industry_2022_D,
    '2022_I_Distribution': industry_2022_I 
}).reset_index(drop=True)





# ==========================================
# [4] 카이제곱 검정 (Chi-Square Test)
# ==========================================

# 관측 데이터 행렬 구성: [감소지역, 증가지역] × [산업]
observed_data = df_industry_distribution[['2022_D_Distribution', '2022_I_Distribution']].values
print(observed_data)
# 카이제곱 검정 수행 (증가지역과 감소지역의 산업 분포 차이 유의성 검정)
chi2_stat, p_val, dof, expected = chi2_contingency(observed_data)
print("Chi2 statistic:", chi2_stat)
print("P-value:", p_val)
print("dof", dof)
print("expected", expected)


# ==========================================
# [5] 산업 비율 그래프 (비교 시각화)
# ==========================================
# 비율 기준으로 재정규화 (산업별 비율 합 = 1)
df_industry_distribution_graph = pd.DataFrame({
    'Industry': industry_2022_D.index,
    '2022_D_Distribution': industry_2022_D / total_2022_D,
    '2022_I_Distribution': industry_2022_I / total_2022_I
}).reset_index(drop=True)


print(df_industry_distribution_graph)




# ------------------------------------------
# 두 그룹의 산업 비율 비교 그래프 생성
# ------------------------------------------



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



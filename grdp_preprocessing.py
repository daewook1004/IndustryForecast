import pandas as pd
df = pd.read_csv('GRDP/US_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
df['Region'] = '울산광역시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농림어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공행정, 국방 및 사회보장 행정']
df.to_csv('GRDP/US_GRDP_Modify.csv', index=False)



df = pd.read_csv('GRDP/Seoul_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
df['Region'] = '서울특별시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농·림·어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타 서비스업', 'R')
df = df[df['Industry'] != '공공행정,국방 및 사회보장행정']
print(df[0:20])
df.to_csv('GRDP/Seoul_GRDP_Modify.csv', index=False)




df = pd.read_csv('GRDP/KK_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
# df['Region'] = '서울특별시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/KK_GRDP_Modify.csv', index=False)




df = pd.read_csv('GRDP/KB_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
# df['Region'] = '서울특별시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업,어업 및 임업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기가스증기및공기조절업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매및소매업', 'G')
df['Industry'] = df['Industry'].replace('운수및창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육서비스업(정부)', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지서비스업', 'Q')
df['Industry'] = df['Industry'].replace('예술, 스포츠 및 여가관련 서비스', 'R')
df = df[df['Industry'] != '공공행정,국방및사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/KB_GRDP_Modify.csv', index=False)



df = pd.read_csv('GRDP/JN_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
# df['Region'] = '서울특별시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/JN_GRDP_Modify.csv', index=False)



df = pd.read_csv('GRDP/JJ_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
# df['Region'] = '서울특별시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/JJ_GRDP_Modify.csv', index=False)



df = pd.read_csv('GRDP/JB_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
# df['Region'] = '서울특별시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농림어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/JB_GRDP_Modify.csv', index=False)



df = pd.read_csv('GRDP/IC_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
df['Region'] = '인천광역시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/IC_GRDP_Modify.csv', index=False)



df = pd.read_csv('GRDP/GW_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
# df['Region'] = '인천광역시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/GW_GRDP_Modify.csv', index=False)




df = pd.read_csv('GRDP/GN_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
# df['Region'] = '인천광역시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/GN_GRDP_Modify.csv', index=False)


df = pd.read_csv('GRDP/GJ_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
df['Region'] = '광주광역시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/GJ_GRDP_Modify.csv', index=False)



df = pd.read_csv('GRDP/DJ_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
df['Region'] = '대전광역시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/DJ_GRDP_Modify.csv', index=False)




df = pd.read_csv('GRDP/DG_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
df['Region'] = '대구광역시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/DG_GRDP_Modify.csv', index=False)

df = pd.read_csv('GRDP/CN_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
# df['Region'] = '대구광역시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/CN_GRDP_Modify.csv', index=False)


df = pd.read_csv('GRDP/CB_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
# df['Region'] = '대구광역시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/CB_GRDP_Modify.csv', index=False)




df = pd.read_csv('GRDP/BS_GRDP.csv', encoding='cp949')
df = df[1:]
df.columns = ['Region', 'Industry', 'GRDP', 'Factor_Income']
print(df[0:17])
df['Region'] = '부산광역시' + ' ' + df['Region']

df['Industry'] = df['Industry'].replace('농업, 임업 및 어업', 'A')
df['Industry'] = df['Industry'].replace('광업', 'B')
df['Industry'] = df['Industry'].replace('제조업', 'C')
df['Industry'] = df['Industry'].replace('전기, 가스, 증기 및 공기 조절 공급업', 'D')
df['Industry'] = df['Industry'].replace('건설업', 'F')
df['Industry'] = df['Industry'].replace('도매 및 소매업', 'G')
df['Industry'] = df['Industry'].replace('운수 및 창고업', 'H')
df['Industry'] = df['Industry'].replace('숙박 및 음식점업', 'I')
df['Industry'] = df['Industry'].replace('정보통신업', 'J')
df['Industry'] = df['Industry'].replace('금융 및 보험업', 'K')
df['Industry'] = df['Industry'].replace('부동산업', 'L')
df['Industry'] = df['Industry'].replace('사업서비스업', 'N')
df['Industry'] = df['Industry'].replace('교육 서비스업', 'P')
df['Industry'] = df['Industry'].replace('보건업 및 사회복지 서비스업', 'Q')
df['Industry'] = df['Industry'].replace('문화 및 기타서비스업', 'R')
df = df[df['Industry'] != '공공 행정, 국방 및 사회보장 행정']
df['GRDP'] = df['GRDP'].replace('-', 0)
df['Factor_Income'] = df['Factor_Income'].replace('-', 0)

print(df[0:20])
df.to_csv('GRDP/BS_GRDP_Modify.csv', index=False)







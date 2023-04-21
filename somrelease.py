import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import streamlit as st
from minisom import MiniSom
from pylab import plasma, pcolor, colorbar, plot, show

un  = pd.read_csv('univer_uniq.csv')
un = un.drop(columns=['Unnamed: 0','federal_district','region_code','okato','id', 'year'])


target = un.drop(columns=['federal_district_short','region_name','name','name_short'])


label = un.region_name
target = target.fillna(target.median())
train = StandardScaler().fit_transform(target.values)
data = train


som_shape = (11, 11)
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=1.5, learning_rate=.7, 
              neighborhood_function='gaussian', random_seed=0)

#som.train_batch(data, 1000, verbose=True)
som.train(data, 1000, verbose=True)


arl = []
for i, x in enumerate(train):
    arl.append(str(som.winner(x)[0])  + str(som.winner(x)[1]))


dt = pd.DataFrame(arl)
ans = target
ans['cluster'] = dt[0]
ans['region_name'] = un.region_name
ans['name_short'] = un.name_short
ans['name'] = un.name


st.title('Альтернативные университеты')
st.markdown('_Пример реализации самоорганизующихся карт Кохонена для кластеризации вузов РФ.  Полученные кластеры могут быть использованы для нового способа ранжирования образовательных организаций и построения обоснованной рекомендательной системы для абитуриентов._')

option_city = st.selectbox(
    'Выбирите город целевого ВУЗа',
    (ans.region_name.unique()))
    
    
option_univer = st.selectbox(
    'Выбирите ВУЗ',
    (ans[ans.region_name == option_city].name_short.tolist()))

fin = ans[ans.region_name == option_city]    
option_cluster = fin[fin.name_short == option_univer].cluster.tolist()[0]
    
    

outer_df = ans[ans.cluster == option_cluster]
outer_df = outer_df.drop(outer_df[outer_df.name_short == option_univer].index)
st.divider()
st.markdown('**Всего найдено ' + str(outer_df.shape[0]) +' альтернативных вариантов:**')

for i in range(0,outer_df.shape[0]):
	with st.container():
		st.subheader(outer_df.name_short.tolist()[i])
		st.markdown(outer_df.name.tolist()[i])
		st.markdown(outer_df.region_name.tolist()[i])
		
st.divider()	
agree = st.checkbox('Показать таблицу признаков для выбранного кластера')

if agree:
    st.dataframe(outer_df)
   
   
image = st.checkbox('Показать матрицу расстояний')

if image:
	st.markdown('Выбранный ВУЗ относится к класетру **"' + option_cluster +'"** и выделен на матрице **белым квадратом**:')
	plasma()
	plt.figure(figsize=(8, 6))
	pcolor(som.distance_map().T) 
	colorbar()
	im = 0
	markers = ['s']
	for i, x in enumerate(train):
		w = som.winner(x)
		if str(ans.cluster.tolist()[i]) == option_cluster:
			plot(w[0] + 0.5, 
			w[1] + 0.5,
			markers[0], 
			markersize = 25,
			markerfacecolor = 'white',
			markeredgecolor = 'white',
			markeredgewidth = 6)
			plt.text(w[0]+0.2,  w[1]+0.3,  str(ans.cluster.tolist()[i]), color='black', fontdict={ 'weight': 'bold', 'size': 10})
		else:
			plt.text(w[0]+0.2,  w[1]+0.3,  str(ans.cluster.tolist()[i]), color='white', fontdict={ 'size': 10})
			 	
		im = im + 1
	st.pyplot(plt)
	
	st.header('Просмотр значений по номеру кластера')
	st.caption('_Отображенная выше матрица и информация о выбранном вузе не будет изменена_')
	option_choose_cluster = st.selectbox(
    'Введите номер кластера',
    (ans.cluster.unique()))
    		
		
	option_cluster = option_choose_cluster
	outer_df = ans[ans.cluster == option_choose_cluster]
	outer_df = outer_df.drop(outer_df[outer_df.name_short == option_univer].index)
		
	for i in range(0,outer_df.shape[0]):
		with st.container():
			st.subheader(outer_df.name_short.tolist()[i])
			st.markdown(outer_df.name.tolist()[i])
			st.markdown(outer_df.region_name.tolist()[i])
				

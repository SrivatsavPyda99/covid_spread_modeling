import matplotlib.pyplot as plt
import csv
import math
from validate_model import create_test_list
import statsmodels.api as sm
import pandas as pd

def graph_model(m, data_path, v_set):
	
	score_dict = dict()
	with open(data_path, 'r') as f:
		reader = csv.reader(f)
		for line in reader:
			
			score = float(line[0])
			if not math.isnan(score):
				k = []
				for vert in line[1:3]:
					k.append(vert)

				z_set = []
				if len(line) > 3:
					for vert in line[3:]:
						z_set.append(vert)
				z_set = sorted(z_set)
				for z in z_set:
					k.append(z)
				k = tuple(k)
				score_dict[k] = score
	con, uncon = create_test_list(m, v_set)

	con_set = []
	for item in con:
		
		k1 = [item[0], item[1]]
		k2 = [item[1], item[0]]
		
		z_set = []
		for z in item[2]:
			z_set.append(z)

		z_set = sorted(z_set)

		for z in z_set:
			k1.append(z)
			k2.append(z)

		k1 = tuple(k1)
		k2 = tuple(k2)

		if k1 in score_dict:
			con_set.append(score_dict[k1])

			
		elif k2 in score_dict:
			con_set.append(score_dict[k2])
			
		else:
			pass



	uncon_set = []

	for item in uncon:
		k1 = [item[0], item[1]]
		k2 = [item[1], item[0]]
		
		z_set = []
		for z in item[2]:
			z_set.append(z)

		z_set = sorted(z_set)

		for z in z_set:
			k1.append(z)
			k2.append(z)

		k1 = tuple(k1)
		k2 = tuple(k2)

		if k1 in score_dict:
			uncon_set.append(score_dict[k1])

			
			
		elif k2 in score_dict:
			
			uncon_set.append(score_dict[k2])

		else:
			pass
	
	
	total_set = []
	for item in con_set:
		total_set.append(item)
	for item in uncon_set:
		total_set.append(item)
	fig, axs = plt.subplots(1, 3, sharey=True, figsize= (10, 4))
	
	#axs[0].hist(total_set,bins= [1 - (20-x)/20 for x in range(0, 21)])
	axs[0].hist(con_set,bins= [1 - (20-x)/20 for x in range(0, 21)])
	axs[0].set(ylabel = 'Number of Pairs')
	axs[1].hist(uncon_set,bins= [1 - (20-x)/20 for x in range(0, 21)])
	axs[1].set(xlabel = 'P-Value')
	axs[2].hist(total_set,bins= [1 - (20-x)/20 for x in range(0, 21)])
	axs[0].set_title('D-Connected Pairs')
	axs[1].set_title('D-Separated Pairs')
	axs[2].set_title('All Pairs')
	plt.savefig('P_val.png')


def make_residuals_bd_graph(path):
	df = pd.read_csv(path)
	y = df['percent change in total Cases']
	m = df['percent change in average movement']
	X = df[[
            'White_Percentage', 
            'Black_Percentage', 
            'Native_Percentage', 
            'Asian_Percentage', 
            'Hawaiian_Percentage', 
            'Other_Percentage', 
            'Republican_Percentage', 
            'Democrat_Percentage']]

	X = sm.add_constant(X)

	model = sm.OLS(y, X).fit()
	preds = model.predict(X)
	res = y - preds

	sc = pd.concat([m, res], axis= 1)
	sc.rename(columns = {'percent change in average movement':'percent change in average movement', 0:'percent change in total cases'}, inplace=True)
	print(sc)
	ax = sc.plot.scatter(x='percent change in average movement', y ='percent change in total cases', c='DarkBlue')


	plt.savefig('resid.png')
	plt.show()

if __name__ == '__main__':




	make_residuals_bd_graph('Data_Files/average_movement_Cases_data_7_day.csv')
	
	'''
	test_m_1 = {

		'Cases':{'in':set(( 'Political', 'Movement', 'Race')), 'out':set(('Increasce',))}, 
		'Movement':{'in':set(('Political', 'Age')), 'out':set(('Cases',))}, 
		'Wealth':{'in':set(('Race', 'Sex')), 'out':set()}, 
		'Race':{'in':set(), 'out':set(('Age', 'Wealth', 'Cases', 'Increasce'))},
		'Sex':{'in':set(('Density',)), 'out':set(('Wealth', ))}, 
		'Age':{'in':set(('Race',  'Density')), 'out':set(( 'Movement',))}, 
		'Political':{'in':set(), 'out':set(('Cases', 'Movement' ))}, 
		'Density':{'in':set(), 'out':set(( 'Age', 'Sex'))}, 
		'Increasce':{'in': set(('Cases', 'Race')), 'out':set()}

	}
	graph_model(test_m_1, 'independence.csv', set(('Cases', 'Movement', 'Wealth', 'Race', 'Sex', 'Age', 'Political', 'Density', 'Increasce')))
	'''

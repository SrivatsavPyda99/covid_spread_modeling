import csv
from copy import deepcopy
import datetime


data_dict = dict()
with open('csv_data/racial_data.csv', 'r') as f:
	reader = csv.reader(f)
	titles = next(reader)
	next(reader)
	for line in reader:
		new_dict = dict()

		place = (line[1].split(',')[0] +':'+ line[1].split(',')[1][1:]).lower()
		
		data_dict[place] = new_dict
		data_dict[place]['White_Percentage'] = float(line[4])/float(line[2])
		data_dict[place]['Black_Percentage'] = float(line[6])/float(line[2])
		data_dict[place]['Native_Percentage'] = float(line[8])/float(line[2])
		data_dict[place]['Asian_Percentage'] = float(line[10])/float(line[2])
		data_dict[place]['Hawaiian_Percentage'] = float(line[12])/float(line[2])
		data_dict[place]['Other_Percentage'] = float(line[14])/float(line[2])
		data_dict[place]['Multi_Percentage'] = (float(line[16]) + float(line[18]) +  float(line[20]))/float(line[2])
		data_dict[place]['Total_Population'] = int(line[2])


with open('csv_data/political_data.csv', 'r') as f:
	reader = csv.reader(f, delimiter=';')
	titles = next(reader)
	for line in reader:

		
		place = (line[3].split(',')[0] +':'+ line[3].split(',')[1][1:]).lower()
		
		try:
			data_dict[place]['Republican_Percentage'] = float(line[10])/100
			data_dict[place]['Democrat_Percentage'] = float(line[11])/100
			data_dict[place]['Independent_Percentage'] = 1 - data_dict[place]['Republican_Percentage'] - data_dict[place]['Democrat_Percentage']
			data_dict[place]['Income'] = float(line[23])
		except:

			try:
				del(data_dict[place])
			except:
				pass
			

with open('csv_data/area_data.csv', 'r') as f:
	reader = csv.reader(f)
	titles = next(reader)
	next(reader)

	current_state = None
	for line in reader:
		if len(line[0].split(',')) == 1:

			current_state = 'county:{}'.format(line[0].lower())

		else:
			current_state = 'county:{}'.format(current_state.split(':')[1])
			try:
				place = '{} {}'.format(line[0].split(',')[0].lower(), current_state)
				data_dict[place]['Land_Area'] = float(line[23])
			except:
				try:
					current_state = 'city:{}'.format(current_state.split(':')[1])
					place = '{} {}'.format(line[0].split(',')[0].lower(), current_state)
					data_dict[place]['Land_Area'] = float(line[23])
				except:
					try:
						current_state = 'parish:{}'.format(current_state.split(':')[1])
						place = '{} {}'.format(line[0].split(',')[0].lower(), current_state)
						data_dict[place]['Land_Area'] = float(line[23])
					except:
						try:
							current_state = 'municipality:{}'.format(current_state.split(':')[1])
							place = '{} {}'.format(line[0].split(',')[0].lower(), current_state)
							data_dict[place]['Land_Area'] = float(line[23])
						except:
							try:
								current_state = 'census area:{}'.format(current_state.split(':')[1])
								place = '{} {}'.format(line[0].split(',')[0].lower(), current_state)
								data_dict[place]['Land_Area'] = float(line[23])
							except:
								try:
									current_state = 'borough:{}'.format(current_state.split(':')[1])
									place = '{} {}'.format(line[0].split(',')[0].lower(), current_state)
									data_dict[place]['Land_Area'] = float(line[23])
								except:
									pass
									


with open('csv_data/age_sex_data.csv', 'r') as f:
	reader = csv.reader(f)
	titles = next(reader)
	z = next(reader)
	for line in reader:
		
		try:
			place = (line[1].split(',')[0] +':'+ line[1].split(',')[1][1:]).lower()
			
			data_dict[place]['Male_Percentage'] = float(line[6])/float(line[2])
			data_dict[place]['Female_Percentage'] = float(line[10])/float(line[2])
			data_dict[place]['Under_25_Percentage'] = (float(line[16]) + float(line[28]) +  float(line[40]) +  float(line[52]) + float(line[64]))/100
			data_dict[place]['25_49_Percentage'] = (float(line[76]) + float(line[88]) +  float(line[100]) +  float(line[112]) + float(line[124]))/100
			data_dict[place]['50_74_Percentage'] = (float(line[136]) + float(line[148]) +  float(line[160]) +  float(line[172]) + float(line[184]))/100
			data_dict[place]['Over_74_Percentage'] = (float(line[196]) + float(line[208]) +  float(line[220]) )/100
			
				
		except:

			try:
				del(data_dict[place])
			except:
				pass
			

with open('csv_data/corona_data.csv', 'r') as f:
	reader = csv.reader(f)
	titles = next(reader)
	old_date = None
	date_dict = dict()
	for key in data_dict:
		date_dict[key] = [0, 0]
	full_corona_dict = dict()
	for line in reader:
		date = line[0]
		if date != old_date:
			if old_date is not None:
				full_corona_dict[old_date] = deepcopy(date_dict)
			old_date = date

		county = line[1].lower()
		state = line[2].lower()


		if '{} county:{}'.format(county, state) in date_dict:
			date_dict['{} county:{}'.format(county, state)] = [int(line[4]), int(line[5])]
		
		elif '{} city:{}'.format(county, state) in date_dict:
			date_dict['{} city:{}'.format(county, state)] = [int(line[4]), int(line[5])]
		
		elif '{} parish:{}'.format(county, state) in date_dict:
			date_dict['{} parish:{}'.format(county, state)] = [int(line[4]), int(line[5])]
		
		elif '{} municipality:{}'.format(county, state) in date_dict:
			date_dict['{} municipality:{}'.format(county, state)] = [int(line[4]), int(line[5])]
		
		elif '{} borough:{}'.format(county, state) in date_dict:
			date_dict['{} borough:{}'.format(county, state)] = [int(line[4]), int(line[5])]
		
		elif '{} census area:{}'.format(county, state) in date_dict:
			date_dict['{} census area:{}'.format(county, state)] = [int(line[4]), int(line[5])]




with open('csv_data/movement_data.csv', 'r') as f:
	reader = csv.reader(f)
	titles = next(reader)
	end_set = ('parish', 'borough', 'city', 'county')
	movement_dict = dict()
	for line in reader:
		if line[1] == 'United States' and line[3] != '':
			date = line[4]
			if date not in movement_dict:
				movement_dict[date] = dict()
			
			state = line[2].lower()
			county = line[3].lower()

			place = None
			if county.split(' ')[-1] in end_set:
				county = county[:-len(county.split(' ')[-1])]

			
			if '{}county:{}'.format(county, state) in date_dict:
				place = '{}county:{}'.format(county, state)
		
			elif '{}city:{}'.format(county, state) in date_dict:
				place = '{} city:{}'.format(county, state)
			
			elif '{}parish:{}'.format(county, state) in date_dict:
				place = '{} parish:{}'.format(county, state)
			
			elif '{}municipality:{}'.format(county, state) in date_dict:
				place = '{} municipality:{}'.format(county, state)
			
			elif '{}borough:{}'.format(county, state) in date_dict:
				place = '{} borough:{}'.format(county, state)

			elif '{}census area:{}'.format(county, state) in date_dict:
				place = '{} census area:{}'.format(county, state)


			if place in date_dict:
				movement = []


				

				for x in range(5, 11):
					try:
						movement.append(int(line[x]))
					except:
						movement.append(None)
				movement_dict[date][place] = movement

movement_lag = 7
movement_dict_2 = dict()

for date in movement_dict:
	temp = date.split('-')
	year = temp[0]
	month = temp[1]
	day = temp[2]

	real_date = datetime.datetime(int(year), int(month), int(day))
	past_date = real_date - datetime.timedelta(days=movement_lag)

	past_year = past_date.year
	past_month = past_date.month
	if past_month < 10:
		past_month = '0{}'.format(past_month)
	past_day = past_date.day
	if past_day < 10:
		past_day = '0{}'.format(past_day)

	past_date_key = '{}-{}-{}'.format(past_year, past_month, past_day)

	
	movement_dict_2[date] = dict()
	if past_date_key in movement_dict:
		for place in movement_dict[past_date_key]:
			has_data = True
			avg_movement = [0, 0, 0, 0, 0, 0]
			for x in range(0, movement_lag):
				temp_date = past_date + datetime.timedelta(days=x)

				temp_year = past_date.year
				temp_month = past_date.month
				if temp_month < 10:
					temp_month = '0{}'.format(temp_month)
				temp_day = temp_date.day
				if temp_day < 10:
					temp_day = '0{}'.format(temp_day)

				temp_date_key = '{}-{}-{}'.format(temp_year, temp_month, temp_day)

				if temp_date_key in movement_dict and place in movement_dict[temp_date_key]:

					for i in range(0, len(movement_dict[temp_date_key][place])):
						try:
							avg_movement[i] += movement_dict[temp_date_key][place][i]
						except:
							avg_movement[i] = None
				else:
					has_data = False
			for i in range(0, 6):
				try:
					avg_movement[i] = avg_movement[i]/7
				except:
					avg_movement[i] = None
			if has_data:
				movement_dict_2[date][place] = avg_movement

movement_dict = movement_dict_2
index_dict = {0: 'average', 1:'retail and recreation', 2:'grocery and pharmacy', 3:'transit stattions', 4:'workplaces', 5:'residential'}

def make_csv(out_path, time_frame, index=0, data_type='Cases'):
	z = 0

	y = 0
	
	with open(out_path, 'w') as f:
		writer = csv.writer(f)
		if index != 6:
			writer.writerow(['percent change in {} movement'.format(index_dict[index]), 'percent change in total {}'.format(data_type), 'Income', 'White_Percentage', 
				'Black_Percentage', 'Native_Percentage', 'Asian_Percentage', 'Hawaiian_Percentage', 
				'Other_Percentage', 'Multi_Percentage', 'Republican_Percentage', 
				'Democrat_Percentage', 'Independent_Percentage', 'Population_Density', 
				'Male_Percentage', 'Female_Percentage', 'Under_25_Percentage', 
				'25_49_Percentage', '50_74_Percentage', 'Over_74_Percentage',
				'Number_of_Cases', 'Percent_w_Corona', 'Place'])
		else:
			writer.writerow(['percent change in retail and recreation movement', 'percent change in grocery and pharmacy movement', 
				'percent change in transit stattions movement', 'percent change in workplaces movement', 'percent change in residential movement',
				'percent change in total {}'.format(data_type), 'Income', 'White_Percentage', 
				'Black_Percentage', 'Native_Percentage', 'Asian_Percentage', 'Hawaiian_Percentage', 
				'Other_Percentage', 'Multi_Percentage', 'Republican_Percentage', 
				'Democrat_Percentage', 'Independent_Percentage', 'Population_Density', 
				'Male_Percentage', 'Female_Percentage', 'Under_25_Percentage', 
				'25_49_Percentage', '50_74_Percentage', 'Over_74_Percentage',
				'Number_of_Cases', 'Percent_w_Corona', 'Place'])
		for date in movement_dict:

			temp = date.split('-')
			year = temp[0]
			month = temp[1]
			day = temp[2]
			real_date = datetime.datetime(int(year), int(month), int(day))
			next_date = real_date + datetime.timedelta(days=time_frame)


			next_year = next_date.year
			next_month = next_date.month
			if next_month < 10:
				next_month = '0{}'.format(next_month)
			next_day = next_date.day
			if next_day < 10:
				next_day = '0{}'.format(next_day)

			next_date_key = '{}-{}-{}'.format(next_year, next_month, next_day)
			
			y = 0
			for place in movement_dict[date]:
				y += 1

		
				movement_decreasce = 0
				obs = 0
				found_obs = False
				if index == 0:
					for ob in movement_dict[date][place]:
						if ob is not None:
							obs += 1
							movement_decreasce += ob
							found_obs = True
					if found_obs:
						movement_decreasce = movement_decreasce/obs
				elif index != 6:
					movement_decreasce = movement_dict[date][place][index]
					if movement_decreasce is not None:
						found_obs = True

				else:
					movement_decreasce = []
					for j in range(1, 6):
						movement_decreasce.append(movement_dict[date][place][j])
					if None not in movement_decreasce:
						found_obs = True


				percent_increasce_cases = 0
				percent_increasce_death = 0
				current_date_in_data = False

				past_date = real_date - datetime.timedelta(days=14)
				past_year = past_date.year
				past_month = past_date.month
				if past_month < 10:
					past_month = '0{}'.format(past_month)
				past_day = past_date.day
				if past_day < 10:
					past_day = '0{}'.format(past_day)

				past_date = '{}-{}-{}'.format(past_year, past_month, past_day)


				next_past_date = next_date - datetime.timedelta(days=14)
				next_past_year = next_past_date.year
				next_past_month = next_past_date.month
				if next_past_month < 10:
					next_past_month = '0{}'.format(next_past_month)
				next_past_day = next_past_date.day
				if next_past_day < 10:
					next_past_day = '0{}'.format(next_past_day)

				next_past_date = '{}-{}-{}'.format(next_past_year, next_past_month, next_past_day)
				
				current_date_in_data = False

				if date in full_corona_dict and past_date in full_corona_dict:
					if data_type == 'Cases':
						current_count = full_corona_dict[date][place][0] - full_corona_dict[past_date][place][0]

					if data_type == 'Deaths':
						current_count = full_corona_dict[date][place][1] 

					current_date_in_data = True

				elif date in full_corona_dict:
					if data_type == 'Cases':
						current_count = full_corona_dict[date][place][0] 

					if data_type == 'Deaths':
						current_count = full_corona_dict[date][place][1] 

					current_date_in_data = True

				if date in full_corona_dict and past_date in full_corona_dict:
					
					active_count = full_corona_dict[date][place][0] - full_corona_dict[past_date][place][0]


				elif date in full_corona_dict:
					active_count = full_corona_dict[date][place][0]
					

				next_date_in_data = False

				if next_date_key in full_corona_dict and next_past_date in full_corona_dict:
					if data_type == 'Cases':
						next_count = full_corona_dict[next_date_key][place][0] - full_corona_dict[next_past_date][place][0]

					if data_type == 'Deaths':
						next_count = full_corona_dict[next_date_key][place][1] 

					next_date_in_data = True

				elif next_date_key in full_corona_dict:
					if data_type == 'Cases':
						next_count = full_corona_dict[next_date_key][place][0] 

					if data_type == 'Deaths':
						next_count = full_corona_dict[next_date_key][place][1] 

					next_date_in_data = True
				
				

				have_increasce_data = False
				
				if current_date_in_data and current_count != 0 and next_date_in_data:
					percent_increasce = ((next_count - current_count)/current_count)*100
					have_increasce_data = True

				if have_increasce_data and found_obs and index != 6:
					place_vector = ([movement_decreasce, percent_increasce, data_dict[place]['Income'], data_dict[place]['White_Percentage'],data_dict[place]['Black_Percentage'], 
						data_dict[place]['Native_Percentage'], data_dict[place]['Asian_Percentage'], data_dict[place]['Hawaiian_Percentage'], 
						data_dict[place]['Other_Percentage'], data_dict[place]['Multi_Percentage'], data_dict[place]['Republican_Percentage'], 
						data_dict[place]['Democrat_Percentage'], data_dict[place]['Independent_Percentage'], data_dict[place]['Total_Population']/data_dict[place]['Land_Area'], 
						data_dict[place]['Male_Percentage'], data_dict[place]['Female_Percentage'], data_dict[place]['Under_25_Percentage'], 
						data_dict[place]['25_49_Percentage'], data_dict[place]['50_74_Percentage'], data_dict[place]['Over_74_Percentage'],
						active_count, active_count/data_dict[place]['Total_Population'], place])
					z += 1
			
					writer.writerow(place_vector)
				elif have_increasce_data and found_obs and index == 6:
					place_vector = ([movement_decreasce[0], movement_decreasce[1], movement_decreasce[2], movement_decreasce[3], movement_decreasce[4],
					    percent_increasce, data_dict[place]['Income'], data_dict[place]['White_Percentage'],data_dict[place]['Black_Percentage'], 
						data_dict[place]['Native_Percentage'], data_dict[place]['Asian_Percentage'], data_dict[place]['Hawaiian_Percentage'], 
						data_dict[place]['Other_Percentage'], data_dict[place]['Multi_Percentage'], data_dict[place]['Republican_Percentage'], 
						data_dict[place]['Democrat_Percentage'], data_dict[place]['Independent_Percentage'], data_dict[place]['Total_Population']/data_dict[place]['Land_Area'], 
						data_dict[place]['Male_Percentage'], data_dict[place]['Female_Percentage'], data_dict[place]['Under_25_Percentage'], 
						data_dict[place]['25_49_Percentage'], data_dict[place]['50_74_Percentage'], data_dict[place]['Over_74_Percentage'],
						active_count, active_count/data_dict[place]['Total_Population'], place])
					z += 1
			
					writer.writerow(place_vector)

for ind in range(0, 6):
	for time in [7, 14, 21, 28]:
		for dat_type in ['Cases', 'Deaths']:			
			make_csv('Data_Files/{}_movement_{}_data_{}_day.csv'.format(index_dict[ind], dat_type, time), time, index = ind, data_type = dat_type)
			print(ind, time, dat_type)

for time in [7, 14, 21, 28]:
	for dat_type in ['Cases', 'Deaths']:			
		make_csv('Data_Files/{}_movement_{}_data_{}_day.csv'.format('All', dat_type, time), time, index = 6, data_type = dat_type)
		print(6, time, dat_type)

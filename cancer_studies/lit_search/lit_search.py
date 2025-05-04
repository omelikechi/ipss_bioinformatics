# Search Europe PMC for features related to cancer

import csv
import os
import requests
import time

import numpy as np
import pickle

do_lit_search = False
save_results = False

replace_feature = {'C20orf112':'NOL4L', 'C10orf4':'FRA10AC1', 'C18orf10':'TPSG2', 'C14orf109':'LYSET', 'MOSC1':'MARC1',
	'FLJ38482':'TMEM192', 'RP11-679B17.1':'TMEM170B'}

method_list = ['vita', 'boruta', 'ssboost', 'deeppink', 'kol', 'korf', 'koglm', 'ipssl', 'ipssrf', 'ipssgb']
cancer_type = 'ovarian'
feature_type = 'mirna'
response = 'status'

fdr_max = 0.9
year_cutoff = 2020

missing_methods = []
feature_list = []
q_value_dict = {}

for method in method_list:

	filepath = f'./cancer_results/{method}/{method}_{cancer_type}_{feature_type}_{response}.pkl'
	if not os.path.exists(filepath):
		missing_methods.append(f'{method} results do not exist.')
		continue
	with open(filepath, "rb") as f:
		results = pickle.load(f)

	print(f'\n-----{method}-----')
	q_values = results['q_values']
	feature_names = results['metadata']['feature_names']
	for feature, q_value in q_values.items():
		name = feature_names[feature]
		if q_value <= fdr_max:
			print(f'{name}: {q_value:.2f}')
			feature_list.append(name)
			if name not in q_value_dict:
				q_value_dict[name] = {}
			q_value_dict[name][method] = q_value

feature_list = list(set(feature_list))
total_features = len(feature_list)
print(f'\nTotal number of features: {total_features}')

print()
print(f'Missing methods')
print(f'--------------------------------')
for string in missing_methods:
	print(string)

if do_lit_search:

	results = []

	def query_europe_pmc(feature, delay=1):
		if cancer_type == 'glioma':
			query = f'"{feature}" AND "{cancer_type}" AND "prognosis"' 
		else: 
			query = f'"{feature}" AND "{cancer_type} cancer" AND "prognosis"'
		base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
		cursor = "*"
		page_size = 1000

		total_articles = 0
		total_citations = 0
		recent_articles = 0
		recent_citations = 0

		while True:
			url = f"{base_url}?query={query}&format=json&pageSize={page_size}&cursorMark={cursor}"
			try:
				response = requests.get(url)
				response.raise_for_status()
				data = response.json()

				hits = data.get("resultList", {}).get("result", [])
				for hit in hits:
					citations = int(hit.get("citedByCount", 0))
					total_articles += 1
					total_citations += citations

					year_str = hit.get("pubYear", "")
					if year_str.isdigit() and int(year_str) >= year_cutoff:
						recent_articles += 1
						recent_citations += citations

				next_cursor = data.get("nextCursorMark")
				if not next_cursor or next_cursor == cursor:
					break

				cursor = next_cursor
				time.sleep(delay)

			except Exception as e:
				print(f"{feature}: error - {e}")
				results.append({
					"feature": feature, "articles": "error", "citations": "error",
					"recent_articles": "error", "recent_citations": "error"
				})
				return

		print(f"{feature}: {total_articles} articles, {total_citations} citations")

		row = {
			"feature": feature,
			"articles": total_articles,
			"citations": total_citations,
			"recent_articles": recent_articles,
			"recent_citations": recent_citations
		}

		# Add q-values per method
		for method in method_list:
			row[method] = q_value_dict.get(feature, {}).get(method, "")

		results.append(row)

	print()
	print(f'Articles and citations')
	print(f'--------------------------------')
	for feature in feature_list:
		if feature in replace_feature.keys():
			feature = replace_feature[feature]
		query_europe_pmc(feature)

	if save_results:
		fieldnames = ["feature", "articles", "citations", "recent_articles", "recent_citations"] + method_list
		with open(f"{cancer_type}_{feature_type}_{response}.csv", "w", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(results)






# # Search Europe PMC for features related to cancer

# import csv
# import os
# import requests
# import time

# import numpy as np
# import pickle

# do_lit_search = True
# save_results = True

# method_list = ['vita', 'boruta', 'ssboost', 'deeppink', 'kol', 'korf', 'koglm', 'ipssl', 'ipssrf', 'ipssgb']
# cancer_type = 'ovarian'
# feature_type = 'rnaseq'
# response = 'status'

# fdr_max = 0.5

# missing_methods = []
# feature_list = []
# q_value_dict = {}

# for method in method_list:

# 	filepath = f'./cancer_results/{method}/{method}_{cancer_type}_{feature_type}_{response}.pkl'
# 	if not os.path.exists(filepath):
# 		missing_methods.append(f'{method} results do not exist.')
# 		continue
# 	with open(filepath, "rb") as f:
# 		results = pickle.load(f)

# 	print(f'\n-----{method}-----')
# 	q_values = results['q_values']
# 	feature_names = results['metadata']['feature_names']
# 	for feature, q_value in q_values.items():
# 		name = feature_names[feature]
# 		if q_value <= fdr_max:
# 			print(f'{name}: {q_value:.2f}')
# 			feature_list.append(name)
# 			if name not in q_value_dict:
# 				q_value_dict[name] = {}
# 			q_value_dict[name][method] = q_value

# feature_list = list(set(feature_list))
# total_features = len(feature_list)
# print(f'\nTotal number of features: {total_features}')

# print()
# print(f'Missing methods')
# print(f'--------------------------------')
# for string in missing_methods:
# 	print(string)

# if do_lit_search:

# 	results = []

# 	def query_europe_pmc(feature, delay=1):
# 		query = f'"{feature}" AND "{cancer_type} cancer"'
# 		base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# 		cursor = "*"
# 		page_size = 1000

# 		total_articles = 0
# 		total_citations = 0

# 		while True:
# 			url = f"{base_url}?query={query}&format=json&pageSize={page_size}&cursorMark={cursor}"
# 			try:
# 				response = requests.get(url)
# 				response.raise_for_status()
# 				data = response.json()

# 				hits = data.get("resultList", {}).get("result", [])
# 				total_articles += len(hits)
# 				total_citations += sum(int(hit.get("citedByCount", 0)) for hit in hits)

# 				next_cursor = data.get("nextCursorMark")
# 				if not next_cursor or next_cursor == cursor:
# 					break

# 				cursor = next_cursor
# 				time.sleep(delay)

# 			except Exception as e:
# 				print(f"{feature}: error - {e}")
# 				results.append({"feature": feature, "articles": "error", "citations": "error"})
# 				return

# 		print(f"{feature}: {total_articles} articles, {total_citations} citations")

# 		row = {
# 			"feature": feature,
# 			"articles": total_articles,
# 			"citations": total_citations
# 		}

# 		# Add q-values per method
# 		for method in method_list:
# 			row[method] = q_value_dict.get(feature, {}).get(method, "")

# 		results.append(row)

# 	print()
# 	print(f'Articles and citations')
# 	print(f'--------------------------------')
# 	for feature in feature_list:
# 		query_europe_pmc(feature)

# 	if save_results:
# 		fieldnames = ["feature", "articles", "citations"] + method_list
# 		with open(f"{cancer_type}_{feature_type}_{response}.csv", "w", newline="") as f:
# 			writer = csv.DictWriter(f, fieldnames=fieldnames)
# 			writer.writeheader()
# 			writer.writerows(results)













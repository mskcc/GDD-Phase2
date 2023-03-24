import pandas as pd
import sys, json, re

def write_output_json(row): #single output
	feature_annotation = pd.read_csv('resources/feature_annotations.csv', index_col = 0) 
	pred = {'meta-data': dict(), 'results': list(),'feature_list': list()}
	pred['meta-data']['sample_id'] = row['SAMPLE_ID'].values[0]
	# hardcoded rn
	pred['meta-data']['version'] = '2.0.0'
	pred['results'].append({'tumor_type': re.sub('\.',' ',row['Pred1'].values[0]), 'confidence': str(row['Conf1'].values[0])})
	pred['results'].append({'tumor_type': re.sub('\.',' ',row['Pred2'].values[0]), 'confidence': str(row['Conf2'].values[0])})
	pred['results'].append({'tumor_type': re.sub('\.',' ',row['Pred3'].values[0]), 'confidence': str(row['Conf3'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp1'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var2'].values[0], 'variable_imp': str(row['Imp2'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var3'].values[0], 'variable_imp': str(row['Imp3'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var4'].values[0], 'variable_imp': str(row['Imp4'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var5'].values[0], 'variable_imp': str(row['Imp5'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var6'].values[0], 'variable_imp': str(row['Imp6'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var7'].values[0], 'variable_imp': str(row['Imp7'].values[0])})    
	pred['feature_list'].append({'variable_name': row['Var8'].values[0], 'variable_imp': str(row['Imp8'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var9'].values[0], 'variable_imp': str(row['Imp9'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var10'].values[0], 'variable_imp': str(row['Imp10'].values[0])})
	return pred

if __name__ == "__main__":
	try:
		gdd_output_filepath = sys.argv[1]
	except Exception as e:
		print('specify gdd output file path')
	try:
		output_json_filepath = sys.argv[2]
	except Exception as e:
		print('specify output json filepath')

	gdd_output = pd.read_csv(gdd_output_filepath)
	output_json = write_output_json(gdd_output)
	with open(output_json_filepath, "w") as outfile:
		json.dump(output_json, outfile)



	
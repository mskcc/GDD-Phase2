import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib
from pybedtools import BedTool, helpers
from pyfaidx import Fasta

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
import shap
import re
import json

sys.path.insert(0, '/home/darmofam/morris/classifier/final/')
from train_gdd_nn import MLP, MyDataset, create_loader, create_unshuffled_loader
from gdd_ensemble import EnsembleClassifier


def process_data_single(single_data, colnames): 
	### process_data_all will create train, test, validation folds for all classes with min_samples number of samples
	values = [single_data[col].values[0] if col in single_data.columns else 0 for col in colnames]
	values = pd.DataFrame(values).T
	values.columns = colnames
	values.to_csv('prediction_input.csv')
	return np.array(values)

def pred_results(logits_list, label):
	#similar to softmax_predictive_accuracy function
	probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
	fold_preds = [torch.max(probs_list[i], 1)[1].cpu().data.numpy() for i in range(len(probs_list))]
	probs_tensor = torch.stack(probs_list, dim = 2)
	probs = torch.mean(probs_tensor, dim=2)
	allprobs = probs.cpu().data.numpy()
	top_preds = np.argpartition(allprobs[0], -3)[-3:]
	top_probs = allprobs[0][top_preds]
	pred_probs, pred_class = torch.max(probs, 1)
	probs = probs.cpu().data.numpy()
	pred_probs = pred_probs.cpu().data.numpy()
	preds = pred_class.cpu().data.numpy()
	return top_preds[::-1], top_probs[::-1], probs


def top_shap_single(train_x, test_x, test_y, n_types, colnames, save_link):
	#validate shapley values by calculating across all types
	precalc_shap_fn = '/home/darmofam/morris/classifier/final/implementation/gddnn_kmeans_output.bz2'
	#example_shap = shap.kmeans(np.array(train_x), n_types)
	#joblib.dump(example_shap, filename=example_filename, compress=('bz2', 9))
	precalc_shap = joblib.load(filename=precalc_shap_fn)
	pred_ind = test_y
	def spec_prob_function(x):
			#use this to grab probability from ensemble classifier (SHAP package formatting)
		x = torch.from_numpy(x).float()
		x = x.to(device)
		logits_list = fold_ensemble(x)
		probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
		probs_tensor = torch.stack(probs_list, dim = 2)
		probs = torch.mean(probs_tensor, dim=2)
		pred_probs= probs[:,pred_ind]
		pred_probs = pred_probs.cpu().data.numpy().reshape(len(x),)
		return pred_probs
	explainer = shap.KernelExplainer(spec_prob_function, precalc_shap)
	shap_values = explainer.shap_values(test_x, nsamples=2000)
		
	#format data
	shap_values = pd.DataFrame(shap_values)
	shap_values = shap_values.append(pd.DataFrame(test_x))
	shap_values.columns = colnames
	shap_values.index = ['Shapley_Values', 'Feature_Values']
	shap_values = shap_values[[i for i in shap_values.columns if i in feature_annotation.in_col.values]]
	shap_values = shap_values.T
	shap_values.sort_values(by = 'Shapley_Values', axis = 0, inplace=True, ascending=False)
	shap_cols = []
	for col, fv in zip(shap_values.index, shap_values.Feature_Values):
		new_col = feature_annotation[(feature_annotation.in_col==col)]
		if len(new_col)==2:
			shap_cols.append(new_col[new_col.value==fv].out_col.values[0])
		else:
			shap_cols.append(new_col.out_col.values[0])
	shap_values = shap_values.assign(Shapley_Columns = shap_cols)
	return shap_values[:10]

def parse_dmp_json(data):
	#new format
	features = {}
	try:
		# tumor sample barcode was renamed sample ID
		features['SAMPLE_ID'] = data['meta-data']['dmp_sample_id']
		features['Diagnosed_Cancer_Type'] = data['meta-data']['tumor_type_code']
		features['Diagnosed_Cancer_Type_Detailed'] = data['meta-data']['tumor_type_code']
		features['Sample_Type'] = 'Metastasis' if data['meta-data']['is_metastasis'] else 'Primary'
		features['Primary_Site'] = data['meta-data']['primary_site']
		features['Metastatic_Site'] = data['meta-data']['metastasis_site']
		features['Cancer_Type'] = data['meta-data']['tumor_type_code']
		# dummy classification category feature as 'test'
		features['Classification_Category'] = 'test'
		# When DMP uses 0/1 for gender, they mean 0=Female and 1=Male. Swap this around for GDD.
		if data['meta-data']['gender'] == 0 or data['meta-data']['gender'] == 1:
			features['Gender_F'] = 1 - data['meta-data']['gender']
		elif data['meta-data']['gender'] == "Male":
			features['Gender_F'] = 0
		else:
			features['Gender_F'] = 1
		features['MSI_SCORE'] = float(data['msi']['msi_score']) if data['msi']['msi_score'] != '' else 0
	except Exception as e:
		print("ERROR loading metadata")
		raise Exception("ERROR loading metadata")

	try:
		# reading in hotspot list for gene hotspot features
		hotspot = pd.read_csv("/data/morrisq/darmofam/final_hotspot_list.csv", index_col = 0) #TODO - change
		# for calculating SNV and indel burden
		impact_gene_panel = {}
		impact_gene_panel['IM3'] = 896665
		impact_gene_panel['IM5'] = 1016478
		impact_gene_panel['IM6'] = 1139322
		impact_gene_panel['IM7'] = 1213770
		norm = 10**6
		# Parse out somatic substitutions and small indels
		features['Mutation_Count'] = 0
		features['LogSNV_Mb'] = 0
		features['LogINDEL_Mb'] = 0
		for mut in data['snp-indels']: #TODO - make sure all of these name bugs are fixed
			# Skip this entry if it's the duplicates that DMP lists for CDKN2A's p14ARF and p16INK4A
			if re.search('p14ARF|p16INK4A', mut['gene_id']):
				continue
			# Mark this gene as mutated, and also create an entry for the specific mutation
			features[mut['gene_id']] = 1
			mut['aa_change'] = re.sub('^p\.', '', mut['aa_change'])
			# hotspot annotation and check if hotspot feature already exists
			# IMPORTANT -- HOTSPOT LIST TAKES AA CHANGES IN RAW CVR FORM (NOT THE VERSION AFTER RE.SUB '*' --> '.' CHANGES)
			if ('p.'.join([mut['gene_id'], mut['aa_change']]) in hotspot['Hotspot_Label'].tolist()) and ('_'.join([mut['gene_id'], 'hotspot']) not in features.keys()):
				features['_'.join([mut['gene_id'], 'hotspot'])] = 1
				mut['aa_change'] = re.sub('\*', '.', mut['aa_change'])
				features['.'.join([mut['gene_id'], mut['aa_change']])] = 1
			# Handle the GDD bug where it expects 'TP53_X125_s_ice' instead of 'TP53_X125_splice'
			if 'TP53_X125_splice' in features:
				features['TP53_X125_s_ice'] = 1 #TODO: make sure this is still a bug
			# For truncating events, make a new feature with '_TRUNC' appended to the gene name
			if not re.search('^Missense|^nonsynonymous_SNV|^missense|^5\'Flank|^upstream', mut['variant_class']):
				features['_'.join([mut['gene_id'], 'TRUNC'])] = 1
			# For promoter mutations, make a new feature with 'p' appended to the gene name
			if re.search('^(promoter|5\'Flank|upstream)$', mut['variant_class']):
				features[''.join([mut['gene_id'], 'p'])] = 1
			features['Mutation_Count'] += 1
			# accruing counts for SNV
			if len(mut['alt_allele']) == 1 & len(mut['ref_allele']) == 1:
				features['LogSNV_Mb'] += 1

			# accruing counts for INDEL
			else:
				features['LogINDEL_Mb'] += 1
		# count/panel size * 1 mil + 1 (+1 is to be consistent with GDD repo)
		SNV_count = features['LogSNV_Mb']
		gene_panel = features['SAMPLE_ID'][-3:]
		features['LogSNV_Mb'] = np.log10((features['LogSNV_Mb']/(impact_gene_panel[gene_panel]/norm)+1))
		features['LogINDEL_Mb'] = np.log10(features['LogINDEL_Mb']/(impact_gene_panel[gene_panel]/norm)+1)
	except Exception as e:
		print("ERROR loading SNV")
		raise Exception("ERROR loading SNV")

	try:
		# signatures are used when 10 SNVs are present
		if SNV_count >= 10 and 'mut-sig' in data:
			# mut-sig might not be in json, also mut-sig should not be empty
			if data['mut-sig'] != {}:
				# rough reference of signature types -- can be updated in the future
				signature_cat_dict = {
					'Age':[1],'APOBEC':[2,13],'BRCA':[3],
					'Smoking':[4,24],'MMR':[6,15,20,26],
					'UV':[7],'POLE':[10],'TMZ':[11]
				} 
				# sigs_dict = ast.literal_eval(re.sub('\n','',data['mut-sig']))
				# dict of signature values within dict(data)['mut-sig']
				sigs_dict = data['mut-sig']
				for key in signature_cat_dict:
					sig_exp = 0.0
					for signature in signature_cat_dict[key]:
						sig_exp += float(sigs_dict[str('Signature.'+str(signature))]['mean'])
					if sig_exp > 0.4:
						features['_'.join(['Sig',key])] = 1
	except Exception as e:
		print("ERROR loading Signatures")
		raise Exception("ERROR loading Signatures")

	try:
		# Parse out gene-level somatic copy-number variants
		for cnv in data['cnv-variants']:
			cnv_type = 'Amp' if cnv['cnv_class_name'] == 'COMPLETE_GENE_GAIN' else 'HomDel'
			features['_'.join([cnv['gene_id'], cnv_type])] = 1
		# Parse out structural variants, and pull out the gene pairs at breakpoints
		for sv in data['sv-variants']: #TODO restructure these
			fusion_name = '_'.join(sorted([sv['site1_gene'],sv['site2_gene']]))
			if 'ALK' in fusion_name:
				features['ALK_fusion'] = 1
			if 'BRAF' in fusion_name:
				features['BRAF_fusion'] = 1
			if 'ETV6' in fusion_name:
				features['ETV6_fusion'] = 1
			if 'EWSR1' in fusion_name:
				features['EWSR1_fusion'] = 1
			if 'FGFR2' in fusion_name:
				features['FGFR2_fusion'] = 1
			if 'FGFR3' in fusion_name:
				features['FGFR3_fusion'] = 1
			if 'NTRK1' in fusion_name:
				features['NTRK1_fusion'] = 1
			if 'NTRK2' in fusion_name:
				features['NTRK2_fusion'] = 1
			if 'NTRK3' in fusion_name:
				features['NTRK3_fusion'] = 1
			if 'PAX8' in fusion_name:
				features['PAX8_fusion'] = 1
			if 'RET' in fusion_name:
				features['RET_fusion'] = 1
			if 'ROS1' in fusion_name:
				features['ROS1_fusion'] = 1
			if 'TMPRSS2' in fusion_name:
				features['TMPRSS2_fusion'] = 1
			if fusion_name == 'EGFR_EGFR' and bool(re.search('intragenic',sv['annotation'])):
				features['EGFR_SV'] = 1
			if fusion_name == 'MET_MET' and bool(re.search('intragenic',sv['annotation'])):
				features['MET_SV'] = 1


		# Parse out CN segments to calculate CN burden, and also determine arm-level Amp/Del events
		log_ratio_threshold = 0.2 #TODO make sure this is the same as mine
		cn_length = total_length = 0
		# Expected header for segments is ["chrom", "loc.start", "loc.end", "num.mark", "seg.mean"]
		data['seg-data'] = [seg for seg in data['seg-data'] if seg[0] != "chrom"] # Remove header
		amp_segs = [seg for seg in data['seg-data'] if float(seg[4]) >= log_ratio_threshold]
		del_segs = [seg for seg in data['seg-data'] if float(seg[4]) <= -log_ratio_threshold]
		# Measure CN burden = Sum of lengths of Amp or Del segments / Total length of all segments
		for seg in amp_segs + del_segs:
			cn_length += int(seg[2]) - int(seg[1])
		for seg in data['seg-data']:
			total_length += int(seg[2]) - int(seg[1])
		if total_length != 0:
			features['CN_Burden'] = round((cn_length / total_length) * 100, 1)
		# If >50% of a cytoband arm intersects Amp/Del segments, add an Amp/Del feature for it
		with open('/data/morrisq/darmofam/cytoband_table.txt', 'r') as f:
			bedfile = f.readlines()[1:]

		arms = BedTool(bedfile)
		amps = BedTool('\n'.join(' '.join(seg) for seg in amp_segs), from_string=True)
		dels = BedTool('\n'.join(' '.join(seg) for seg in del_segs), from_string=True)
		# (round up to 3 decimal points) + (>= & <=) to be ABSOLUTELY the same (example P-0037654-T01-IM6 Amp_Xq)
		for arm in arms.coverage(amps):
			if round(float(arm[7]),3) >= 0.5:
				features['_'.join(["Amp", arm.name])] = 1
		for arm in arms.coverage(dels):
			if round(float(arm[7]),3) >= 0.5:
				features['_'.join(["Del", arm.name])] = 1
		# Clean up any temp files created by pybedtools if found
		helpers.cleanup()
	except Exception as e:
		print("ERROR loading SV and CNV")
		raise Exception("ERROR loading SV and CNV")

	try:
		ref_fasta = Fasta('/data/morrisq/darmofam/gr37.fasta')
		pyrimidines = ['C', 'T']
		purines = ['G', 'A']
		swap_dic = {'A':'T','T':'A','C':'G','G':'C'}
		def norm(tnc):
			#normalize so that pyrimidines in the center
			if tnc[1] not in pyrimidines:
				return ''.join([swap_dic[nt] for nt in tnc[::-1]])
			return tnc

		for mut in data['snp-indels']:
			if len(mut['alt_allele']) == 1 & len(mut['ref_allele']) == 1:
				chrom, start, end = mut['chromosome'], mut['start_position'], mut['start_position']
				chrom, start, end = chrom, int(start-2), int(end+1)
				ref_tri = norm(ref_fasta[chrom][start:end].seq)
				norm_alt = mut['alt_allele'] if mut['ref_allele'] in pyrimidines else swap_dic[mut['alt_allele']]
				transition_form = ref_tri[:2] + norm_alt + ref_tri[-1] 
				if transition_form in features:
					features[transition_form] += 1
				else:
					features[transition_form] = 1

	except Exception as e:
		raise Exception('Error loading SBS Counts')
		print('ERROR loading SBS counts')
	
	single_data = pd.DataFrame.from_dict(features, orient = 'index').T
	return single_data

def write_output_json(row): #single output
	feature_annotation = pd.read_csv('/home/darmofam/morris/classifier/final/implementation/feature_annotations.csv', index_col = 0) 
	pred = {'meta-data': dict(), 'results': list(),'feature_list': list()}
	pred['meta-data']['sample_id'] = row['SAMPLE_ID'].values[0]
	# hardcoded rn
	pred['meta-data']['version'] = '2.0.0'
	pred['results'].append({'tumor_type': re.sub('\.',' ',row['Pred1'].values[0]), 'confidence': str(row['Conf1'].values[0])})
	pred['results'].append({'tumor_type': re.sub('\.',' ',row['Pred2'].values[0]), 'confidence': str(row['Conf2'].values[0])})
	pred['results'].append({'tumor_type': re.sub('\.',' ',row['Pred3'].values[0]), 'confidence': str(row['Conf3'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp1'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp2'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp3'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp4'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp5'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp6'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp7'].values[0])})    
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp8'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp9'].values[0])})
	pred['feature_list'].append({'variable_name': row['Var1'].values[0], 'variable_imp': str(row['Imp10'].values[0])})
	return pred

if __name__ == "__main__":
	torch.manual_seed(3407)
	np.random.seed(0)
	print('single prediction')
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
	label = ''
	#single_data = pd.read_table('/home/darmofam/morris/classifier/final/implementation/prediction_input.txt')
	fold_ensemble = torch.load('/home/darmofam/morris/classifier/final/ensemble.pt')
	colnames = pd.read_csv('/home/darmofam/morris/classifier/final/ft_test.csv', index_col = 0).columns
	ctypes = sorted(list(pd.read_csv('/home/darmofam/morris/classifier/final/train_N.csv', index_col = 0).index.values))
	n_types = len(ctypes)
	feature_annotation = pd.read_csv('/home/darmofam/morris/classifier/final/implementation/feature_annotations.csv', index_col = 0)
	json_file_path = '/home/darmofam/morris/classifier/final/implementation/gdd_request.json'
	with open(json_file_path, 'r') as j:
		s = j.read()
		s = s.replace("\'", "\"")
		contents = json.loads(s)
	single_data = parse_dmp_json(contents)
	#run predictions
	test_x = process_data_single(single_data, colnames)
	pred_data = torch.from_numpy(test_x).float()
	pred_data = pred_data.to(device)
	fold_logits = fold_ensemble(pred_data)
	preds, probs, allprobs = pred_results(fold_logits, label) #should be zero or close to it
	pred_label = [ctypes[int(pred)] for pred in preds]
	top_pred = pred_label[0]
	res = pd.DataFrame([preds,probs,pred_label]).T
	res.columns = ['pred', 'prob', 'pred_label']

	#calculate shapley values
	train_x = np.array(pd.read_csv('/home/darmofam/morris/classifier/final/ft_train.csv', index_col=0))
	test_y = np.array([preds[0]])
	shap_values = top_shap_single(train_x, test_x, test_y, n_types, colnames, '/home/darmofam/morris/classifier/final/implementation/single_sv.csv')
	#format final res
	final_cols = ['SAMPLE_ID', 'Cancer_Type', 'Classification_Category', 'Pred1', 'Conf1', 'Pred2', 'Conf2', 'Pred3', 'Conf3'] + ctypes
	final_res = [single_data.SAMPLE_ID.values[0], single_data.Diagnosed_Cancer_Type.values[0], single_data.Classification_Category.values[0], res.pred_label.values[0], res.prob.values[0], res.pred_label.values[1], res.prob.values[1], res.pred_label.values[2], res.prob.values[2]] + list(allprobs[0])
	final_cols.extend(['Var1', 'Imp1', 'Var2', 'Imp2',  'Var3', 'Imp3',  'Var4', 'Imp4', 'Var5', 'Imp5',  'Var6', 'Imp6',  'Var7', 'Imp7',  'Var8', 'Imp8',  'Var9', 'Imp9', 'Var10', 'Imp10'])
	sv_res = []
	for fv, imp in zip(shap_values.Shapley_Columns, shap_values.Shapley_Values):
		sv_res.extend([fv, imp])
	final_res.extend(sv_res)
	final_res = pd.DataFrame(final_res).T
	final_res.columns = final_cols
	output_json = write_output_json(final_res)
	predictions_json = '/home/darmofam/morris/classifier/final/implementation/predictions_output.json'
	with open(predictions_json, "w") as outfile:
		json.dump(output_json, outfile)
	final_res.to_csv('/home/darmofam/morris/classifier/final/implementation/prediction_output.csv')



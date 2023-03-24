import pandas as pd
import numpy as np
import sys, re, json

from pybedtools import BedTool, helpers
from pyfaidx import Fasta

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
		hotspot = pd.read_csv("resources/final_hotspot_list.csv", index_col = 0) #TODO - change
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
		with open('resources/cytoband_table.txt', 'r') as f:
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
		ref_fasta = Fasta('resources/gr37.fasta')
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

if __name__ == "__main__":
	try:
		json_file_path = sys.argv[1]
	except Exception as e:
		print('specify json file path')
	try:
		output_path = sys.argv[2]
	except Exception as e:
		print('specify output csv filepath')
	with open(json_file_path, 'r') as j:
		s = j.read()
		s = s.replace("\'", "\"")
		contents = json.loads(s)
	print("Hello")
	single_data = parse_dmp_json(contents)
	single_data.to_csv(output_path)


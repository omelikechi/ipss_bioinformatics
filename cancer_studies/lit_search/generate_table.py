# Generate a LaTeX table

import pandas as pd

# specifications
cancer_type = 'glioma'
feature_type = 'mirna'
response = 'status'
max_features = 10
include_recent = False
include_article_counts = False
include_counts = True
citation_threshold = 100
fdr_max = 0.5

method_names = {'deeppink':'DeepPINK', 'ipssgb':'IPSSGB', 'ipssl':'IPSSL1', 'ipssrf':'IPSSRF', 'koglm':'KOGLM', 
	'kol':'KOL1', 'korf':'KORF', 'ssboost':'SSBoost'}

csv_path = f'./lit_results/{cancer_type}_{feature_type}_{response}.csv'
if feature_type == 'mirna':
	caption = (
		f"\\textit{{MicroRNAs and prognosis.}} MiRNAs are ordered by citation count, "
		f"with the number of publications in parentheses. A missing $q$-value indicates "
		f"the miRNA was assigned a $q$-value of less than {fdr_max} by the corresponding method."
	)
else:
	caption = (
		f"\\textit{{RNA-seq and prognosis.}} Genes are ordered by citation count, "
		f"with the number of publications in parentheses. A missing $q$-value indicates "
		f"the gene was assigned a $q$-value of less than {fdr_max} by the corresponding method."
	)
table_label = f'tab:{cancer_type}_{feature_type}_{response}'
feature_label = 'miRNA' if feature_type == 'mirna' else 'Gene' if feature_type == 'rnaseq' else print(f'Error')

# load data
df = pd.read_csv(csv_path)
df = df.fillna("")

# Combine citation and article info
df["Citations"] = df.apply(
	lambda row: f"{row['citations']}({row['articles']})" if row["citations"] != "error" else "--",
	axis=1
)

# Combine citation and article info
if include_article_counts:
	df["Citations"] = df.apply(
		lambda row: f"{row['citations']}({row['articles']})" if row["citations"] != "error" else "--",
		axis=1
	)
else:
	df["Citations"] = df.apply(
		lambda row: f"{row['citations']}" if row["citations"] != "error" else "--",
		axis=1
	)

# Optionally add recent citation column
if include_recent:
	if include_article_counts:
		df["Since 2020"] = df.apply(
			lambda row: f"{row['recent_citations']}({row['recent_articles']})"
			if row.get("recent_citations", "") != "error" else "--",
			axis=1
		)
	else:
		df["Since 2020"] = df.apply(
			lambda row: f"{row['recent_citations']}"
			if row.get("recent_citations", "") != "error" else "--",
			axis=1
		)

# Order by citation count and keep top max_features
df["_sort"] = pd.to_numeric(df["citations"], errors="coerce")
df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"]).head(max_features)

# Select method columns
method_cols = [col for col in df.columns if col not in [
	"feature", "citations", "articles", "recent_citations", "recent_articles", "Citations", "Since 2020"
]]
ordered_methods = ['ipssgb', 'ipssrf', 'ipssl', 'koglm', 'korf', 'kol', 'deeppink', 'ssboost']
ordered_methods = [m for m in ordered_methods if m in method_cols]

# Build final column order
latex_cols = ["feature", "Citations"]
if include_recent:
	latex_cols.append("Since 2020")
latex_cols += ordered_methods
df = df[latex_cols]

# Format values for LaTeX
def format_val(val):
	try:
		val = float(val)
		return f"{val:.2f}"
	except:
		return "--"

df_formatted = df.copy()
df_formatted[ordered_methods] = df_formatted[ordered_methods].map(format_val)

# Generate LaTeX
print("\\begin{table}[ht]")
print("\\centering")
col_format = "l" + "c" * (df_formatted.shape[1] - 1)
print(f"\\begin{{tabular}}{{{col_format}}}")
print("\\toprule")

header = [f"\\textbf{{{feature_label}}}", "\\textbf{Citations}"]
if include_recent:
	header.append("\\textbf{Since 2020}")
for method in ordered_methods:
	header += [f"\\texttt{{{method_names[method]}}}"]
print(" & ".join(header) + " \\\\")
print("\\midrule")

# for _, row in df_formatted.iterrows():
# 	print(" & ".join(str(val) for val in row) + " \\\\")

for _, row in df.iterrows():
	formatted_row = [str(row["feature"]), str(row["Citations"])]
	if include_recent:
		formatted_row.append(str(row["Since 2020"]))
	for method in ordered_methods:
		val = row[method]
		try:
			val = float(val)
			formatted_row.append(f"{val:.2f}" if val <= fdr_max else "--")
		except:
			formatted_row.append("--")
	print(" & ".join(formatted_row) + " \\\\")

if include_counts:
	print("\\midrule")
	# Compute summary rows on full data
	df_full = pd.read_csv(csv_path).fillna("")
	cit_raw_full = pd.to_numeric(df_full["citations"], errors="coerce")

	summary_rows = {
		f'$\\geq$ {citation_threshold}': [],
		f'$<$ {citation_threshold}': [],
		"Total": []
	}

	for method in ordered_methods:
		if method not in df_full.columns:
			summary_rows[f'$\\geq$ {citation_threshold}'].append("--")
			summary_rows[f'$<$ {citation_threshold}'].append("--")
			summary_rows["Total"].append("--")
			continue

		qvals = pd.to_numeric(df_full[method], errors="coerce")
		is_selected = qvals < fdr_max

		above = ((cit_raw_full > citation_threshold) & is_selected).sum()
		below = ((cit_raw_full <= citation_threshold) & is_selected).sum()
		total = is_selected.sum()

		summary_rows[f'$\\geq$ {citation_threshold}'].append(str(above))
		summary_rows[f'$<$ {citation_threshold}'].append(str(below))
		summary_rows["Total"].append(str(total))

	# Print summary rows to LaTeX
	for label, values in summary_rows.items():
		row_prefix = f"{label.format(threshold=citation_threshold)} & --"
		if include_recent:
			row_prefix += " & --"
		print(row_prefix + " & " + " & ".join(values) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print(f"\\caption{{{caption}}}")
print(f"\\label{{{table_label}}}")
print("\\end{table}")
print()






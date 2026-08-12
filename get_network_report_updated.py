#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats
from math import comb
import scipy
from statsmodels.stats.multitest import multipletests
from scipy.stats import combine_pvalues

# =============================================================================
# Filtering parameters
# =============================================================================

present = 11

### Difference filters
KO_WK3_MIN = -0.4
KO_WK3_MAX = 0.4

KO_WK6_MIN = -0.4
KO_WK6_MAX = 0.4

WK3_WK6_MIN = -100
WK3_WK6_MAX = 100

DIFF = "yes"

META_P = 0.05
FISHER_P = 1

### Meta-analysis filters
META_FDR = 1
FISHER_FDR = 1

### Indivival p-value filters
KO_PVAL = 0.17
WK3_PVAL = 0.48
WK6_PVAL = 0.41

KO_N = 12
WK3_N = 5
WK6_N = 6

# =============================================================================
# Files
# =============================================================================

INPUT_CSV = f"./TLS_present{present}_100pct_concordant.csv"

OUTPUT_REPORT = f"./reports/TLS_present{present}_KO-{KO_PVAL}_Wk3-{WK3_PVAL}_Wk6-{WK6_PVAL}_comb-{META_P}_FDR-{META_FDR}_DIFF_{KO_WK3_MAX}_FLIP_report.txt"

NODE_SIGN_FILE = ("/nfs7/PHARM/Morgun_Lab/amanda/TkNA_analysis/TLS_network/combined_TLS_diff.csv")

# =============================================================================
# Load network
# =============================================================================

df = pd.read_csv(
    INPUT_CSV,
    low_memory=False
)

starting_edges = len(df)

print(f"Starting edges: {starting_edges:,}")

# =============================================================================
# Load node signs
# =============================================================================

node_sign_df = pd.read_csv(
    NODE_SIGN_FILE
)

node_sign = {}

for _, row in node_sign_df.iterrows():

    gene = str(row["variable"]).strip()

    if row["avg_log2FC"] > 0:
        node_sign[gene] = "+"

    elif row["avg_log2FC"] < 0:
        node_sign[gene] = "-"

# =============================================================================
# PUC Function
# =============================================================================

def is_puc(edge_name, corr_value):

    try:
        gene1, gene2 = [x.strip() for x in edge_name.split("<==>")]
    except ValueError:
        return False

    # Missing node sign
    if gene1 not in node_sign or gene2 not in node_sign:
        return False

    sign1 = node_sign[gene1]
    sign2 = node_sign[gene2]

    # Expected correlation sign
    if sign1 == sign2:
        expected = 1      # positive correlation
    else:
        expected = -1     # negative correlation

    observed = np.sign(corr_value)

    return observed == expected

df["PUC_Compliant"] = df.apply(
    lambda row: is_puc(
        row["Edge Name"],
        row["KO_median_median"]
    ),
    axis=1
)

### Apply PUC
df = (
    df[df["PUC_Compliant"]]
    .drop(columns="PUC_Compliant")
    .copy()
)

edges_after_PUC = len(df)

# =============================================================================
# Edge correlation statistics
# =============================================================================

KO_edges = df["KO_median_median"]
WK3_edges = df["wk3_median_median"]
WK6_edges = df["wk6_median_median"]

def summarize(values):

    values = np.asarray(values)

    values = values[~np.isnan(values)]

    return {
        "N": len(values),
        "Mean": np.mean(values),
        "Median": np.median(values),
        "Std": np.std(values, ddof=1),
        "Minimum": np.min(values),
        "Maximum": np.max(values),
        "Mean_abs": np.mean(np.abs(values)),
        "Median_abs": np.median(np.abs(values))
    }



df["KO_minus_wk3"] = (
    np.abs(KO_edges)
    -
    np.abs(WK3_edges)
)

df["KO_minus_wk6"] = (
    np.abs(KO_edges)
    -
    np.abs(WK6_edges)
)

df["wk3_minus_wk6"] = (
    np.abs(WK3_edges)
    -
    np.abs(WK6_edges)
)

df = df[
    (df["KO_minus_wk3"] >= KO_WK3_MIN) &
    (df["KO_minus_wk3"] <= KO_WK3_MAX) &
    (df["KO_minus_wk6"] >= KO_WK6_MIN) &
    (df["KO_minus_wk6"] <= KO_WK6_MAX) &
    (df["wk3_minus_wk6"] >= WK3_WK6_MIN) &
    (df["wk3_minus_wk6"] <= WK3_WK6_MAX)
].copy()

edges_after_diff = len(df)

# =============================================================================
# Meta-analysis Function
# =============================================================================

def metacor(r, n, do_fisher=True, cor_type="spearman", meta_type="fixed"):

    if cor_type not in ["pearson", "spearman"]:
        raise Exception(f'Invalid correlation type "{cor_type}"')

    if meta_type not in ["fixed", "random"]:
        raise Exception(f'Invalid meta analysis type "{meta_type}"')

    if not do_fisher:
        raise Exception("Non-Fisher transform not implemented")

    if meta_type == "random":
        raise Exception("Random effects not implemented")

    if do_fisher:

        eps = 1e-12
        r = np.clip(r, -1 + eps, 1 - eps)
        r = np.arctanh(r)

    if cor_type == "pearson":

        variance = 1 / (n - 3)

    else:

        variance = (1 + (r**2)/2) / (n - 3)

    weight = 1 / variance

    variance_common = 1 / np.sum(weight, axis=1)

    r_common = np.sum(weight * r, axis=1) * variance_common

    stddev_common = np.sqrt(variance_common)

    z = r_common / stddev_common

    p = 2 * scipy.stats.norm.sf(np.abs(z))

    if do_fisher:

        r_common = np.tanh(r_common)

    return r_common, p




# =============================================================================
# Apply Meta-analysis
# =============================================================================


sample_sizes = [KO_N, WK3_N, WK6_N]

r = df[
    [
        "KO_median_median",
        "wk3_median_median",
        "wk6_median_median"
    ]
].to_numpy()

n = np.array(
    [sample_sizes] * len(df)
)

meta_r, meta_p = metacor(r, n)

meta_fdr = multipletests(
    meta_p,
    method="fdr_bh"
)[1]

df["meta_r"] = meta_r
df["meta_p"] = meta_p
df["meta_fdr"] = meta_fdr

# =============================================================================
# Apply Fisher's combined p-value
# =============================================================================

pvals = df[
    [
        "KO_pval_median",
        "wk3_pval_median",
        "wk6_pval_median"
    ]
].to_numpy()

fisher_p = np.empty(len(df))

for i in range(len(df)):
    _, fisher_p[i] = combine_pvalues(
        pvals[i],
        method="fisher"
    )

fisher_fdr = multipletests(
    fisher_p,
    method="fdr_bh"
)[1]

df["fisher_p"] = fisher_p
df["fisher_fdr"] = fisher_fdr


# =============================================================================
# Filter by median p-value
# =============================================================================

df = df[
    (df["KO_pval_median"] < KO_PVAL) &
    (df["wk3_pval_median"] < WK3_PVAL) &
    (df["wk6_pval_median"] < WK6_PVAL)
].copy()

edges_after_indpval = len(df)


### Filter by combined P (meta)
df = df[
    (df["meta_p"] < META_P) &
    (df["fisher_p"] < FISHER_P)
].copy()

edges_after_combP = len(df)

### Filter by FDR
df = df[
    (df["meta_fdr"] < META_FDR) &
    (df["fisher_fdr"] < FISHER_FDR)
].copy()

edges_after_FDR = len(df)

# =============================================================================
# Filter by median p-value
# =============================================================================

#df = df[
#    (df["KO_pval_median"] < KO_PVAL) &
#    (df["wk3_pval_median"] < WK3_PVAL) &
#    (df["wk6_pval_median"] < WK6_PVAL)
#].copy()

#edges_after_indpval = len(df)


# =============================================================================
# Final edge vectors (post-filtering)
# =============================================================================

KO_edges = df["KO_median_median"]
WK3_edges = df["wk3_median_median"]
WK6_edges = df["wk6_median_median"]

# =============================================================================
# Optionally: Save filtered df
# =============================================================================

OUTPUT_EDGES = (
    f"./filtered_edges_present{present}"
    f"_KO-{KO_PVAL}_Wk3-{WK3_PVAL}_Wk6-{WK6_PVAL}"
    f"_comb-{META_P}_FDR-{META_FDR}_DIFF_{KO_WK3_MAX}_FLIP.csv"
)

df.to_csv(
    OUTPUT_EDGES,
    index=False
)

# =============================================================================
# Network topology
# =============================================================================

edge_count = len(df)

nodes = set()

for edge in df["Edge Name"]:

    a, b = edge.split("<==>")

    nodes.add(a)
    nodes.add(b)


node_count = len(nodes)


possible_edges = (
    node_count *
    (node_count - 1)
    / 2
)

density = edge_count / possible_edges


# =============================================================================
# Node count
# =============================================================================

pos_nodes = 0
neg_nodes = 0

for node in nodes:

    if node_sign.get(node) == "+":
        pos_nodes += 1

    elif node_sign.get(node) == "-":
        neg_nodes += 1


# =============================================================================
# Edge correlation sign balance
# =============================================================================

pos_edges = 0
neg_edges = 0


for _, row in df.iterrows():

    # use one condition because edges are already concordant
    corr = row["KO_median_median"]

    if corr > 0:
        pos_edges += 1

    elif corr < 0:
        neg_edges += 1


# -----------------------------------
# Observed ratio
# -----------------------------------

if neg_edges > 0:

    obs_posneg_ratio = round(
        pos_edges / neg_edges,
        2
    )

else:

    obs_posneg_ratio = np.nan

# -----------------------------------
# Expected ratio
# -----------------------------------

expec_pos = (
    comb(pos_nodes, 2)
    if pos_nodes >= 2
    else 0
)

expec_pos += (
    comb(neg_nodes, 2)
    if neg_nodes >= 2
    else 0
)


expec_neg = (
    pos_nodes * neg_nodes
)


if expec_neg > 0:

    ideal_ratio_posneg = round(
        expec_pos / expec_neg,
        2
    )

else:

    ideal_ratio_posneg = np.nan

# -----------------------------------
# Normalized deviation
# -----------------------------------

if (
    pd.notna(obs_posneg_ratio)
    and pd.notna(ideal_ratio_posneg)
    and ideal_ratio_posneg != 0
):

    dev_norm_posneg = round(
        (
            obs_posneg_ratio -
            ideal_ratio_posneg
        )
        /
        ideal_ratio_posneg,
        2
    )

else:

    dev_norm_posneg = np.nan


def run_wilcoxon(x, y):

    diff = (
        np.abs(x)
        -
        np.abs(y)
    )

    diff = diff[~np.isnan(diff)]


    if len(diff) == 0:

        return np.nan, np.nan, "No data"


    if np.all(diff == 0):

        return np.nan, np.nan, "All differences are zero"


    result = stats.wilcoxon(
        np.abs(x),
        np.abs(y)
    )

    return (
        result.statistic,
        result.pvalue,
        ""
    )



wilcox = {

    "KO vs wk3":
        run_wilcoxon(
            KO_edges,
            WK3_edges
        ),

    "KO vs wk6":
        run_wilcoxon(
            KO_edges,
            WK6_edges
        ),

    "wk3 vs wk6":
        run_wilcoxon(
            WK3_edges,
            WK6_edges
        )
}

# =============================================================================
# Write report
# =============================================================================

with open(OUTPUT_REPORT, "w") as f:

    f.write("="*70 + "\n")
    f.write("TLS Network Report\n")
    f.write("="*70 + "\n\n")


    f.write("Filtering\n")
    f.write("-"*70 + "\n")

    f.write(f"Starting edges:                {starting_edges:,}\n")
    f.write(f"After PUC:                     {edges_after_PUC:,}\n")
    f.write(f"After difference filters:      {edges_after_diff:,}\n")
    f.write(f"After combined p (meta):       {edges_after_combP:,}\n")
    f.write(f"After meta/Fisher FDR:         {edges_after_FDR:,}\n")
    f.write(f"After individual p-values:     {edges_after_indpval:,}\n\n")

    f.write(f"KO vs Wk3 cutoff: Between {KO_WK3_MIN} and {KO_WK3_MAX}\n")
    f.write(f"KO vs Wk6 cutoff: Between {KO_WK6_MIN} and {KO_WK6_MAX}\n")
    f.write(f"Wk3 vs Wk6 cutoff: Between {WK3_WK6_MIN} and {WK3_WK6_MAX}\n")

    f.write(f"KO p-value cutoff: {KO_PVAL}\n")
    f.write(f"wk3 p-value cutoff: {WK3_PVAL}\n")
    f.write(f"wk6 p-value cutoff: {WK6_PVAL}\n")

    f.write(f"Meta combined p-value cutoff: {META_P}\n")
    f.write(f"Fisher combined p-value cutoff: {FISHER_P}\n")

    f.write(f"Meta FDR cutoff: {META_FDR}\n")
    f.write(f"Fisher FDR cutoff: {FISHER_FDR}\n\n")


    f.write("Network topology\n")
    f.write("-"*70 + "\n")

    f.write(f"Edges: {edge_count:,}\n")
    f.write(f"Nodes: {node_count:,}\n")
    f.write(f"Density: {density:.6f}\n\n")

    f.write(f"Positive nodes: {pos_nodes:,}\n")
    f.write(f"Negative nodes: {neg_nodes:,}\n\n")

    f.write(f"Positive edges: {pos_edges:,}\n")
    f.write(f"Negative edges: {neg_edges:,}\n")

    f.write(f"Observed +:- ratio: {obs_posneg_ratio:.4f}\n")
    f.write(f"Expected +:- ratio: {ideal_ratio_posneg:.4f}\n")
    f.write(f"Normalized deviation: {dev_norm_posneg:.4f}\n\n")


    f.write("Edge correlation statistics\n")
    f.write("-"*70 + "\n")

    for name, values in {
        "KO": KO_edges,
        "wk3": WK3_edges,
        "wk6": WK6_edges
    }.items():

        s = summarize(values)

        f.write(f"\n{name}\n")

        for k, v in s.items():

            f.write(
                f"{k}: {v:.4f}\n"
            )


    f.write("\n\nAbsolute correlation differences\n")
    f.write("-"*70 + "\n")

    for name, values in {
        "KO-wk3": df["KO_minus_wk3"],
        "KO-wk6": df["KO_minus_wk6"],
        "wk3-wk6": df["wk3_minus_wk6"]
    }.items():

        s = summarize(values)

        f.write(f"\n{name}\n")

        for k, v in s.items():

            f.write(
                f"{k}: {v:.4f}\n"
            )


    f.write("\n\nMeta FDR statistics\n")
    f.write("-"*70 + "\n")

    s = summarize(df["meta_fdr"])

    for k, v in s.items():

        f.write(
            f"{k}: {v:.4f}\n"
        )


    f.write("\n\nWilcoxon tests\n")
    f.write("-"*70 + "\n")

    for name, result in wilcox.items():

        stat, p, note = result

        if note:

            f.write(
                f"{name}: {note}\n"
            )

        else:

            f.write(
                f"{name}: "
                f"Statistic={stat:.4f}, "
                f"P={p:.4e}\n"
            )


print(f"Report written: {OUTPUT_REPORT}")

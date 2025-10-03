import gzip
import pandas as pd

# Paths to OrthoDB files
species_file = "./orthodb/odb12v1_species.tab.gz"
genes_file = "./orthodb/odb12v1_genes.tab.gz"
og2genes_file = "./orthodb/odb12v1_OG2genes.tab.gz"

# Step 1: Get OrthoDB species IDs for axolotl and human
species_map = {}
with gzip.open(species_file, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            odb_species_id, taxid, species_name = parts[0], parts[1], parts[2]
            if taxid == "9606_0":  # Human
                species_map["human"] = taxid
            elif taxid == "8319_0":  # Axolotl
                species_map["axolotl"] = taxid

print(f"Species IDs: {species_map}")

# Step 2: Parse OG2genes and keep only human/axolotl rows
rows = []
with gzip.open(og2genes_file, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            og_id, gene_id, species_id = parts[0], parts[1], parts[2]
            if species_id in species_map.values():
                rows.append((og_id, gene_id, species_id))

df = pd.DataFrame(rows, columns=["OG_ID", "Gene_ID", "Species_ID"])
print(f"Filtered OG2genes rows: {len(df)}")

# Step 3: Load gene symbols for those Gene_IDs
gene_map = {}
with gzip.open(genes_file, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 4:
            gene_id, species_id, symbol, description = parts[0], parts[1], parts[2], parts[3]
            if gene_id in df["Gene_ID"].values:
                gene_map[gene_id] = symbol

# Add symbols to df
df["Gene_Symbol"] = df["Gene_ID"].map(gene_map)

# Step 4: Pivot so each OG_ID has human & axolotl genes side by side
pivoted = df.pivot_table(
    index="OG_ID",
    columns="Species_ID",
    values="Gene_Symbol",
    aggfunc=lambda x: ";".join(x)
).reset_index()

pivoted.rename(columns={
    species_map["human"]: "Human_Genes",
    species_map["axolotl"]: "Axolotl_Genes"
}, inplace=True)

# Save result
pivoted.to_csv("axolotl_human_orthologs.csv", index=False)
print("Saved axolotl_human_orthologs.csv")

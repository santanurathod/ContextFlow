# Define file paths
rds_matrix_file_path <- "/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/h5ad_processed_datasets/GSE092025.rds"
rds_metadata_file_path <- "/Users/rssantanu/Desktop/codebase/constrained_FM/datasets/h5ad_processed_datasets/GSE092025_metadata.rds"

# Load the gene expression matrix RDS file
gene_expression_matrix <- readRDS(rds_matrix_file_path)

# Check the structure of the loaded object
str(gene_expression_matrix)

# If the object is not a matrix, you might need to extract it from a list or other structure
# For example, if it's a list, you might need to do something like:
# gene_expression_matrix <- gene_expression_matrix$your_matrix_name

# Load the metadata RDS file
metadata <- readRDS(rds_metadata_file_path)

# Check the structure of the metadata
str(metadata)

# Save the matrix and metadata to CSVge
write.csv(gene_expression_matrix, "gene_expression_matrix.csv", row.names = TRUE)
write.csv(metadata, "metadata.csv", row.names = FALSE)

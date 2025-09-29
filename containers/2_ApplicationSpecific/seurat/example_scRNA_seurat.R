# --------- Load Libraries ---------
library(Seurat)
library(Matrix)
library(dplyr)
library(ggplot2)

# --------- Define Input Files and Sample Labels ---------
# Sample input files for this example are located in /util/software/data/seurat/.
# To use your own files, replace these samples with your own, ensuring the file names and numbering follow your intended convention.
data_path <- "/util/software/data/seurat"

sample_files <- c(
  "sample1.csv", "sample2.csv",
  "sample3.csv", "sample4.csv"
)
sample_labels <- c("S1", "S2", "S3", "S4")

# --------- Load and Create Seurat Objects ---------
sample_list <- list()
for (i in seq_along(sample_files)) {
  file <- file.path(data_path, sample_files[i])
  label <- sample_labels[i]

  message("Reading: ", file)
  counts <- read.csv(file, row.names = 1)
  counts <- as.matrix(counts)
  counts <- Matrix(counts, sparse = TRUE)

  seu <- CreateSeuratObject(counts = counts, project = label, min.cells = 3, min.features = 200)
  seu$sample <- label
  sample_list[[label]] <- seu
}

# --------- Merge All Samples ---------
combined <- Reduce(function(x, y) merge(x, y, add.cell.ids = c(x$sample[1], y$sample[1])), sample_list)

# --------- correct batch effec-------------
sample_list <- lapply(sample_list, NormalizeData)
sample_list <- lapply(sample_list, FindVariableFeatures)
anchors <- FindIntegrationAnchors(object.list = sample_list)
combined <- IntegrateData(anchorset = anchors)
DefaultAssay(combined) <- "integrated" # If using integration

# --------- Quality Control ---------
DefaultAssay(combined) <- "RNA"
combined[["RNA"]] <- JoinLayers(combined[["RNA"]])

combined[["percent.mt"]] <- PercentageFeatureSet(combined, pattern = "^MT-")

png("QC_violin_plots.png", width = 1000, height = 400)
VlnPlot(combined, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
dev.off()

png("FeatureScatter_nCount_vs_nFeature.png", width = 600, height = 500)
FeatureScatter(combined, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
dev.off()

png("FeatureScatter_percentMT_vs_nCount.png", width = 600, height = 500)
FeatureScatter(combined, feature1 = "nCount_RNA", feature2 = "percent.mt")
dev.off()

# --------- Filter Cells check QC_violin_plots.png plot to set up the cutoffs ---------
combined <- subset(combined, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 10)

combined <- ScaleData(combined)

# --------- PCA & Elbow Plot ---------
combined <- RunPCA(combined, features = VariableFeatures(combined))

png("ElbowPlot.png", width = 600, height = 500)
ElbowPlot(combined, ndims = 50)
dev.off()

# --------- Clustering and Dimension Reduction. Check ElbowPlot to set up the 1:20 cutoff to 1:n---------
combined <- FindNeighbors(combined, dims = 1:20)
combined <- FindClusters(combined, resolution = 0.5)
combined <- RunUMAP(combined, dims = 1:20)
combined <- RunTSNE(combined, dims = 1:20)

# --------- Plot UMAP and t-SNE ---------
png("umap_by_cluster.png", width = 800, height = 700)
DimPlot(combined, reduction = "umap", group.by = "seurat_clusters", label = TRUE) + ggtitle("UMAP - Clusters")
dev.off()

png("tsne_by_sample.png", width = 800, height = 700)
DimPlot(combined, reduction = "tsne", group.by = "sample") + ggtitle("t-SNE - Sample Origin")
dev.off()

# --------- Output Genes Expressed Per Cluster ---------
cluster_ids <- levels(Idents(combined))

#  Genes expressed in ≥10% of cells in each cluster
 threshold <- 0.10
 for (cluster in cluster_ids) {
 cells <- WhichCells(combined, idents = cluster)
 mat <- GetAssayData(combined, layer = "counts")[, cells]
 freq <- Matrix::rowMeans(mat > 0)
 genes <- names(freq[freq >= threshold])
 write.table(genes, paste0("frequent_genes_10pct_cluster_", cluster, ".txt"),
              row.names = FALSE, col.names = FALSE, quote = FALSE)
}

# --------- Find Marker Genes (per cluster) DEG ---------
deg_markers <- FindAllMarkers(
  object = combined,
  assay = "RNA",
  only.pos = TRUE,
  min.pct = 0.25,
  logfc.threshold = 0.25,
  layer = "data"
)
write.csv(deg_markers, "DEGs_by_cluster.csv", row.names = FALSE)


# --------- Save Final Seurat Object ---------
saveRDS(combined, file = "combined_seurat_final.rds")

message("✅ Analysis complete.")

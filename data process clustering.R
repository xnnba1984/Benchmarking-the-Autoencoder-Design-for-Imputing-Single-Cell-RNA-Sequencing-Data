library(Seurat)
library(Matrix)
library(aricode)

# read data matrix
# after reading, suppose the data matrix is saved in 'data'
# the true cell type labels are saved in 'type'
# Seurat preprocessing
data.seurat <- CreateSeuratObject(counts = data, project = "data", min.cells = 3);data.seurat
data.seurat <- NormalizeData(data.seurat, normalization.method = "LogNormalize", scale.factor = 10000)
data.seurat <- FindVariableFeatures(data.seurat, selection.method = "vst", nfeatures = 2000);data.seurat
gene.variable <- VariableFeatures(data.seurat)
norm <- data.seurat@assays[["RNA"]]@data; dim(norm)
norm <- norm[gene.variable,]; dim(norm)

# kmeans clustering on datasets before imputation
# similar code can also apply to data after imputation
set.seed(2020)
system.time({
  kmeans.result <- kmeans(t(norm), centers = length(table(type)), nstart = 25)
})
ARI <- round(ARI(type,kmeans.result$cluster), 4); ARI
AMI <- round(AMI(type,kmeans.result$cluster), 4); AMI

# split to training and validation
set.seed(2020)
train.index <- sample(1:ncol(norm), size = round(0.8 * ncol(norm)), replace = F)
norm.train <- norm[,train.index]; dim(norm.train)
norm.test <- norm[,-train.index]; dim(norm.test)

# save processed data matrix in mtx format for Python/Pytorch
# save cell types
# variable dataset is a user-specified data name
folder.save <- 'cluster dataset new/'
writeMM(norm.train, paste(folder.save, dataset, '_', 'train.mtx', sep = ''))
writeMM(norm.test, paste(folder.save, dataset, '_', 'test.mtx', sep = ''))
writeMM(norm, paste(folder.save, dataset, '.mtx', sep = ''))
saveRDS(type, paste(folder.save, dataset, '_type.rds', sep = ''))



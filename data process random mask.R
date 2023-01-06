library(Seurat)
library(Matrix)

seed <- 2020
set.seed(seed)

# read data matrix
# after reading, suppose the data matrix is saved in 'data'
# add col and row names for seurat object
colnames(data) <- as.character(1:dim(data)[2]); colnames(data)
rownames(data) <- as.character(1:dim(data)[1]); rownames(data)

# Seurat preprocessing
data.seurat <- CreateSeuratObject(counts = data, project = "data", min.cells = 3, min.features = 200);data.seurat
data.seurat <- NormalizeData(data.seurat, normalization.method = "LogNormalize", scale.factor = 10000)
data.seurat <- FindVariableFeatures(data.seurat, selection.method = "vst", nfeatures = 2000);data.seurat
gene.variable <- VariableFeatures(data.seurat)

# obtain data matrix containing highly variable genes
norm <- data.seurat@assays[["RNA"]]@data; dim(norm)
norm <- norm[gene.variable,]; dim(norm)

# nonzero index
index.nonzero <- which(norm!=0, arr.ind = T); dim(index.nonzero)
set.seed(seed)
# sample 50% nonzero index
mask.row <- sample(x = 1:nrow(index.nonzero), size = round(nrow(index.nonzero)/2),replace = F)
index.zero <- index.nonzero[mask.row,]; dim(index.zero)
index.keep <- index.nonzero[-mask.row,]; dim(index.keep)

# mask matrix 
mask <- sparseMatrix(i = index.zero[,1], j = index.zero[,2], x = T, dims = dim(norm)); dim(mask)
# no mask matrix
nomask <- sparseMatrix(i = index.keep[,1], j = index.keep[,2], x = T, dims = dim(norm)); dim(nomask)

# mask to zero
norm.mask = norm
norm.mask[mask] <- 0

# split to 80% training and 20% validation
set.seed(seed)
train.index <- sample(1:ncol(norm.mask), size = round(0.8 * ncol(norm.mask)), replace = F)
norm.mask.train <- norm.mask[,train.index]; dim(norm.mask.train)
norm.mask.test <- norm.mask[,-train.index]; dim(norm.mask.test)

# save processed dataset and mask matrix in mtx format for Python/Pytorch
# variable dataset is a user-specified data name
writeMM(norm.mask.train, paste('masked dataset/', dataset, '_mask_train.mtx', sep = ''))
writeMM(norm.mask.test, paste('masked dataset/', dataset, '_mask_test.mtx', sep = ''))
writeMM(mask, paste('masked dataset/', dataset, '_mask.mtx', sep = ''))
writeMM(norm, paste('masked dataset/', dataset, '_data.mtx', sep = ''))


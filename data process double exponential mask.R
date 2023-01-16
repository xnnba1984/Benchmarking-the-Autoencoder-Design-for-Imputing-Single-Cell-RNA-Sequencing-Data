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
cell.name <- colnames(data.seurat)

# get counts on hvg 
data.variable <- data[gene.variable, cell.name]; dim(data.variable); data.variable[1:10, 1:10]
data.variable <- log(data.variable+1)

# zero rate after mask 50% non-zero element
rate.zero <- (sum(data.variable==0) + sum(data.variable!=0) * 0.5) / 
  (dim(data.variable)[1] * dim(data.variable)[2]); rate.zero

# solve lambda in double exponential
f <- function(lambda){
  data.addzero <- apply(data.variable, 1, FUN = function(x){
    index.zero <- rbinom(length(x), size = 1, prob = exp(-lambda*(mean(x[x > 0]))^2))
    return(x * (1-index.zero))
  })
  return(sum(data.addzero == 0) / (dim(data.addzero)[1] * dim(data.addzero)[2]) - rate.zero)
}
system.time({
  set.seed(seed)
  lambda <- uniroot(f, c(0,200))$root
}); lambda

# mask data by probability and lambda
mask <- apply(data.variable, 1, FUN = function(x){
  return(rbinom(length(x), size = 1, prob = exp(-lambda*(mean(x[x > 0]))^2)))
}); dim(mask)
mask <- t(mask); dim(mask)
mask <- (data.variable * mask) > 0; dim(mask)
mask <- as(as.matrix(mask), 'ngTMatrix'); mask[1:10,1:10]
sum((mask*data.variable)==0) / (dim(data.variable)[1] * dim(data.variable)[2])

# get norm data before masking
norm <- data.seurat@assays[["RNA"]]@data; dim(norm)
norm <- norm[gene.variable,]; dim(norm); norm[1:10, 1:10]

# mask on norm
norm.mask <- norm
norm.mask[mask] <- 0; norm.mask[1:10, 1:10]; norm[1:10, 1:10]
dim(norm.mask)

# split to training and validation
set.seed(seed)
train.index <- sample(1:ncol(norm.mask), size = round(0.8 * ncol(norm.mask)), replace = F)
norm.mask.train <- norm.mask[,train.index]; dim(norm.mask.train)
norm.mask.test <- norm.mask[,-train.index]; dim(norm.mask.test)

# save processed dataset and mask matrix in mtx format for Python/Pytorch
# variable dataset is a user-specified data name
writeMM(norm.mask.train, paste('masked dataset double exponential/', dataset, '_mask_train.mtx', sep = ''))
writeMM(norm.mask.test, paste('masked dataset double exponential/', dataset, '_mask_test.mtx', sep = ''))
writeMM(mask, paste('masked dataset double exponential/', dataset, '_mask.mtx', sep = ''))
writeMM(norm, paste('masked dataset double exponential/', dataset, '_data.mtx', sep = ''))










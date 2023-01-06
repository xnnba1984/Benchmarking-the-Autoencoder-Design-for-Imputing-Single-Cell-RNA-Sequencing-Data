library(Seurat)
library(Matrix)
library(scDesign)

# read data matrix
# after reading, suppose the data matrix is saved in 'data'
# add col and row names for seurat object
colnames(data) <- as.character(1:dim(data)[2])
rownames(data) <- as.character(1:dim(data)[1])

# Seurat preprocessing
data.seurat <- CreateSeuratObject(counts = data, project = "data", min.cells = 3, min.features = 1);data.seurat
data.seurat <- NormalizeData(data.seurat, normalization.method = "LogNormalize", scale.factor = 10000);data.seurat
data.seurat <- FindVariableFeatures(data.seurat, selection.method = "vst", nfeatures = 2000);data.seurat
gene.variable <- VariableFeatures(data.seurat)
count <- data[gene.variable,]; dim(count)

# library size
lsize <- median(apply(count, 2, sum)); lsize

# set parameters for simulator scDesign
cell.number <- 1000
squence.depth <- lsize * cell.number
cell.type <- 2
pUp <- .05
pDown <- .05
fU <- 5
fL <- 1.5 
ncores <- 1

# simulate data with DE genes
set.seed(2020)
system.time(simdata <- design_data(realcount = count, S = rep(squence.depth, cell.type), 
                                   ncell = rep(cell.number, cell.type), ngroup = cell.type, 
                                   pUp = pUp, pDown = pDown, fU = fU, fL = fL, ncores = ncores))
sim1 <- simdata$count[[1]]; dim(sim1)
sim2 <- simdata$count[[2]]; dim(sim2)
sim <- cbind(sim1, sim2); dim(sim)
mean(sim==0)
de.truth <- c(simdata[["genesUp"]][[2]], simdata[["genesDown"]][[2]])

# DE gene analysis on data before imputation
# similar code can also apply to data after imputation
set.seed(2020)
de.seurat <- CreateSeuratObject(counts = sim, project = "data", min.cells = 0, min.features = 0);de.seurat
de.seurat <- NormalizeData(de.seurat, normalization.method = "LogNormalize", scale.factor = 10000);de.seurat
de.seurat <- FindVariableFeatures(de.seurat, selection.method = "vst", nfeatures = 2000);de.seurat
levels(de.seurat)
de <- FindMarkers(de.seurat, ident.1 = "C1", ident.2 = "C2", test.use = 'MAST')
de.test <- rownames(de[de$p_val_adj <= 0.05,])
norm <- de.seurat@assays[["RNA"]]@data; dim(norm)

# DE result
tp <- length(intersect(de.test, de.truth)); tp
fp <- length(setdiff(de.test, de.truth)); fp
fn <- length(setdiff(de.truth, de.test)); fn
nde.truth <- setdiff(rownames(de.seurat), de.truth); length(nde.truth)
nde.test <- setdiff(rownames(de.seurat), de.test); length(nde.test)
tn <- length(intersect(nde.truth, nde.test)); tn

precision <- tp / (tp + fp); round(precision, 4)
recall <- tp / (tp + fn); round(recall, 4)
tnr <- tn / (tn + fn); round(tnr, 4)
f1 <- 2 * precision * recall / (precision + recall); round(f1, 4)

# split simulated DE data to training and validation
set.seed(2020)
train.index <- sample(1:ncol(norm), size = round(0.8 * ncol(norm)), replace = F)
norm.train <- norm[,train.index]; dim(norm.train)
norm.test <- norm[,-train.index]; dim(norm.test)

# save processed data matrix in mtx format for Python/Pytorch
# save ground-truth DE genes and all genes
# variable dataset is a user-specified data name
folder.save <- 'DE dataset new/'
writeMM(norm.train, paste(folder.save, dataset, '_', 'train.mtx', sep = ''))
writeMM(norm.test, paste(folder.save, dataset, '_', 'test.mtx', sep = ''))
writeMM(norm, paste(folder.save, dataset, '.mtx', sep = ''))
saveRDS(de.truth, paste(folder.save, dataset, '_de.rds', sep = ''))
saveRDS(list(rownames(de.seurat),colnames(de.seurat)), paste(folder.save, dataset, '_gene_cell.rds', sep = ''))











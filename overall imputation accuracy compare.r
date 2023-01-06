library(Matrix)
library(RcppCNPy)
library(reticulate)
library(readxl)

##########################################################
# depth and width
##########################################################
# user-specified data folder and name
folder.data <- 'masked dataset median/'
dataset <- 'mbrain'
file.mask <- paste(folder.data, dataset, '_mask.mtx', sep = '')
file.norm <- paste(folder.data, dataset, '_data.mtx', sep = '')
np <- import("numpy")
mask <- readMM(file.mask); dim(mask)
norm <- readMM(file.norm); dim(norm)

mask <- as.matrix(mask)
norm <- as.matrix(norm)
dim(mask); dim(norm)

# col: layer #; row: repeat #
r <- 10  # repeat
h <- 15  # depth
width <- c(32,64,128,256)
folder.imputed <- 'imputed dataset median/'
nrmse.cor.matrix <- matrix(data = 0, ncol = length(width)*2, nrow = h)

system.time({
for(j in 1:length(width)){
  w <- width[j]
  print('================================')
  print(paste('width', w, sep = ' '))
  print('================================')
  nrmse.mask.matrix <- matrix(data = 0, ncol = h, nrow = r)
  cor.mask.matrix <- matrix(data = 0, ncol = h, nrow = r)
  
  for(layer in 1:h){
    print(paste('layer',layer, sep = ' '))
    for(i in 1:r){
      # read imputed dataset
      file.imputed <- paste(folder.imputed, dataset, '_h', layer, '_', w, '_', i, '.npy', sep = '')
      data.imputed <- np$load(file.imputed)
      data.imputed <- t(data.imputed); dim(data.imputed)
      
      # normalized root mse on masked value
      nrmse.mask.matrix[i, layer] <- sqrt(mean((data.imputed[mask] - norm[mask])^2)) / mean(norm[mask])
      cor.mask.matrix[i, layer] <- cor(data.imputed[mask], norm[mask])
    }
  }
  # compute NRMSE and correlation on masked value
  nrmse.mean <- round(apply(nrmse.mask.matrix, 2, mean), 4); nrmse.mean
  cor.mean <- round(apply(cor.mask.matrix, 2, mean), 4); cor.mean

  # save to matrix
  nrmse.cor.matrix[,j*2-1] <- nrmse.mean
  nrmse.cor.matrix[,j*2] <- cor.mean
}
})

write.table(nrmse.cor.matrix, 'mse_cor.txt', row.names = F, col.names = F)

##########################################################
# activation function
##########################################################
datasets <- c('jurkat','monocyte','mbrain','lymphoma','pbmc','293t','bmmc','human_mix','mouse_spleen',
              'mouse_cortex','mouse_skin','cbmc')

for(dataset in datasets){
  print(dataset)
  file.mask <- paste('masked dataset/', dataset, '_mask.mtx', sep = '')
  file.nomask <-paste('masked dataset/', dataset, '_nomask.mtx', sep = '')
  file.norm <- paste('masked dataset/', dataset, '_data.mtx', sep = '')
  np <- import("numpy")
  mask <- readMM(file.mask)
  nomask <- readMM(file.nomask)
  norm <- readMM(file.norm)
  mask <- as.matrix(mask)
  nomask <- as.matrix(nomask)
  norm <- as.matrix(norm)
  dim(mask); dim(nomask); dim(norm)
  # col: layer #; row: repeat #
  r <- 20   # repeat
  h <- 10   # depth
  w <- 32   # width
  af <- c('Sigmoid()','Tanh()','ReLU()','LeakyReLU(negative_slope=0.01)','LeakyReLU(negative_slope=0.2)',
          'ELU(alpha=1.0)','SELU()')
  nrmse.mask.matrix <- matrix(data = 0, ncol = length(af), nrow = r)
  cor.mask.matrix <- matrix(data = 0, ncol = length(af), nrow = r)
  system.time({
    for(a in 1:length(af)){
      for(i in 1:r){
        file.imputed <- paste('imputed dataset/', dataset, '_h', h, '_', af[a], '_', w, '_', i, '.npy', sep = '')
        data.imputed <- np$load(file.imputed)
        data.imputed <- t(data.imputed); dim(data.imputed)
        
        # normalized root mse and correlation on masked value
        nrmse.mask.matrix[i, a] <- sqrt(mean((data.imputed[mask] - norm[mask])^2)) / mean(norm[mask])
        cor.mask.matrix[i, a] <- cor(data.imputed[mask], norm[mask])
      }
    }
  })
  write.table(cor.mask.matrix, paste(dataset,'.txt', sep = ''), sep = '\t', col.names = F, row.names = F)
}

##########################################################
# weight decay
##########################################################
# user-specified dataset
dataset <- 'jurkat'
file.mask <- paste('masked dataset/', dataset, '_mask.mtx', sep = '')
file.nomask <-paste('masked dataset/', dataset, '_nomask.mtx', sep = '')
file.norm <- paste('masked dataset/', dataset, '_data.mtx', sep = '')
np <- import("numpy")
mask <- readMM(file.mask)
nomask <- readMM(file.nomask)
norm <- readMM(file.norm)
mask <- as.matrix(mask)
nomask <- as.matrix(nomask)
norm <- as.matrix(norm)
dim(mask); dim(nomask); dim(norm)

# col: layer #; row: repeat #
r <- 5
reg <- c('1e-07', '5e-07', '1e-06', '5e-06', '1e-05', '5e-05', '0.0001', '0.0005')
nrmse.mask.matrix <- matrix(data = 0, ncol = length(reg), nrow = r)
cor.mask.matrix <- matrix(data = 0, ncol = length(reg), nrow = r)
system.time({
  for(a in 1:length(reg)){
    for(i in 1:r){
      file.imputed <- paste('imputed dataset/', dataset, '_wd_', reg[a], '_', i, '.npy', sep = '')
      data.imputed <- np$load(file.imputed)
      data.imputed <- t(data.imputed); dim(data.imputed)
      
      # normalized root mse and correlation on masked value
      nrmse.mask.matrix[i, a] <- sqrt(mean((data.imputed[mask] - norm[mask])^2)) / mean(norm[mask])
      cor.mask.matrix[i, a] <- cor(data.imputed[mask], norm[mask])
    }
  }
})
nrmse.mean <- apply(nrmse.mask.matrix, 2, mean); nrmse.mean
cor.mean <- apply(cor.mask.matrix, 2, mean); cor.mean
write.table(cbind(nrmse.mean, cor.mean), 'af.txt', sep = '\t', col.names = F, row.names = F)

##########################################################
# dropout
##########################################################
datasets <- c('jurkat','monocyte','mbrain','lymphoma','pbmc','293t','bmmc','human_mix','mouse_spleen',
              'mouse_cortex','mouse_skin','cbmc')

for(dataset in datasets){
  print(dataset)
  file.mask <- paste('masked dataset/', dataset, '_mask.mtx', sep = '')
  file.nomask <-paste('masked dataset/', dataset, '_nomask.mtx', sep = '')
  file.norm <- paste('masked dataset/', dataset, '_data.mtx', sep = '')
  np <- import("numpy")
  mask <- readMM(file.mask)
  nomask <- readMM(file.nomask)
  norm <- readMM(file.norm)
  mask <- as.matrix(mask)
  nomask <- as.matrix(nomask)
  norm <- as.matrix(norm)
  dim(mask); dim(nomask); dim(norm)
  
  # col: layer #; row: repeat #
  r <- 5
  reg <- c('0.01', '0.02', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4')
  
  nrmse.mask.matrix <- matrix(data = 0, ncol = length(reg), nrow = r)
  cor.mask.matrix <- matrix(data = 0, ncol = length(reg), nrow = r)
  system.time({
    for(a in 1:length(reg)){
      for(i in 1:r){
        file.imputed <- paste('imputed dataset/', dataset, '_dp_', reg[a], '_', i, '.npy', sep = '')
        data.imputed <- np$load(file.imputed)
        data.imputed <- t(data.imputed); dim(data.imputed)
        # normalized root mse and correlation on masked value
        nrmse.mask.matrix[i, a] <- sqrt(mean((data.imputed[mask] - norm[mask])^2)) / mean(norm[mask])
        cor.mask.matrix[i, a] <- cor(data.imputed[mask], norm[mask])
      }
    }
  })
  nrmse.mean <- apply(nrmse.mask.matrix, 2, mean); nrmse.mean
  cor.mean <- apply(cor.mask.matrix, 2, mean); cor.mean
  write.table(cbind(nrmse.mean, cor.mean), paste(dataset, '.txt', sep = ''), sep = '\t', col.names = F, row.names = F)
}





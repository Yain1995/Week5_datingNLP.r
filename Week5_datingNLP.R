#NLP and clustering Assignment
#Big-Data course
#Okcupid
#Orel Maymon
#Yeynit Asraf
#Adiya Blum
#We can't run this script because lack of memory,
#so we have a link to the colab.

#Packages 
library(stringr)
library(dplyr)
library(ggplot2)
#install.packages("mosaic") #This takes time to download
library(dplyr)
library(stringr)
#install.packages("xtable")
library(xtable)
#install.packages("gridExtra")
library(gridExtra)
#install.packages("stopwords")
library(stopwords)
#install.packages("quanteda")
library(quanteda) #for texting process
#install.packages("caret")
library(caret)#For cross-validation
#install.packages("ggfortify") #For colur the pca graph
#install.packages("cluster")
#install.packages("factoextra") #For k-means algorithms
#install.packages("tidyverse") # For plot pca of k-means

# Set rounding to 2 digits
options(digits=2)

profiles <- read.csv('profiles.csv', header=TRUE, stringsAsFactors=FALSE)
profiles <- filter(profiles, height>=55 & height <=80)#To reduce variables
profiles<-profiles[1:35000,] 

#Pick 10 answers each participant
essays <- select(profiles, starts_with("essay"))
#Merge the answers to one corpus
#However, Merge only 5 answers, beacuse all the ten answers will cause crush in the train function below.
#essays$essay0<-paste(essays$essay0, essays$essay1, essays$essay2, essays$essay3,essays$essay4, essays$essay5, essays$essay6, essays$essay7, essays$essay8, essays$essay9)
essays$essay0<-paste(essays$essay0, essays$essay1, essays$essay2, essays$essay3,essays$essay4)
essays <- apply(essays, MARGIN = 1, FUN = paste, collapse=" ")

#Clean the data: make a dictionary  
html <- c( "<a[^>]+>", "class=[\"'][^\"']+[\"']", "&[a-z]+;", "\n", "\\n", "<br ?/>", "</[a-z]+ ?>" )
stop.words <-  c( "a", "am", "an", "and", "as", "at", "are", "be", "but", "can", "do", "for", "have", "i'm", "if", "in", "is", "it", "like", "love", "my", "of", "on", "or", "so", "that", "the", "to", "with", "you", "i" )
noisy_words<-stopwords::stopwords("english")
noisy_words<-append(noisy_words, html)
noisy_words<-append(noisy_words, stop.words )
essays <- str_replace_all(essays,noisy_words, " ")#Replace noisy words with ' '

# Tokenize essay texts
all.tokens <- tokens(essays, what = "word", remove_numbers = TRUE,
                     remove_punct = TRUE, remove_twitter= TRUE,
                     remove_symbols = TRUE, remove_hyphens = TRUE)
# Lower case the tokens.
all.tokens <- tokens_tolower(all.tokens)
all.tokens[[357]]
# Use quanteda's built-in stopword list for English.
all.tokens <- tokens_select(all.tokens, stopwords(),
                            selection = "remove")

# Perform stemming on the tokens.
all.tokens <- tokens_wordstem(all.tokens, language = "english")
# remove single-word tokens after stemming. Meaningless
all.tokens <- tokens_select(all.tokens, "^[a-z]$",
                            selection = "remove", valuetype = "regex")

# Create a bag-of-words model (document-term frequency matrix)
all.tokens.dfm <- dfm(all.tokens, tolower = FALSE)
rm(all.tokens) #is used to delete the object from the memory.
all.tokens.dfm[1:10]

#docs    love th nk nd ntellectu ther dumbest smart guy smartest
#text1    4 10  1  7         1    1       1     1   2        1
#text2    4  0  0  0         0    0       0     2   1        0
#text3    1  0  0  0         0    0       0     0   0        0
#text4    0  0  0  0         0    0       0     0   0        0
#text5    0  0  0  0         0    0       0     0   0        0
#text6    7  0  0  0         0    0       0     0   0        0

rm(essays,html,noisy_words,stop.words)

#Remove low frequent words in order to reduce variables 
dfm.trimmed <- dfm_trim(all.tokens.dfm, min_docfreq = 1200, min_termfreq = 1000, verbose = TRUE)
dim(dfm.trimmed)
# Removing features occurring: 
# fewer than 1000 times:  191,483
# in fewer than 600 documents: 120,221
# Total features removed: 120,2215 (98.3%).
rm(all.tokens.dfm) 
dfm.trimmed
# top-50 frequent features
topfeatures(dfm.trimmed, 50)

#Build TF-IDF
#docs-35,000

#We don't want to run this cell because it will take long time
#So in the next chunck we had a function that does all immediatly
#Calculate DF
tf<-function(row){
  row/(1+sum(row)) #vector's operator each cell in the row
}
#Calculate IDF
idf<-function(col){
  #docNum=35,000
  docNum<-length(col)
  docWithTerm<-length(which(col>0))
  log10(docNum/docWithTerm)
}

#Calculate IDF-DF
tfDotidf<-function(tf,idf){
  tf*idf #each cell
}

term_freq_mat<-apply(dfm.trimmed, MARGIN = 1,FUN=tf)#1 = row
idf_mat<-apply(dfm.trimmed, MARGIN=2, FUN=idf)#2 = column
tf_idf_mat<-apply(term_freq_mat, MARGIN =2, FUN =tfDotidf, idf=idf_mat)
tf_idf<-t(tf_idf_mat)
rm(tf_idf_mat)

#A function build TF-IDF matrix 
#tf_idf<-dfm_tfidf(dfm.trimmed)
gc()
rm(dfm.trimmed)

# Setup a the feature data frame with labels.
tf_idf <- as.data.frame(tf_idf, row.names = NULL, optional = FALSE,make.names=TRUE)
#convert(tf_idf, to = "data.frame")
#label=data.frame(profiles$sex)
l = profiles$sex
all.tokens.df <- cbind(Label = l, tf_idf)


#all.tokens.df <- cbind(data.frame(profiles$sex),tf_idf)
#gender<-all.tokens.df$profiles.sex[1:35000] #character type
rm(profiles)

# rectify the names of the variables column names.
names(all.tokens.df) <- make.names(names(all.tokens.df))

set.seed(23145)
#CV-We need to seperate for train and test
# Use caret to create stratified folds for 10-fold cross validation repeated 
# 3 times (i.e., create 30 random stratified samples)
#ten_folds <- createMultiFolds(all.tokens.df$Label, k = 10, times = 3)
#BUT WE DON"T NEED TO RUN THE FUNCTION ABOVE-because the train function below will do it!
control_group <- trainControl(method = "repeatedcv", number =10,
                              repeats = 3) 
#index = ten_folds)

#12 minutes to run in the colab.
#Use a single decision method- 'rpart' to train the model- a fast way.
#label ~ . - to predict by all the rest of the data.
# Measuer the time runnig
start.time <- Sys.time()
trained_model<- train(Label ~ ., data = all.tokens.df, method = "rpart", na.action=na.exclude, 
                      trControl = control_group, tuneLength = 3)

total.time <- Sys.time() - start.time
total.time
save(file='Week5_datingNLP.rdata', trained_model, tf_idf) 

print(total.time)
"Time difference of 11 mins"
trained_model
"CART 

35000 samples
  860 predictor
    2 classes: 'f', 'm' 

No pre-processing
Resampling: Cross-Validated (10 fold, repeated 3 times) 
Summary of sample sizes: 31500, 31500, 31499, 31500, 31500, 31501, ... 
Resampling results across tuning parameters:

  cp      Accuracy  Kappa
  0.0049  0.67      0.24 
  0.0114  0.66      0.23 
  0.0550  0.63      0.10 

Accuracy was used to select the optimal model using the largest value.
The final value used for the model was cp = 0.0049."

confusionMatrix(trained_model)
"Cross-Validated (10 fold, repeated 3 times) Confusion Matrix 

(entries are percentual average cell counts across resamples)
 
          Reference
Prediction    f    m
         f 13.9  8.2
         m 25.2 52.7
                            
 Accuracy (average) : 0.6665"

library(rpart)
#To test the confusion matrix:
confMat <- rpart(formula = Label~., data = all.tokens.df, y = TRUE)
confMat
"n= 35000 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

 1) root 35000 14000 m (0.39 0.61)  
   2) love>=0.0027 13123  6300 f (0.52 0.48)  
     4) guy< 0.0017 10244  4400 f (0.57 0.43)  
       8) girl>=0.0012 2117   540 f (0.75 0.25) *
       9) girl< 0.0012 8127  3800 f (0.53 0.47)  
        18) danc>=0.0015 2260   830 f (0.63 0.37) *
        19) danc< 0.0015 5867  2900 m (0.49 0.51)  
          38) famili>=0.00095 2644  1200 f (0.56 0.44) *
          39) famili< 0.00095 3223  1400 m (0.43 0.57) *
     5) guy>=0.0017 2879   950 m (0.33 0.67) *
   3) love< 0.0027 21877  6900 m (0.31 0.69) *"

#We think those words below are batch effect: identical words for men and women.
names(trained_model$finalModel$splits)
final_df<-(all.tokens.df)
final_df<-subset(final_df, select = -c(th,cuddl,organ,like,got,tom,goe,la))

library(ggfortify)

#Plot PCA
mat_pca<-data.matrix(final_df)
pca <- prcomp(mat_pca, scale=TRUE)
## plot pc1 and pc2
plot(pca$x[,1], pca$x[,2])
title('PC1 vs. PC2')
autoplot(pca, data = mat_pca, colour='blue')

## plot pc1 only
plot(pca$x[,1], rep(0,nrow(pca$x)),col='red')

abline(h=0, lty = 2)
title('PC1 only')

## make a scree (elbow) plot
pca.var <- pca$sdev^2
pca.var.per <- round(pca.var/sum(pca.var)*100, 1)
barplot(pca.var.per, main="Scree Plot", xlab="Principal Component",ylab="Percent Variation")

#K-Means
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
k2 <- kmeans(mat_pca, 2, nstart=25)
plot(mat_pca, col = k2$cluster)

#install.packages("tidyverse")

library(tidyverse)  # data manipulation
fviz_cluster(k2, data =mat_pca )

k3 <- kmeans(mat_pca, centers = 3, nstart = 25)
k4 <- kmeans(mat_pca, centers = 4, nstart = 25)
k10 <- kmeans(mat_pca, centers = 10, nstart = 25)

# plots to compare
p1 <- fviz_cluster(k2, geom = "point", data = mat_pca) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = mat_pca) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = mat_pca) + ggtitle("k = 4")
p4 <- fviz_cluster(k10, geom = "point",  data = mat_pca) + ggtitle("k = 10")

#library(gridExtra)
grid.arrange(p1, p2, p3, p4, nrow = 2)

save(file='Week5_datingNLP.rdata', trained_model, tf_idf, pca)
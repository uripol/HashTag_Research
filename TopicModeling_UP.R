#install.packages("lda")

#package ‘lda’ successfully unpacked and MD5 sums checked
#The downloaded binary packages are in
#C:\Users\C5243543\AppData\Local\Temp\RtmpyMpghU\downloaded_packages--moved to my folder
library(lda)

#source("C:\\Users\\user\\Dropbox\\Hashtags\\DataStudy\\SplitHashtag.R")
setwd("C:\\Users\\C5243543\\Desktop\\Reseacrch\\TopicModeling")


# load tweets
data=read.csv("SportPolitics.csv")
## https://github.com/vinaykola/twitter-topic-classifier/blob/master/training.txt
names(data)=c("id", "topic", "tweet")

tweets=data$tweet

#Phase 1: Prepare the words of each tweet

##################################################
### hashtags
#Replicates "" in the same length as the tweets, in order to put there all the "clean" words of the hashtags tweets
hashtagsarr=rep("", length(tweets))
for (t in c(1:length(tweets))){
	print(t)
	#searching if has up/low character or number,followed by - or _. + means to have it at lease one time, and insert all to one vector per each tweet
  hashtags=gregexpr("#[a-zA-Z0-9\\-\\_]+", tweets[t])
	if (unlist(hashtags)[1]==-1) {
		next
		ntweetswithnohashtag=ntweetswithnohashtag+1
	}
	hashtagslist=regmatches (tweets[t], hashtags)[[1]]
	#Reviewing all the phases that included #, per item in the hashtag list
	for (hashtag in hashtagslist){
		#why the need for splitHashtag and not only gsub???
	  words=splitHashTag(gsub("#", "", hashtag))
		#making sure to have the last # phase
	  hashtagsarr[t]=paste(words, collapse=" ")
	}
}
write.csv(hashtagsarr, "hashtagsarr.csv") 

#But how will know in the cluster phase- what tweet is from first type and what from the second(=several word at 1 #?)- unless it's becasue the space in the result we get
###############################################

#Phase 2- prepare the data before comparsion

hashtagsarr=read.csv("hashtagsarr.csv")



##################################################
### hashtags- why the second command is needed??
tweets=paste(tweets, hashtagsarr[,2], sep=" ")
tweets=hashtagsarr[,2]
###############################################


#Clean Text
cleantext<-function(tweets){
	tweets = gsub("(RT|via)((?:\\b\\W*@\\w+)+)"," RETWEET ",tweets)
	tweets = gsub("[ \t]{2,}", "", tweets)
	tweets = gsub("http[^[:blank:]]+", "http ", tweets)
	tweets = gsub("@\\w+", "REPLYTO ", tweets)
	tweets = gsub("^\\s+|\\s+$", "", tweets)
	tweets = gsub('\\d+', '', tweets)
	tweets = gsub("[[:punct:]]", " ", tweets)
	tweets = tolower(tweets)
	stopWords <- c("rt", "the", "on", "amp", "sign", "in" , "to", "a", "my",
               "or", "is", "are", "i", "an", "and", "of", "for", "with")
	stopWords <- tolower(stopWords)
	tweets <- tolower(tweets) 
	for (i in 1:length(stopWords)){
		tweets <- gsub(paste0('\\<',stopWords[i],'\\>'), "", tweets)
	return(tweets)
}
# create a vector of length/2 (places of) tweets which not null
createtraining<-function{tweets){
	s=sample(c(1:length(tweets)), length(tweets)/2)
	s=intersect(which(tweets!=""), s)
	return(s)
}
#Phase 3- Analazing..- by first "paste" topic to the sample tweets

computeRMSE<-function(){
  #Creates on the first function, 2 elements, based on the sample above:
  #documents: A list of document matrices in the format described in lda.collapsed.gibbs.sampler.
  #vocab: A character vector of unique tokens occurring in the corpus
  corpus1 = lexicalize(tweets[s], lower=TRUE)
	annotations = as.integer(data$topic)-1
num.topics=length(unique(data$topic))
params <- sample(c(-1, 1), num.topics, replace=TRUE)
## Only keep words that appear at least once:
to.keep <- corpus1$vocab[word.counts(corpus1$documents, corpus1$vocab) >= 1]
### Re-lexicalize, using this subsetted vocabulary- taking words only from words list that appeared more than once
documents <- lexicalize(tweets[s], lower=TRUE, vocab=to.keep)

## Initialize the params (why at all needed, and why needed again???)
params <- sample(c(-1, 1), num.topics, replace=TRUE)

#We assume it's Gibbs Sample, and model of LDA in order to predict the Y value of each observation. and it's exaplain why we "thin" the sample- beacuse we 
# collect words/tweets into documents, and assume that each document is a mix of all the relevant topics, and a new observation should have one of those topics.  
# on document- documents[[i]][1, j] is a 0-indexed word identifier for the jth word in document i. That is, this should be an index - 1 into vocab. documents[[i]][2, j] is an integer specifying the number of times that word appears in the document.
if (min(sapply(documents, length)) <= 0) {
  stop("Uriel,All documents must have positive length -bigger than 0")
}
X<- sapply(documents, length) > 0
#for (i in 1:length(documents)){
 # print(length(documents[i]))
}

  
#The function returns an estimated point of unknown parameter which used the paramaters bellow on last iteration 
result <- slda.em(documents=X,
                   K=num.topics,
                   vocab=to.keep,
                   num.e.iterations=10,
                   num.m.iterations=4,
                   alpha=1.0, eta=0.1,
                   as.integer(annotations[s]),
                   params,
                   variance=0.25,
                   lambda=1.0,
                   logistic=T,
                   method="sLDA")

## Make a pretty picture.
require("ggplot2")

Topics <- apply(top.topic.words(result$topics, 10, by.score=TRUE),
                 2, paste, collapse=" ")

predictions <- slda.predict(documents,
                             result$topics, 
                             result$model,
                             alpha = 1.0,
                             eta=0.1)
qplot(predictions,
       fill= data$topic[s],
       xlab = "predicted topic",
       ylab = "density",
       alpha=I(0.5),
       geom="density") +
   geom_vline(aes(xintercept=0))

predProbs<-cbind(0,predictions)
predInd<-t(apply(predProbs,1,which.max))
print(table(predInd,data$topic[s]))

library(ROCR)
performance(prediction(predictions, data$topic[-s]), "auc")
perf1 <- performance(prediction(predictions, data$topic[s]), "sens", "spec")
plot(perf1)



corpus2 = lexicalize(tweets[-s], lower=TRUE)
to.keep <- corpus1$vocab[word.counts(corpus2$documents, corpus1$vocab) >= 1]
documents <- lexicalize(tweets[-s], lower=TRUE, vocab=to.keep)

predictions <- slda.predict(documents,
                             result$topics, 
                             result$model,
                             alpha = 1.0,
                             eta=0.1)


qplot(predictions,
       fill= data$topic[-s],
       xlab = "predicted topic",
       ylab = "density",
       alpha=I(0.5),
       geom="density") +
   geom_vline(aes(xintercept=0)) 

predProbs<-cbind(0,predictions)
predInd<-t(apply(predProbs,1,which.max))
print(table(predInd,data$topic[-s]))

library(ROCR)
performance(prediction(predictions, data$topic[-s]), "auc")
perf1 <- performance(prediction(predictions, data$topic[-s]), "sens", "spec")
plot(perf1)



RMSE=sqrt(mean((predictions-annotations[-s])^2, na.rm=T))



# Week5_datingNLP.r
NLP + clustering Assignment: classifiaction of words by men or women from their answers to dating site Okupid.
Big-Data course

Orel Maymon, Yeynit Asraf, Adiya Blum 

#We run our job in google colab because lack of memory in R.
#And we minimaize the data because a lack of memory too in the colab.

data.file <- 'profiles.csv'
source('Week5_datingNLP.r')

# ———— confusion matrix ————
##               Reference
## prediction     f     m
##         f     13.9   8.9
##         m     25.2   52.7


# ———— training times ————
#Time difference of 11 mins

# ———— list of female words ————
#[1] "wine" "hair" "loving" "dancing," "love,"
#[6] "laughing" "harry" "smile." "heart" "water"
#[11] "laugh." "beautiful" "appreciate" "favorites" "dog"
#[16] "local" ";)" "east" "hate" "outside"
#[21] "30" "kids" "close" "laugh," "giving"
#[26] "passionate" "sushi," "person." "loved" "day."
#[31] "healthy" "chocolate" "hip" "types" "planning"
#[36] "able" "week" "change" "include" "green"
#[41] "fresh"


# ———— list of male words ————
#[1] "guy" "star" "video" "company" "cool" "/"
#[7] "science" "r" "computer" "that." "sports" "business" [13] "south" "daily" "well." "breaking" "stuff." "history"
#[19] "started" "hang" "blue" "well," "three" "generally" [25] "bike" "arrested" "couple" " " "run" "future"
#[31] "2" "classic" "profile" "away" "past" "type"
#[37] "world," "another" ""the" "until" "become"

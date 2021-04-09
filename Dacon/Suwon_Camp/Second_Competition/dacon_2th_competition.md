
# Data Description


```R
library(data.table) # Fast CSV read
library(tidyverse)    #Visualisation
library(DataExplorer)
library(knitr)
library(gridExtra)
library(scales)
library(xgboost) #Xgboost
library(lightgbm) #Lightgbm
```


```R
pick <- fread("../input/analyst_pick.csv", na.strings=c("NA","NaN", ""),encoding="UTF-8")
train <- fread("../input/toto_train.csv", na.strings=c("NA","NaN", ""),encoding="UTF-8")
test <- fread("../input/toto_test.csv", na.strings=c("NA","NaN", ""),encoding="UTF-8")
```


```R
cat("toto_train : (" , nrow(train) , "," , ncol(train) , ")\n")
cat("toto_test : (" , nrow(test) , "," , ncol(test) , ")\n")
cat("Analyst_Pick : (" , nrow(pick) , "," , ncol(pick) , ")")
```


```R
plot_missing(train)
plot_missing(test)
plot_missing(pick)
```


```R
colSums(is.na(train))
```


```R
colSums(is.na(test))
```

* Train,Test에서 최근 10경기 성적에 결측값이 있음

# Data Check


```R
train  %>% count(result)
```

* Target이 Imbalance하지 않습니다.

중복된 값이 있는지 데이터 확인


```R
# game_id와 date를 제외한 중복 데이터 확인 2315개
duplicated1 = train[duplicated(train[,3:8])]
```


```R
duplicated1  %>% arrange(home_team,win_percentage)  %>% head()
```


```R
# game_id와 date를 제외한 중복 데이터 확인 1350개
duplicated2 = test[duplicated(test[,3:7])]
```


```R
duplicated2  %>% arrange(home_team,win_percentage)  %>% head()
```


```R
# unique train data
new_train = train[!duplicated(train[,3:8])]
dim(new_train)
```


```R
# unique test data
new_test = test[!duplicated(test[,3:7])]
dim(new_test)
```


```R
match_test=left_join(new_test,duplicated2,by=c('win_percentage','home_team','away_team'))
```

* 데이터에서 하루 내지 이틀 간격으로 똑같은 경기들이 존재한다. 이는 중복되는 경기들끼리 똑같은 결과가 나와야 됨을 의미한다. 
  따라서 우리는 unique한 train과 test들로 모델을 훈련시키고 unique한 test와 unique 하지 않은 test를 결합한 
  match_test를 통하여  어떤 game_id들이 확인하고, 값을 채워넣도록 하겠다.

# Feature Engineering


```R
train= new_train  %>% mutate(sep="train")
test= new_test  %>% mutate(sep="test")
```


```R
full=bind_rows(train,test)
```


```R
full  %>% count(sep)
```


```R
full = full  %>% select(-c(home_team_recent_10,away_team_recent_10))
```

### Win/lose_percentage

* 배당과 확률을 통해서 승리에 건 확률과 패배에 배팅한 확률만 따로 뽑아보겠습니다.


```R
full=full  %>% 
        mutate(win_pick=str_extract(win_percentage,"(\\S*%)"))  %>% 
        mutate(win_pick= gsub( '[(%]','',win_pick))  %>% 
        mutate(win_pick= as.numeric(win_pick))

full=full  %>% 
        mutate(lose_pick=str_extract(win_percentage,"[^\\(]\\S*[\\)]$")) %>% 
        mutate(lose_pick= gsub( '[()%]','',lose_pick))  %>% 
        mutate(lose_pick= as.numeric(lose_pick))

full=full  %>% select(-win_percentage)
```


```R
head(full,10)
```


```R
full  %>% filter(sep=="train")  %>% 
    ggplot(aes(x=as.numeric(win_pick),colour=as.factor(result),group=as.factor(result)))+
    geom_density()+
    theme_light() + 
    labs(title = "Win Pick Density By Result", x = "win_pick")+
    theme(plot.title = element_text(hjust = .5))
```


```R
full  %>% filter(sep=="train")  %>% 
    ggplot(aes(x=as.numeric(lose_pick),colour=as.factor(result),group=as.factor(result)))+
    geom_density()+
    theme_light() + 
    labs(title = "Lose Pick Density By Result", x = "Lose Pick")+
    theme(plot.title = element_text(hjust = .5))
```

### date


```R
library(lubridate)

full<-full%>%
      mutate(
          date = ymd(date),
          year = as.factor(lubridate::year(date))
          )
```

* Test의 데이터를 먼저 살펴보겠습니다


```R
full  %>% filter(sep=="test")  %>% count(year)
```

* 2019년 1월~5월의 데이터로만 구성되어 있는 것을 확인할 수 있습니다. 그럼 Train은 어떻게 구성되어 있는지 시각화를 해보겠습니다.


```R
full  %>% filter(sep=="train")  %>% 
    ggplot(aes(x=as.factor(year)))+
    geom_bar(fill = "blue",alpha=0.3)+
    geom_text(aes(label =scales::percent(..count../sum(..count..))),stat = 'count',vjust = -0.5)+
    theme_light() + 
    labs(title = "Count of year", x = "Years")+
    theme(plot.title = element_text(hjust = .5))
```

* 2017년과 2018년 자료가 55%이상으로 구성되있습니다.

### Analyst_Pick 


```R
head(pick)
```


```R
# game_id 별 예측평균
pick= pick  %>% 
        select(-analyst) %>% 
        group_by(game_id) %>% 
        summarise_all(mean,na.rm=T)
```


```R
head(pick)
```


```R
cat("Unique game_id count of Anlaysit pick is : (",length(unique(pick$game_id)), ")\n")
cat("Unique game_id count of Full Data is : (", length(unique(full$game_id)), ")\n")
cat("Intersect game_id count : (" , length(intersect(pick$game_id,full$game_id)) , ")")
```


```R
full= left_join(full,pick,by="game_id")
```


```R
full  %>% filter(sep=="train")  %>% 
    ggplot(aes(x=choosed,colour=as.factor(result),group=as.factor(result)))+
    geom_density()
```

* 실제 경기가 이겼을때 pick의 차이가 엄청 크다는 것을 알 수 있습니다.

# Correlation


```R
str(full)
```


```R
# 형식 변환
full= full  %>% 
        mutate_if(is.integer, list(~as.numeric(.)))  %>% 
        mutate_if(is.character, list(~as.factor(.)))        
```


```R
corr_data = full  %>%
            filter(sep=="train")  %>%
            select(c(result,win_pick,lose_pick,choosed))
            
result = cor(corr_data, use = "pairwise")
result
```


```R
library(reshape2)

melted_cormat <- melt(result,na.rm=TRUE)
head(melted_cormat)
```


```R
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Pearson\nCorrelation") +
  labs(title='Correlation Heatmap')+
  theme(plot.title = element_text(hjust = .5), axis.text.x = element_text(angle=90),
       axis.title.x=element_blank(),
       axis.title.y=element_blank())+
  coord_fixed()+
  geom_text(aes(Var2, Var1, label = round(value,2)), color = "black", size = 3)
```

* 확실히 다른 변수들보다 win_pick(배당 승리 확률),choosed(전문가 확률)가 result에 영향을 끼치는 것을 알 수 있습니다.

  우리는 좀 더 영향을 크게 주기 위해 년도별/팀별 최근 pick 평균을 넣어보겠습니다.


```R
#home_team 별 mean
full2= full  %>% 
        group_by(year,home_team) %>% 
        select_if(is.numeric)  %>% 
        summarise_all(mean,na.rm=T) %>% 
        ungroup() %>% 
        select(year,home_team,win_pick,lose_pick)
```


```R
full2= full2  %>% 
        rename("home_win_pick"=win_pick,
               "home_lose_pick"=lose_pick)
```


```R
#away_team 별 mean
full3= full  %>% 
        group_by(year,away_team) %>% 
        select_if(is.numeric)  %>% 
        summarise_all(mean,na.rm=T) %>% 
        ungroup() %>% 
        select(year,away_team,win_pick,lose_pick)

full3= full3  %>% 
        rename("away_win_pick"=win_pick,
               "away_lose_pick"=lose_pick)
```


```R
# Merge with Full data
full=left_join(full,full2,by=c("year","home_team"))
full=left_join(full,full3,by=c("year","away_team"))
```


```R
corr_data = full  %>%
            filter(sep=="train")  %>%
            select(c(result,home_win_pick,home_lose_pick,away_win_pick,away_lose_pick))
            
result = cor(corr_data, use = "pairwise")
```


```R
melted_cormat <- melt(result,na.rm=TRUE)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Pearson\nCorrelation") +
  labs(title='Correlation Heatmap')+
  theme(plot.title = element_text(hjust = .5), axis.text.x = element_text(angle=90),
       axis.title.x=element_blank(),
       axis.title.y=element_blank())+
  coord_fixed()+
  geom_text(aes(Var2, Var1, label = round(value,2)), color = "black", size =3)
```

# Missing Values


```R
colSums(is.na(full))
```


```R
full = full  %>% mutate(choosed=ifelse(is.na(choosed),0.5,choosed))
```

* 전문가 확률은 0.5로 결측치를 넣어주었습니다.

# Data for Modeling


```R
#모델링에 쓰지 않는 변수 제거
full= full  %>% select(-c(game_id,date,year))
```


```R
glimpse(full)
```


```R
train= full  %>% 
        filter(sep=="train")  %>% select(-c(sep,result))

test= full  %>% 
        filter(sep=="test")%>% select(-c(sep,result))
```


```R
dim(train)

dim(test)
```


```R
train_label= full  %>% 
            filter(sep=="train")  %>% select(result)  %>% as.matrix

test_label= full  %>% 
            filter(sep=="test")  %>% select(result)  %>% as.matrix
```

# XG boost


```R
x_train<- model.matrix(~.-1, data = train) %>% data.frame

x_test <- model.matrix(~.-1, data = test)  %>% data.frame
```


```R
dtrain <- xgb.DMatrix(data = as.matrix(x_train), label=train_label)
dtest <- xgb.DMatrix(data = as.matrix(x_test), label=test_label)
```


```R
param <- list(objective   = "binary:logistic",
              eval_metric = "auc",
              max_depth   = 4,
              eta         = 0.01,
              alpha       = 0.5,
              subsamle    = 0.5,
              colsample_bytree = 0.5)
```


```R
# set.seed(616)
# xgb_cv <- xgb.cv(params  = param,
#               data    = dtrain,
#               nrounds =1000,
#               nfold   = 5,
#               nthread = -1,
#               early_stopping_rounds = 100,
#               verbose = 0)
```


```R
# auc=xgb_cv$evaluation_log
# auc  %>% filter(test_auc_mean==max(auc[,4]))
```


```R
set.seed(616)
xgb <- xgb.train(params  = param,
              data    = dtrain,
              nrounds = 50,
              print_every_n = 10,
              verbose = 0
                )
```


```R
XGB_pred <- predict(xgb, dtest)
```


```R
summary(XGB_pred)
```


```R
xgb.importance(colnames(dtrain), model = xgb) %>% 
  xgb.plot.importance(top_n = 10)
```


```R
new_test= new_test  %>%
            select(game_id)  %>% 
            mutate(result=XGB_pred)
```


```R
match_test= match_test  %>% select(game_id.x,game_id.y)    %>% rename(game_id=game_id.x)
```


```R
last=left_join(match_test,new_test,by='game_id')
```


```R
df1= last  %>% select(game_id,result)
df2= last  %>% select(game_id.y,result)  %>% rename(game_id=game_id.y)

df1=unique(df1)
df2=df2[complete.cases(df2),]
```


```R
df=bind_rows(df1,df2)
```


```R
submission_xgb <- read.csv('../input/submission.csv')
submission_xgb = left_join(submission_xgb,df,by="game_id")  %>% select(-result.x)  %>% rename(result=result.y)
write.csv(submission_xgb, file='submission_xgb.csv', row.names = F)
```

# Random Forest


```R
train= full  %>% 
        filter(sep=="train")  %>% select(-c(sep))

train$result=as.factor(train$result)
```


```R
library(caret)
library(ranger)

set.seed(616)

# myControl <- trainControl(
#   method = "cv", number = 5
# )

# rf_grid=expand.grid(mtry=3:5,
#                    min.node.size=1,
#                    splitrule="extratrees")

# model_rf <- train(result ~., data = train, method='ranger',trControl = myControl,tuneGrid=rf_grid)
```


```R
set.seed(616)

fit.rf <- ranger(train_label~., data=x_train, min.node.size=1,importance= c("impurity"),  splitrule="extratrees")

rf_pred <- predict(fit.rf, x_test)
```


```R
fit.rf
```


```R
rf_pred <- rf_pred$predictions
```


```R
head(rf_pred)
```


```R
new_test= new_test  %>%
            mutate(result=rf_pred)
```


```R
last=left_join(match_test,new_test,by='game_id')
```


```R
df1= last  %>% select(game_id,result)
df2= last  %>% select(game_id.y,result)  %>% rename(game_id=game_id.y)

df1=unique(df1)
df2=df2[complete.cases(df2),]

df=bind_rows(df1,df2)
```


```R
submission_rf <- read.csv('../input/submission.csv')
submission_rf = left_join(submission_rf,df,by="game_id")  %>% select(-result.x)  %>% rename(result=result.y)
write.csv(submission_xgb, file='submission_rf.csv', row.names = F)
```

# H2o


```R
train= full  %>% 
        filter(sep=="train")  %>% select(-c(sep))

test= full  %>% 
        filter(sep=="test")%>% select(-c(sep,result))
```


```R
library(h2o)
```


```R
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '6g', nthreads = -1)
h2o.train <- as.h2o(train)
h2o.test <-  as.h2o(test)
```


```R
y <- "result"
x <- setdiff(names(h2o.train), y)
```


```R
logit_fit <- h2o.glm(x,y,training_frame =h2o.train,family = 'binomial')
```


```R
logit_fit
```


```R
glm.pred <- h2o.predict(logit_fit, h2o.test)
```


```R
head(glm.pred)
```


```R
glm.pred=as.data.frame(glm.pred)
```


```R
new_test= new_test  %>%
            mutate(result=glm.pred$p1)

last=left_join(match_test,new_test,by='game_id')

df1= last  %>% select(game_id,result)
df2= last  %>% select(game_id.y,result)  %>% rename(game_id=game_id.y)

df1=unique(df1)
df2=df2[complete.cases(df2),]

df=bind_rows(df1,df2)
```


```R
submission_glm <- read.csv('../input/submission.csv')
submission_glm = left_join(submission_glm,df,by="game_id")  %>% select(-result.x)  %>% rename(result=result.y)
write.csv(submission_glm, file='submission_glm.csv', row.names = F)
```

# Blending


```R
blending=bind_cols(submission_xgb,submission_rf,submission_glm)

blending= blending   %>% select(-c(game_id1,game_id2))
```


```R
head(blending,10)
```


```R
graph=melt(id=1,blending)
```


```R
ggplot(graph,aes(x=value, col=variable)) +
    geom_density(alpha=0.25)+
    scale_color_discrete(labels=c("XG boost","Random Forest","Logistic Regression"))
```


```R
blending=blending %>% 
        mutate(sum= (result1+result2)/2)
```


```R
submission_blending <- read.csv('../input/submission.csv')
submission_blending$result <- blending$sum
write.csv(submission_blending, file='submission_blending.csv', row.names = F)
```

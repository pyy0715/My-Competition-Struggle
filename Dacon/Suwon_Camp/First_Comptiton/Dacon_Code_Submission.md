
# 타이타닉 분석

## 1. 준비작업
### 1.1 Packages


```R
library(readr)
library(stringr) # 문자열 처리 패키지
library(doBy)
library(ggplot2)
library(scales)
library(RColorBrewer)
library(corrplot)
library(doBy)
library(dplyr) # 전처리
library(randomForest)
library(gridExtra)
library(DataExplorer)
library(tidyverse)
library(lightgbm)
library(xgboost)
library(ranger)
library(caret)
```

### 1.2 Loading the data


```R
train <- read_csv('../input/train.csv')
test <- read_csv('../input/test.csv')
full <- bind_rows(train, test)

full <- full %>% # ticket과 cabin은 파생변수 생성을 위해 문자열로 놔둠
  mutate(Survived = factor(Survived),
         Pclass   = factor(Pclass, ordered = T),
         Name     = factor(Name),
         Sex      = factor(Sex),
         Embarked = factor(Embarked))

str(full)
```

# 2. 탐색적 데이터 분석(EDA)

## 2.1 수치값을 활용한 data 확인


```R
head(full)
```


```R
str(full)
```


```R
summary(full)
```

1. 사망자가 생존자보다 많다 <br />
2. 남성이 여성보다 2배 가까이 더 많다 <br />
3. SibSp의 3분위값이 1이므로 대부분 부부끼리 혹은 형제끼리 탑승했다 <br />
4. Parch의 3분위수가 0이므로 부모나 자녀와 함께 탑승한 승객이 많지 않다 <br />
5. Fare의 최대값이 512로 이상치가 아닌지 확인이 필요해 보인다 <br />
6. 결측치가 많은 데이터임을 확인할 수있다. <br />


```R
sapply(train, function(x) length(unique(x)))
```

## 2.2 결측치 확인 및 시각화


```R
plot_missing(full)
```


```R
colSums(is.na(full))
```

## 2.3 변수 EDA

### Sex


```R
table(full$Sex)
```


```R
full %>% group_by(Survived, Sex) %>% count()
```


```R
prop.table(table(full$Sex,full$Survived),1) #여자들이 생존할 확률이 높음
```


```R
# 성별 막대그래프
sex.p1 <- full %>% 
  dplyr::group_by(Sex) %>% 
  summarize(N = n()) %>% 
  ggplot(aes(Sex, N)) +
  geom_col() +
  geom_text(aes(label = N), size = 5, vjust = 1.2, color = "#FFFFFF") + 
  ggtitle("Bar plot of Sex") +
  labs(x = "Sex", y = "Count")

# 성별에 따른 생존률 막대그래프
sex.p2 <- full%>%
  filter(!is.na(Survived)) %>%
  ggplot(aes(factor(Sex), fill = factor(Survived))) +
  geom_bar(position = "fill") + 
  scale_y_continuous(labels = percent) +
  scale_fill_brewer(palette = "Set1") +  # palette에 어떤색 넣을지 지정
   # 일정한 간격으로 x축과 y축 설정 : scale_x_continuous(breaks=seq())
  # 분석가 마음대로 x축과 y축 설정 : scale_x_continuous(breaks=c())
  ggtitle("Survival Rate by Sex") + 
  labs(x = "Sex", y = "Rate")

grid.arrange(sex.p1,sex.p2,ncol=2)
```

### Pclass


```R
table(full$Pclass)
```


```R
prop.table(table(full$Pclass,full$Survived),1) # 더 좋은 객실 이용자일수록 생존할 확률이 높음
```


```R
# Pclass 막대그래프
pclass.p1 <- full %>% 
  dplyr::group_by(Pclass) %>% 
  summarize(N = n()) %>% 
  ggplot(aes(Pclass, N)) +
  geom_col() +
  geom_text(aes(label = N), size = 5, vjust = 1.2, color = "#FFFFFF") + 
  ggtitle("Bar plot of Pclass") +
  labs(x = "Pclass", y = "Count")

# Pclass에 따른 생존률 막대그래프
pclass.p2 <- full%>%
  filter(!is.na(Survived)) %>%
  ggplot(aes(factor(Pclass), fill = factor(Survived))) +
  geom_bar(position = "fill") + 
  scale_fill_brewer(palette = "Set1") +  
  ggtitle("Survival Rate by Pclass") + 
  labs(x = "Pclass", y = "Rate")

grid.arrange(pclass.p1,pclass.p2,ncol=2)
```

### Fare


```R
# fare 히스토그램
Fare.p1 <- full %>%
  ggplot(aes(Fare)) + 
  geom_histogram(col    = "yellow",
                 fill   = "blue", 
                 alpha  = .5) +
  ggtitle("Histogram of passengers Fare") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 15))

# 생존여부에 따른 fare box plot
Fare.p2 <- full %>%
  filter(!is.na(Survived)) %>% 
  ggplot(aes(Survived, Fare)) +  # x축에 생존 y축에 fare
  # 관측치를 회색점으로 찍되, 중복되는 부분은 퍼지게 그려줍니다.
  geom_jitter(col = "gray") + 
  # 상자그림 : 투명도 50% 
  geom_boxplot(alpha = .5) + 
  ggtitle("Boxplot of passengers Fare") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 15))

grid.arrange(Fare.p1,Fare.p2,ncol=2)
```

### Age


```R
# 나이 분포 히스토그램
age.p1 <- full %>% 
  ggplot(aes(Age)) +     # x값에 따른 y값을 그리는 것이 아니므로 축 지정 안해줘도 됨 
  # 히스토그램 그리기, 설정
  geom_histogram(breaks = seq(0, 80, by = 1), # 간격 설정 
                 col    = "red",              # 막대 경계선 색깔 
                 fill   = "green",            # 막대 내부 색깔 
                 alpha  = .5) +               # 막대 투명도 = 50% 
  # Plot title
  ggtitle("All Titanic passengers age hitogram") +
  theme(plot.title = element_text(face = "bold",    # 글씨체 
                                  hjust = 0.5,      # Horizon(가로비율) = 0.5
                                  size = 15, color = "darkblue"))

# 나이에 따른 생존 분포 파악
age.p2 <- full %>% 
  filter(!is.na(Survived)) %>%
  ggplot(aes(Age, fill = Survived)) + 
  geom_density(alpha = .5) +   # 막대그래프가 아니고 밀도그래프니까 plot으로 축 지정하고 geom_bar 대신에 geom_density
  ggtitle("Titanic passengers age density plot") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5,
                                  size = 15, color = "darkblue"))

grid.arrange(age.p1,age.p2,ncol=2)
```

### Sibsp


```R
table(full$SibSp)
```


```R
train %>% group_by(Survived, SibSp) %>% summarise(freq = n())
```


```R
prop.table(table(train$SibSp,train$Survived),1) #배우자,형제자매가 많을수록 생존률이 떨어짐
```

### Parch


```R
table(train$Parch)
```


```R
train %>% group_by(Survived, Parch) %>% summarise(freq = n())
```


```R
prop.table(table(train$Parch,train$Survived),1) #부모와 자녀를 1~3명 정도 동승했을 경우 생존률이 높음 
```

### Embarked


```R
table(train$Embarked) #결측값 2개
```


```R
train %>% group_by(Survived, Embarked) %>% summarise(freq = n())
```


```R
prop.table(table(train$Embarked,train$Survived),1) # C에서 탑승한 인원들만 생존률이 더 높다
```

# 3. 결측치 처리

EDA 과정에서 결측치가 Cabin에 1014개, Age에 263개, Embarked 2개, Fare에 1개 존재한다는 것을 확인했다<br />
Cabin은 결측치 수가 너무 많아서 그냥 변수를 제거하고, 차후 파생병수 Deck을 생성할 것이다<br />
따라서 Cabin 변수를 제외한 나머지 변수들에 대한 결측치 처리를 수행하려 한다<br />

Age 결측처리는 feature engineering 과정에서 생성한 파생변수 title을 이용하여 처리하기 위해 Age feature engineering 과정에서 결측처리를 병행하도록 한다<br />

따라서 해당 절에서는 Embarked 와 Fare의 결측처리를 우선 수행한다


```R
colSums(is.na(full))
```

## 3.1 Embarked 결측처리


```R
full[is.na(full$Embarked), ] #두개의 관측치 모두 Fare가 80이고, Pclass가 1임
```


```R
embark_fare <- full[!is.na(full$Embarked), ]
```


```R
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), # fare가 80에 line 생성
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous()
```

fare가 80면서 Pclass가 1인 승객들 대다수는 Embark가 C다


```R
full$Embarked[c(62, 830)] <- 'C'
full[c(62, 830),] 
```

## 3.2 Fare 결측처리


```R
full  %>% filter(is.na(full$Fare)) #Pclasss가 3이고, Embarked는 S임
```


```R
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE) #중앙값으로 결측치 처리
full[1044,]
```

# 4. Feature engineering

## 4.1 Name

Name에서 성별과 관련된 이름만을 추출하고 범주화해서 Title 파생변수를 생성한다


```R
Title <- full$Name
Title <- gsub("^.*, (.*?)\\..*$", "\\1", Title) # 정규표현식
full$Title <- Title
unique(full$Title)
```

이 title이라는 파생변수를 그대로 사용할 경우 모델의(특히 Tree based model) 복잡도가 상당히 높아지기 때문에 범주를 줄여줘야한다.
그 전에 descr패키지를 이용해서 각 범주별 빈도수와 비율을 확인해보겠다


```R
# 범주별 빈도수, 비율 확인 
descr::CrossTable(full$Title)
```


```R
# 5개 범주로 단순화 시키는 작업 
full <- full %>%
  # "%in%" 대신 "=="을 사용하게되면 Recyling Rule 때문에 원하는대로 되지 않습니다.
  mutate(Title = ifelse(Title %in% c("Mlle", "Ms", "Lady", "Dona"), "Miss", Title), # %in% 개념
         Title = ifelse(Title == "Mme", "Mrs", Title),
         Title = ifelse(Title %in% c("Capt", "Col", "Major", "Dr", "Rev", "Don",
                                     "Sir", "the Countess", "Jonkheer"), "Officer", Title),
         Title = factor(Title))
```


```R
# 파생변수 생성 후 각 범주별 빈도수, 비율 확인 
descr::CrossTable(full$Title) # 5개의 범주로 축소
```

## 4.2 Sex

성별을 더미화한다


```R
full$Sex <- ifelse(full$Sex == "male" ,0 , 1)
full$Sex <- as.factor(full$Sex)
```

## 4.3 Fsize

Sibsp와 Parch를 이용하여 Fsize 파생변수를 생성한다


```R
full$Fsize <- full$SibSp + full$Parch + 1
table(full$Fsize)
```


```R
# Fsize에 따른 생존율 시각화
Fsize.p1 <- full%>%
  filter(!is.na(Survived)) %>%
  ggplot(aes(Fsize, fill = Survived)) +
  geom_bar(position = "fill") + 
  scale_y_continuous(labels = percent) +
  scale_x_continuous(breaks=c(1:11)) +
  scale_fill_brewer(palette = "Set1") +  # palette에 어떤색 넣을지 지정
  # 일정한 간격으로 x축과 y축 설정 : scale_x_continuous(breaks=seq())
  # 분석가 마음대로 x축과 y축 설정 : scale_x_continuous(breaks=c())
  ggtitle("Survival Rate by Fsize") + 
  labs(x = "Fsize", y = "Rate")

Fsize.p1



#ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
#  geom_bar(stat='count', position='fill') +   #position = 'dodge', 'fill' 구분
#  scale_x_continuous(breaks=c(1:11)) +
#  labs(x = 'Family Size', y = 'Rate') 

```


```R
# 범주화
full$Familysize[full$Fsize == 1] <- 'single'
full$Familysize[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$Familysize[full$Fsize > 4] <- 'large'

full$Familysize <- as.factor(full$Familysize)
table(full$Familysize)
```


```R
# 범주화 후 Familiysize에 따른 생존율 시각화
ggplot(full[1:891,], aes(x = Familysize, fill = Survived)) +
  geom_bar(position = 'fill') +
  ggtitle("Survival Rate by Familysize")+
  labs(x="Familysize", y="Rate")
```

## 4.4 Cabin


```R
full$Cabin[1:28]
```


```R
strsplit(full$Cabin[2], NULL)[[1]]
```


```R
full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))
full$Deck=as.character(full$Deck)
```


```R
#Cabin 변수 제거
full=full[,-11]
```


```R
full$Deck[is.na(full$Deck)] <- "U"

cabin=full %>%filter(!is.na(full$Survived)& full$Deck!='U')

ggplot(cabin,aes(x=Deck, fill=factor(Survived), na.rm=TRUE)) +
        geom_bar(stat='count') +
        facet_grid(.~Pclass) +
        labs(title="Survivor split by Pclass and Deck")
```

* Pclass와 Deck을 살펴보면 ABC는 1등급만, DEF는 2등급만 , G는 3등급만 있음
* 따라서 pclass 가 1일떄 deck은 x, 2일떄 y, 3일떄 z


```R
full=full  %>% 
    mutate(Deck= ifelse(Pclass==1 & Deck=="U","X",
                        ifelse(Pclass==2 & Deck=="U","Y",
                               ifelse(Pclass==3 & Deck=="U","Z",Deck)))
          )
```


```R
full   %>% group_by(Pclass) %>% count(Deck)
```

## 4.5 Age

Age의 결측처리와 변수 가공을 수행한다

#### Sex에 따른 Age 탐색


```R
age.sex <- full %>% 
  ggplot(aes(Age, fill = Sex)) + 
  geom_density(alpha = .5) +  
  ggtitle("Titanic passengers Age density plot") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5,
                                  size = 15, color = "darkblue"))
age.sex
```

#### Pclass에 따른 Age 탐색


```R
age.pclass <- full %>% 
  ggplot(aes(Age, fill = Pclass)) + 
  geom_density(alpha = .5) +  
  ggtitle("Titanic passengers Age density plot") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5,
                                  size = 15, color = "darkblue"))
age.pclass
```

#### Title에 따른 Age 탐색


```R
age.title <- full %>% 
  ggplot(aes(Age, fill = Title)) + 
  geom_density(alpha = .5) +  
  ggtitle("Titanic passengers Age density plot") + 
  theme(plot.title = element_text(face = "bold", hjust = 0.5,
                                  size = 15, color = "darkblue"))
age.title
```

title에 따른 결측처리 방법을 선택한다
각 분포가 정규분포라고 보기 힘드므로 중앙값을 사용한다


```R
# title별 Median Age를 통한 결측값 처리
full=as.data.frame(full)
summaryBy(Age ~ Title, data=full, FUN=c(mean, sd, median), na.rm=TRUE) ## ddply로도
```


```R
full$Age <- ifelse((is.na(full$Age) & full$Title == 'Master'), 4, full$Age)
full$Age <- ifelse((is.na(full$Age) & full$Title == 'Miss'), 22, full$Age)
full$Age <- ifelse((is.na(full$Age) & full$Title == 'Mr'), 29, full$Age)
full$Age <- ifelse((is.na(full$Age) & full$Title == 'Mrs'), 35, full$Age)
full$Age <- ifelse((is.na(full$Age) & full$Title == 'Officer'), 48, full$Age)
```

###  Age 변수 가공


```R
hist(full$Age, freq=F, main='Age',col='lightgreen', ylim=c(0,0.05))

# child : 18세 이하
# adult : 19세 이상 64세 이하
# senior : 65세 이상

full$Age <- ifelse(full$Age <= 18, "child",
                   ifelse(full$Age > 18 & full$Age <= 64, "adult","senior"))
```

## 4.6 Ticket

Ticket 변수를 이용하여 GroupSize 파생변수를 생성한다


```R
length(unique(full$Ticket))
```


```R
head(full$Ticket)
```


```R
full  %>%  arrange(Ticket)  %>% head() #같은 티켓인데도 불구하고 Family가 single, 친구등과 같이 온것으로 유추
```


```R
full$TravelGroup <- NA
```


```R
full <- (transform(full, TravelGroup = match(Ticket, unique(Ticket))))
```


```R
full <- full %>% 
            group_by(TravelGroup) %>% 
            mutate(GroupSize = n()) %>%
            ungroup()
```


```R
full  %>% filter(Deck=="X") %>% filter(GroupSize>=2)  %>% dim()
```


```R
full  %>% filter(Deck=="Y") %>% filter(GroupSize>=2)  %>% dim()
```


```R
library(reshape2)
```


```R
a=dcast(full,Ticket+GroupSize~Deck)
```


```R
a= a  %>% mutate(K=X+Y+Z)
```


```R
a  %>% filter(K!=0)  %>% filter(GroupSize!=K)
```


```R
table(full$GroupSize)
```

# 5. Predict

## 5.1 변수선택
Pclass, Sex, Age, Fare, Embarked, Title, Familysize, GroupSize, Deck


```R
str(full)
```


```R
#범주화 안된 변수들 범주화 처리
factor_vars <- c('Age','Deck','GroupSize')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))
  
                            
full$GroupSize=as.numeric(full$GroupSize)
# #Fare log변환
# full$Fare=log1p(full$Fare)
```


```R
full=full  %>%  select(-c(1,4,7,8,9,13,16))
str(full)
```


```R
train <-full  %>% filter(is.na(Survived)==FALSE)
test <-full  %>% filter(is.na(Survived)==TRUE)  %>% select(-Survived)
```


```R
invisible(gc)
```

# H2o


```R
library(h2o)
```


```R
# 순서형 factor는 인식못함
train= train %>% mutate(Pclass= factor(Pclass,ordered=F))
test= test %>% mutate(Pclass= factor(Pclass,ordered=F))
```


```R
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '6g', nthreads = -1)
h2o.train <- as.h2o(train)
h2o.test <-  as.h2o(test)
```


```R
y <- "Survived"
x <- setdiff(names(h2o.train), y)
```


```R
aml <- h2o.automl(x = x, y = y,
                  training_frame = h2o.train,
                  max_models = 20,
                  seed = 42)
```


```R
lb <- aml@leaderboard
print(lb, n = nrow(lb))
```


```R
plot(best_dl, timestep = "number_of_trees", metric = "auc")
```


```R
h2o.pred <- h2o.predict(aml@leader, h2o.test)
```


```R
head(h2o.pred)
```


```R
h2o.pred=as.data.frame(h2o.pred)
```


```R
submission_h2o <- read.csv('../input/sample_submission.csv')
submission_h2o$Survived <- h2o.pred$p1
write.csv(submission_h2o, file='submission_h2o.csv', row.names = F)
```

# logistic Regrssion


```R
logit_fit <- h2o.glm(x,y,training_frame =h2o.train,family = 'binomial')
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
# library(MlBayesOpt)
# set.seed(42)
# res=xgb_cv_opt(data = train, 
#         label = Survived,
#         objectfun = "binary:logistic",
#         evalmetric = "auc",
#         eta_range = c(0.01, 0.1L),
#         max_depth_range = c(4L, 6L),
#         nrounds_range = c(100, 2000L),
#         subsample_range = c(0.5, 1L),
#         bytree_range = c(0.5, 1L),
#         init_points = 4,
#         n_iter = 10,
#         n_folds = 5,
#         acq = "ucb",
#         eps = 0)
```


```R
submission_log <- read.csv('../input/sample_submission.csv')
submission_log$Survived <- glm.pred$p1
write.csv(submission_log, file='submission_log.csv', row.names = F)
```

# Random Forest


```R
train <-full  %>% filter(is.na(Survived)==FALSE)
test <-full  %>% filter(is.na(Survived)==TRUE)  %>% select(-Survived)
```


```R
set.seed(42)
myControl <- trainControl(
  method = "cv", number = 5
)

rf_grid=expand.grid(mtry=2:8,
                   min.node.size=c(1,3,5),
                   splitrule="extratrees")

model_rf <- train(Survived ~., data = train, method='ranger',trControl = myControl,tuneGrid=rf_grid)
```


```R
model_rf
```


```R
p <- predict(model_rf,train)

confusionMatrix(p,train$Survived)
```


```R
train <-full  %>% filter(is.na(Survived)==FALSE)
test <-full  %>% filter(is.na(Survived)==TRUE)

train_label <- as.numeric(train$Survived)-1
test_label <- test$Survived


x_train<- model.matrix(~.-1, data = train[,-1]) %>% data.frame

x_test <- model.matrix(~.-1, data = test[,-1]) %>% data.frame
```


```R
set.seed(42)
fit.rf <- ranger(train_label~., data=x_train, mtry=8, min.node.size=1,importance= c("impurity"),  splitrule="extratrees")

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
imp=fit.rf$variable.importance
imp=as.data.frame(imp) %>% 
    rownames_to_column()  %>% 
    mutate(rowname = forcats::fct_reorder(rowname,imp))  %>% arrange(desc(imp))


ggplot(imp)+
    geom_col(aes(x = rowname, y = imp))+
    coord_flip()+
    theme_bw()+
    theme_light()+
    theme(plot.title = element_text(hjust = 0.5)) +
    ggtitle("Variable Impotance")
```


```R
submission_rf <- read.csv('../input/sample_submission.csv')
submission_rf$Survived <- rf_pred
write.csv(submission_rf, file='submission_rf.csv', row.names = F)
```

# XG boost


```R
train <-full  %>% filter(is.na(Survived)==FALSE)
test <-full  %>% filter(is.na(Survived)==TRUE)
```


```R
train_label <- as.numeric(train$Survived)-1
test_label <- test$Survived

x_train<- model.matrix(~.-1, data = train[,-1]) %>% data.frame

x_test <- model.matrix(~.-1, data = test[,-1]) %>% data.frame
```


```R
dtrain <- xgb.DMatrix(data = as.matrix(x_train), label=train_label)
dtest <- xgb.DMatrix(data = as.matrix(x_test), label=test_label)
```


```R
param <- list(objective   = "binary:logistic",
              eval_metric = "auc",
              max_depth   = 5,
              eta         = 0.0587,
              gammma      = 0,
              subsamle    = 0.99,
              colsample_bytree = 0.94)
```


```R
# set.seed(42)
# xgb_cv <- xgb.cv(params  = param,
#               data    = dtrain,
#               nrounds = 5000,
#               nfold   = 5,
#               nthread = -1,
#               silent = 1,
#               print_every_n = 100,
#               verbose = 0)
```


```R
# auc=xgb_cv$evaluation_log
# auc  %>% filter(test_auc_mean==max(auc[,4]))
```


```R
set.seed(42)
xgb <- xgb.train(params  = param,
              data    = dtrain,
              nrounds = 1023,
              silent = 1,
              print_every_n = 100,
              verbose = 0)
```


```R
XGB_pred <- predict(xgb, dtest)
```


```R
head(XGB_pred)
```


```R
xgb.importance(colnames(dtrain), model = xgb) %>% 
  xgb.plot.importance(top_n = 30)
```


```R
submission_xgb <- read.csv('../input/sample_submission.csv')
submission_xgb$Survived <- XGB_pred
write.csv(submission_xgb, file='submission_xgb.csv', row.names = F)
```

# Light GBM


```R
train <-full  %>% filter(is.na(Survived)==FALSE)
test <-full  %>% filter(is.na(Survived)==TRUE)

x_train<- model.matrix(~.-1, data = train[,-1]) 

x_test <- model.matrix(~.-1, data = test[,-1]) 


train_label <- as.numeric(train$Survived)-1
test_label <- test$Survived
```


```R
lgb.train = lgb.Dataset(data=x_train, label=train_label)
```


```R
lgb.grid = list(objective = "binary",
                metric="auc",
                learning_rate=0.01,
                num_leaves=25,
                feature_fraction = 0.7,
                bagging_fraction = 0.7,
                bagging_freq = 5,
                nthread = 3)
```


```R
set.seed(42)
system.time(cv <- lgb.cv(lgb.grid, lgb.train, nrounds = 5000, nfold = 5, eval_freq = 100, early_stopping_rounds = 500))
```


```R
best=cv$best_iter
best 
```


```R
# set.seed(2019)
# m_lgb = lgb.train(params = lgb.grid, data = lgb.train, nrounds = 1275,nthread=-1,eval_freq=100)
```


```R
set.seed(42)
m_lgb = lgb.train(params = lgb.grid, data = lgb.train, nrounds = 1305,nthread=-1,eval_freq=100)
```


```R
pred_lgb <- predict(m_lgb, x_test)
```


```R
head(pred_lgb)
```


```R
lgb.importance(model = m_lgb) %>% 
  lgb.plot.importance(top_n = 30)
```


```R
submission_lgb <- read.csv('../input/sample_submission.csv')
submission_lgb$Survived <- pred_lgb
write.csv(submission_lgb, file='submission_lgb.csv', row.names = F)
```

# Blending


```R
blending=bind_cols(submission_rf,submission_xgb,submission_log,submission_lgb,submission_h2o)
```


```R
str(blending)
```


```R
blending=blending %>% 
        mutate(sum2= (Survived + Survived1 + Survived3 + Survived4 )/4)
```


```R
submission_blending2 <- read.csv('../input/sample_submission.csv')
submission_blending2$Survived <- blending$sum2
write.csv(submission_blending2, file='submission_blending2.csv', row.names = F)
```

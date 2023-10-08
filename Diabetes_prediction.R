# Projektarbeit - Diabetes-Prediction


# Pakete
library(tidyverse)
library(caret)
library(data.table)
library(smotefamily)
library(rpart)
library(ranger)

# Daten laden und schauen ----

metadata <- read.csv(("Datasets/diabetes_raw.csv"))

head(metadata)
# Diabetes_012 HighBP HighChol CholCheck BMI Smoker Stroke HeartDiseaseorAttack PhysActivity Fruits Veggies
# 1:  0        1        1         1      40      1      0                    0            0      0       1
# 2:  0        0        0         0      25      1      0                    0            1      0       0
# 3:  0        1        1         1      28      0      0                    0            0      1       0
# 4:  0        1        0         1      27      0      0                    0            1      1       1
# 5:  0        1        1         1      24      0      0                    0            1      1       1
# 6:  0        1        1         1      25      1      0                    0            1      1       1
#     HvyAlcoholConsump     AnyHealthcare NoDocbcCost GenHlth MentHlth PhysHlth DiffWalk Sex Age Education Income
# 1:                 0             1           0       5       18       15        1       0   9         4      3
# 2:                 0             0           1       3        0        0        0       0   7         6      1
# 3:                 0             1           1       5       30       30        1       0   9         4      8
# 4:                 0             1           0       2        0        0        0       0  11         3      6
# 5:                 0             1           0       2        3        0        0       0  11         5      4
# 6:                 0             1           0       2        0        2        0       1  10         6      8

str(metadata)
# $ Diabetes_012        : num  0=no diabetes, 1=prediabetes, 2=diabetes
# $ HighBP              : num  0=no high bloodpressure, 1 = high bloodpressure
# $ HighChol            : num  0=no high cholesterin, 1 = high cholesterin
# $ CholCheck           : num  0=no cholesterin check in 5 years, 1 = yes cholesterin check in 5 years
# $ BMI                 : num  
# $ Smoker              : num  Have you smoked 100 cigarettes in your life? 0=no, 1=yes
# $ Stroke              : num  Ever had a stroke? 0=no, 1=yes
# $ HeartDiseaseorAttack: num  Coronary Heart Disease or Myocaridal infarction? 0=no, 1=yes
# $ PhysActivity        : num  physical activity in past 30 days(not including job)? 0 = no 1 = yes
# $ Fruits              : num  Consume Fruit 1 or more times per day? 0 = no 1 = yes
# $ Veggies             : num  Consume Vegetables 1 or more times per day? 0 = no 1 = yes
# $ HvyAlcoholConsump   : num  men: more than 14 drinks/week, women:more than 7 drinks/week? 0=no, 1=yes
# $ AnyHealthcare       : num  Any kind of health care coverage? 0 = no 1 = yes
# $ NoDocbcCost         : num  Didnt see a doctor in past 12 months because of costs? 0 = no 1 = yes
# $ GenHlth             : num  What in general is your health? 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor
# $ MentHlth            : num  For how many days in last 30days was your mental health (stress, depres., emotions) not good?
# $ PhysHlth            : num  For how many days in last 30days was your physical health (illness, injury) not good?
# $ DiffWalk            : num  Serious difficulty walking/climbin stairs? 0 = no 1 = yes
# $ Sex                 : num  0 = female, 1 = male
# $ Age                 : num  1 = 18- <=24, 2 = 25- <=29, ... 13 = >=80, 14 = Missing
# $ Education           : num  Lvl of education: 1=No school/only kindergarten, 2=Elementary, 3=Some high school, 4=High school grad, 5=Some college/tech school, 6=College grad, 9=refused
# $ Income              : num  1= >10.000$, 2-4: 5.000-Schritte, 5=25-34.999$, 6=35-49.999$,7=50-74.999$ 8= >75.000$, 77=Dont know, 99=Refused, Blank=missing

# Preprocessing ----

# Random-Seed für Reproduzierbarkeit
set.seed(101)

# Umwandlung in Data.table
metadata <- as.data.table(metadata)

# Prädiabetes-Gruppe entfernen
metadata <- metadata[!metadata$Diabetes_012 == 1,]

# Kategorie 2 in 1 umwandeln
metadata$Diabetes_012[metadata$Diabetes_012 == 2] <- "diabetes"
metadata$Diabetes_012[metadata$Diabetes_012 == 0] <- "no diabetes"

# Zielvariable als Faktor umwandeln
metadata[, Diabetes_012 := as.factor(Diabetes_012)]

# Für fehlende Werte checken
sum(is.na.data.frame(metadata))

# Für Duplikate checken und entfernen
duplicates <- metadata[duplicated(metadata), ] # 23897 Duplikate (10%), aber keine ID's. Wie wahrscheinlich ist es, dass es die gleiche Person ist?
metadata <- metadata[!duplicated(metadata), ]

# Split, Resampling, Scaling ----

# Split into Train and Test ("stratify" macht die Funktion automatisch)
# And Split them into X and y
train_partition <- createDataPartition(metadata$Diabetes_012, times = 1, p = 0.8, list = FALSE)

train_data <- metadata[train_partition, ]
#train_y <- train_data$Diabetes_012
#train_data <- select(train_data, -Diabetes_012)

test_data <- metadata[-train_partition, ]
#test_y <- test_data$Diabetes_012
#test_data <- select(test_data, -Diabetes_012)

## Resampling: Up- UND Down-Sampling um Unterschied auf Modell zu testen

#Upsampling / Oversampling
data_ups <- SMOTE(X = train_data[,-"Diabetes_012"], 
                  target = train_data$Diabetes_012)$data

data_ups <- rename(data_ups, Diabetes_012 = class)

count(data_ups, Diabetes_012) 
# Diabetes_012      n
# 1:     diabetes 140390
# 2:  no diabetes 152044
#Keine 1:1-Balance, aber gut genug (1:1 würde mit SMOTE(perc.over=100) funktionieren)


data_ups[duplicated(data_ups), ] # check ob Duplikate generiert wurden

# Downsampling
data_downs <- downSample(x = train_data[,-"Diabetes_012"], 
                         y = train_data$Diabetes_012, 
                         list = FALSE, 
                         yname = "Diabetes_012")

## Skalieren
scaler_ups <- preProcess(select(data_ups, -Diabetes_012), method = c("center", "scale"))
scaler_downs <- preProcess(select(data_downs, -Diabetes_012), method = c("center", "scale"))

# Training und Testdaten skalieren (Upsampled und Downsampled Tables)
scaled_ups <- predict(scaler_ups, data_ups)
scaled_ups_test <- predict(scaler_ups, test_data)

scaled_downs <- predict(scaler_downs, data_downs)
scaled_downs_test <- predict(scaler_downs, test_data)


# Modell bauen Downsampling----

model_downs <- train(Diabetes_012 ~., data = scaled_downs, method = "rpart",
                     trControl = trainControl(method = "cv"))

model_downs # Accuracy 0.7

pred_downs <- predict(model_downs, newdata = scaled_downs_test)
print(confusionMatrix(scaled_downs_test$Diabetes_012, pred_downs)) # Accuracy 0.75


# Modell bauen Oversampling----

# Da das Oversampling-Modell ein klein bisschen besser ist(Undersampling 75%, OVersampling 76%),
# wir nur diese weiter genommen! 

modelLookup("rpart")

#minsplit <- c(2, 5, 10)
#maxdepth <- c(1, 2, 4, 5)     # nicht zu tief, sonst overfitting

model_ups <- train(Diabetes_012 ~ ., 
                   data = scaled_ups, 
                   method = "rpart",
                   trControl = trainControl(method = "cv"),
                   tuneGrid = data.frame(cp = c(0.01, 0.001)),           # nur cp lässt sich direkt tunen
                   #control = rpart.control(minsplit = 10, maxdepth = 5) # den Rest muss man über control machen (und dann die Hyperparameter normal über rpart.control)
                   ) 

model_ups

pred_ups <- predict(model_ups, newdata = scaled_ups_test)
confu_tree <- confusionMatrix(scaled_ups_test$Diabetes_012, pred_ups, mode = "everything")


# Ergebnis
results_tree <- cbind(model_ups$results["Accuracy"], 
                      confu_tree$overall["Accuracy"], 
                      confu_tree$byClass["Recall"], 
                      confu_tree$byClass["F1"])
results_tree <- results_tree[which.max(results_tree$Accuracy), ]
colnames(results_tree) = c("Accuracy Train", "Accuracy Test", "Recall", "F1")
results_tree$model <- "Decision Tree"




# Random Forest: Mehr Bäume = besserer Outcome ----

tune_forest <- expand.grid(mtry = c(4,5),
                           splitrule = c("extratrees", "gini"),
                           min.node.size = c(1)
                           )

model_forest <- train(Diabetes_012 ~ BMI + PhysActivity + Age + HighChol + HvyAlcoholConsump, 
                   data = scaled_ups, 
                   method = "ranger",
                   trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5),
                   num.trees = 5,
                   tuneGrid = tune_forest)
model_forest 

pred_forest <- predict(model_forest, newdata = scaled_ups_test)
confu_forest <- confusionMatrix(scaled_ups_test$Diabetes_012, pred_forest, mode = "everything")

# Ergebnis
results_forest <- cbind(model_forest$results["Accuracy"], 
                        confu_forest$overall["Accuracy"], 
                        confu_forest$byClass["Recall"], 
                        confu_forest$byClass["F1"])
results_forest <- results_forest[which.max(results_forest$Accuracy), ]
colnames(results_forest) = c("Accuracy Train", "Accuracy Test", "Recall", "F1")
results_forest$model <- "Random Forest"




# log. Reg (kein Tuning möglich)---

model_lm <- train(Diabetes_012 ~ BMI + PhysActivity + Age + HighChol + HvyAlcoholConsump, 
                   data = scaled_ups, 
                   method = "glm",
                  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5)
                  )
model_lm
pred_lm <- predict(model_lm, newdata = scaled_ups_test)
confu_lm <- confusionMatrix(scaled_ups_test$Diabetes_012, pred_lm, mode = "everything")

#Ergebnis
results_lm <- cbind(model_lm$results["Accuracy"], 
                    confu_lm$overall["Accuracy"], 
                    confu_lm$byClass["Recall"], 
                    confu_lm$byClass["F1"])
colnames(results_lm) = c("Accuracy Train", "Accuracy Test", "Recall", "F1")
results_lm$model <- "logistic Regression"


# Ergebnisse in Tabelle zusammenfassen

results_final <- rbind(results_tree, results_forest, results_lm)
results_final <- results_final[, c(5,1,2,3,4)]
results_final <- mutate_if(results_final, .predicate = is.numeric, round, digits = 2)

write.csv(results_final, file = "results_final.csv")

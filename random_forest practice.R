library(mlr)
library(Amelia)

#load data
data <-  read.csv('dataR2.csv')
head(data)

#summary of the data
summarizeColumns(data)

#checking missing value
missmap(data)

#reproducible research
set.seed(123)

#split data set
n = nrow(data)
train.set = sample(n, size = 2/3*n)
test.set = setdiff(1:n, train.set)

#making regression task
task <- makeClassifTask(data = data, target = 'Classification', positive = 1)
lrn <- makeLearner('classif.randomForest', predict.type = 'prob')

mod <- train(lrn, task, subset = train.set)
pred <- predict(mod, task, subset = test.set)

performance(pred, measures = list(fpr, fnr, mmce, acc))

df = generateThreshVsPerfData(pred, measures = list(fpr, tpr, mmce, acc))
plotThreshVsPerf(df)
plotROCCurves(df)

calculateConfusionMatrix(pred)

#parameter tuning 
getParamSet('classif.randomForest')

ps = makeParamSet(
  makeIntegerParam('ntree', lower = 500, upper = 1500),
  makeIntegerParam('mtry', lower = 5, upper = 10)
)

#Stratified resampling using 10 fold
ctrl <- makeTuneControlGrid()
rdesc <- makeResampleDesc('CV', iters = 10L, stratify = T)

#Tune process
res <- tuneParams('classif.randomForest', task = task, resampling = rdesc, par.set = ps, control = ctrl, show.info = F)
res

res$x

res$y

tunedata <- generateHyperParsEffectData(res)
plotHyperParsEffect(tunedata, x = 'mtry', y = 'mmce.test.mean', plot.type = 'line')
plotHyperParsEffect(tunedata, x = 'ntree', y = 'mmce.test.mean', plot.type = 'line')


tunedlearners <- setHyperPars(makeLearner('classif.randomForest'), par.vals = res$x)


tunedlearners1 <- makeTuneWrapper(lrn, rdesc, mmce, ps, ctrl, show.info = F)
tunedmod <- train(tunedlearners1, task, subset = train.set)

tunedpred <- predict(tunedmod, task, subset = test.set)

performance(tunedpred, measures = list(fpr, fnr, mmce, acc))

df = generateThreshVsPerfData(tunedpred, measures = list(fpr, tpr, mmce, acc))
plotThreshVsPerf(df)

plotROCCurves(df)

calculateConfusionMatrix(tunedpred)

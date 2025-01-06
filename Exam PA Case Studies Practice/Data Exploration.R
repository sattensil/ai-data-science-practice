library(ISLR2)
college = College
#rownames(college) <- college[, 1]
View(college)
summary(college)
pairs(college[,1:10])
plot(college$Private,college$Outstate)
Elite <- rep("No", nrow(college))
Elite[college$Top10perc > 50] <- "Yes"
Elite <- as.factor(Elite)
college <- data.frame(college, Elite)
summary(college$Elite)
plot(college$Elite,college$Outstate)
par(mfrow = c(2, 2))
hist(college$Outstate, col = 2, breaks = 15)
hist(college$Accept, col = 2, breaks = 15)
hist(college$Apps, col = 2, breaks = 15)
hist(college$Top10perc, col = 2, breaks = 15)


# Install packages (run once only)
install.packages("tidyverse")
install.packages("ggplot2")
install.packages("dplyr")

# Load libraries
library(tidyverse)
library(ggplot2)
library(dplyr)

1.
data <- read.csv("nsmhw_sociodemographic_profile.csv")
head(data)

2.
str(data)
summary(data)

3.
colSums(is.na(data))

4.
data <- distinct(data)

5.
data$Category <- tolower(data$Category)
data$Group <- trimws(data$Group)

6.
data$Frequency <- as.numeric(data$Frequency)
data$Percentage <- as.numeric(data$Percentage)

7.
head(data)
str(data)

8.
category_summary <- data %>%
  group_by(Category) %>%
  summarise(Total = sum(Frequency))

category_summary

9.
ggplot(data, aes(x = Group, y = Frequency)) +
  geom_bar(stat = "identity") +
  coord_flip()

10.
write.csv(data, "cleaned_nsmhw_data.csv", row.names = FALSE)
# ============================================================
# MindTrack PH - Phase 4: Model Development & Data Analytics
# IT 030 - Data Analytics | Group 6
# Dataset: NSMHW Sociodemographic Profile
# ============================================================

# ── 0. Install & Load Packages ────────────────────────────────
packages <- c("tidyverse", "ggplot2", "dplyr", "caret",
              "randomForest", "rpart", "rpart.plot",
              "corrplot", "scales", "ggthemes", "gridExtra",
              "cluster", "factoextra", "e1071")

installed <- rownames(installed.packages())
for (pkg in packages) {
  if (!pkg %in% installed) install.packages(pkg, dependencies = TRUE)
}

library(tidyverse)
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(corrplot)
library(scales)
library(ggthemes)
library(gridExtra)
library(cluster)
library(factoextra)
library(e1071)

# ── 1. Load Cleaned Data ──────────────────────────────────────
data <- read.csv("cleaned_nsmhw_data.csv", stringsAsFactors = FALSE)

cat("=== Dataset Overview ===\n")
print(head(data, 10))
print(str(data))
print(summary(data))
cat("Dimensions:", nrow(data), "rows x", ncol(data), "columns\n\n")

# ── 2. Feature Engineering ────────────────────────────────────
# Encode category as factor
data$Category <- as.factor(data$Category)
data$Group    <- trimws(data$Group)

# Classify percentage into risk bands
data$Risk_Level <- cut(
  data$Percentage,
  breaks = c(0, 20, 40, 60, 100),
  labels = c("Low", "Moderate", "High", "Very High"),
  include.lowest = TRUE
)

# Binary target: High burden (Percentage >= 40%)
data$High_Burden <- ifelse(data$Percentage >= 40, 1, 0)
data$High_Burden_Factor <- as.factor(data$High_Burden)

cat("=== Feature Engineering Complete ===\n")
cat("Risk Level distribution:\n")
print(table(data$Risk_Level))
cat("\nHigh Burden distribution:\n")
print(table(data$High_Burden))

# ── 3. Exploratory Visualizations (Phase 4 Quality) ──────────

# (A) Distribution of Percentage by Category
p1 <- ggplot(data, aes(x = reorder(Category, Percentage, FUN = median),
                        y = Percentage, fill = Category)) +
  geom_boxplot(show.legend = FALSE, alpha = 0.85, color = "gray30") +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title    = "Figure 3: Mental Health Burden Distribution by Category",
    subtitle = "Percentage of respondents per sociodemographic group",
    x        = "Sociodemographic Category",
    y        = "Percentage (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figure3_boxplot_category.png", p1, width = 9, height = 5, dpi = 150)
print(p1)

# (B) Risk Level frequency bar chart
p2 <- ggplot(data, aes(x = Risk_Level, fill = Risk_Level)) +
  geom_bar(show.legend = FALSE, alpha = 0.9, color = "white") +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("Low"       = "#2ecc71",
                                "Moderate"  = "#f1c40f",
                                "High"      = "#e67e22",
                                "Very High" = "#e74c3c")) +
  labs(
    title    = "Figure 4: Distribution of Risk Levels",
    subtitle = "Classified from percentage of affected respondents",
    x        = "Risk Level",
    y        = "Count"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figure4_risk_level_bar.png", p2, width = 7, height = 5, dpi = 150)
print(p2)

# (C) Top 10 groups by Frequency
top10 <- data %>%
  arrange(desc(Frequency)) %>%
  slice_head(n = 10)

p3 <- ggplot(top10, aes(x = reorder(Group, Frequency), y = Frequency, fill = Category)) +
  geom_col(alpha = 0.9, color = "white") +
  coord_flip() +
  scale_fill_brewer(palette = "Paired") +
  scale_y_continuous(labels = comma) +
  labs(
    title    = "Figure 5: Top 10 Sociodemographic Groups by Frequency",
    subtitle = "Number of respondents in each group",
    x        = "Group",
    y        = "Frequency",
    fill     = "Category"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figure5_top10_groups.png", p3, width = 10, height = 5, dpi = 150)
print(p3)

# (D) Scatter – Frequency vs Percentage with risk overlay
p4 <- ggplot(data, aes(x = Frequency, y = Percentage,
                        color = Risk_Level, shape = Category)) +
  geom_point(size = 3.5, alpha = 0.85) +
  scale_color_manual(values = c("Low"       = "#27ae60",
                                 "Moderate"  = "#f39c12",
                                 "High"      = "#e67e22",
                                 "Very High" = "#c0392b")) +
  scale_x_continuous(labels = comma) +
  labs(
    title    = "Figure 6: Frequency vs. Percentage by Risk Level",
    subtitle = "Each point = one sociodemographic group",
    x        = "Frequency (No. of Respondents)",
    y        = "Percentage (%)",
    color    = "Risk Level",
    shape    = "Category"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figure6_scatter_freq_pct.png", p4, width = 9, height = 5, dpi = 150)
print(p4)

# ── 4. Model 1: Decision Tree Classifier ─────────────────────
cat("\n=== MODEL 1: Decision Tree (High Burden Classification) ===\n")

set.seed(42)
dt_data <- data %>%
  select(Frequency, Percentage, Category, High_Burden_Factor) %>%
  drop_na()

train_idx  <- createDataPartition(dt_data$High_Burden_Factor, p = 0.75, list = FALSE)
train_data <- dt_data[train_idx, ]
test_data  <- dt_data[-train_idx, ]

dt_model <- rpart(
  High_Burden_Factor ~ Frequency + Percentage + Category,
  data   = train_data,
  method = "class",
  control = rpart.control(minsplit = 2, cp = 0.01)
)

# Plot decision tree
png("figure7_decision_tree.png", width = 900, height = 600, res = 120)
rpart.plot(dt_model,
           main   = "Figure 7: Decision Tree – High Mental Health Burden",
           type   = 4,
           extra  = 104,
           fallen.leaves = TRUE,
           box.palette   = "GnRd",
           shadow.col    = "gray")
dev.off()

# Evaluate
dt_pred <- predict(dt_model, test_data, type = "class")
dt_cm   <- confusionMatrix(dt_pred, test_data$High_Burden_Factor)
cat("\nDecision Tree Confusion Matrix:\n")
print(dt_cm)
cat("\nDecision Tree Accuracy:", round(dt_cm$overall["Accuracy"] * 100, 2), "%\n")

# ── 5. Model 2: Random Forest ─────────────────────────────────
cat("\n=== MODEL 2: Random Forest ===\n")

rf_model <- randomForest(
  High_Burden_Factor ~ Frequency + Percentage + Category,
  data       = train_data,
  ntree      = 200,
  importance = TRUE
)

print(rf_model)

rf_pred <- predict(rf_model, test_data)
rf_cm   <- confusionMatrix(rf_pred, test_data$High_Burden_Factor)
cat("\nRandom Forest Confusion Matrix:\n")
print(rf_cm)
cat("\nRandom Forest Accuracy:", round(rf_cm$overall["Accuracy"] * 100, 2), "%\n")

# Variable importance plot
imp_df <- as.data.frame(importance(rf_model))
imp_df$Variable <- rownames(imp_df)

p5 <- ggplot(imp_df, aes(x = reorder(Variable, MeanDecreaseGini),
                           y = MeanDecreaseGini, fill = MeanDecreaseGini)) +
  geom_col(show.legend = FALSE, alpha = 0.9) +
  coord_flip() +
  scale_fill_gradient(low = "#aed6f1", high = "#1a5276") +
  labs(
    title    = "Figure 8: Random Forest – Variable Importance",
    subtitle = "Mean Decrease in Gini Index",
    x        = "Variable",
    y        = "Mean Decrease Gini"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figure8_rf_importance.png", p5, width = 7, height = 4, dpi = 150)
print(p5)

# ── 6. Model 3: Linear Regression (Frequency → Percentage) ───
cat("\n=== MODEL 3: Linear Regression ===\n")

lm_model <- lm(Percentage ~ Frequency + Category, data = data)
lm_summary <- summary(lm_model)
print(lm_summary)

cat("\nR-squared:", round(lm_summary$r.squared, 4))
cat("\nAdjusted R-squared:", round(lm_summary$adj.r.squared, 4))
cat("\nRMSE:", round(sqrt(mean(lm_model$residuals^2)), 4), "\n")

# Residual plot
lm_resid_df <- data.frame(
  Fitted    = fitted(lm_model),
  Residuals = residuals(lm_model)
)

p6 <- ggplot(lm_resid_df, aes(x = Fitted, y = Residuals)) +
  geom_point(color = "#2980b9", alpha = 0.7, size = 2.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title    = "Figure 9: Linear Regression – Residuals vs. Fitted",
    subtitle = "Checking model assumptions",
    x        = "Fitted Values",
    y        = "Residuals"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figure9_lm_residuals.png", p6, width = 7, height = 4, dpi = 150)
print(p6)

# ── 7. Model 4: K-Means Clustering ───────────────────────────
cat("\n=== MODEL 4: K-Means Clustering ===\n")

cluster_data <- data %>%
  select(Frequency, Percentage) %>%
  scale() %>%
  as.data.frame()

# Determine optimal k via elbow method
set.seed(42)
wss <- map_dbl(1:6, function(k) {
  kmeans(cluster_data, centers = k, nstart = 25)$tot.withinss
})

elbow_df <- data.frame(k = 1:6, WSS = wss)
p7 <- ggplot(elbow_df, aes(x = k, y = WSS)) +
  geom_line(color = "#2980b9", size = 1.2) +
  geom_point(color = "#c0392b", size = 3) +
  labs(
    title    = "Figure 10: Elbow Method for Optimal K",
    subtitle = "Within-cluster sum of squares",
    x        = "Number of Clusters (k)",
    y        = "Total Within-Cluster SS"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figure10_elbow.png", p7, width = 7, height = 4, dpi = 150)
print(p7)

# Fit K-Means with k=3
km_model <- kmeans(cluster_data, centers = 3, nstart = 25)
data$Cluster <- as.factor(km_model$cluster)

cat("\nCluster sizes:", km_model$size, "\n")
cat("Within-cluster SS:", round(km_model$tot.withinss, 2), "\n")
cat("Between-cluster SS:", round(km_model$betweenss, 2), "\n")

# Cluster visualization
p8 <- fviz_cluster(km_model, data = cluster_data,
                    palette       = c("#2ecc71", "#e74c3c", "#3498db"),
                    geom          = "point",
                    ellipse.type  = "convex",
                    ggtheme       = theme_minimal()) +
  labs(
    title    = "Figure 11: K-Means Clustering (k=3)",
    subtitle = "Grouping sociodemographic profiles by Frequency & Percentage"
  ) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figure11_kmeans_cluster.png", p8, width = 8, height = 5, dpi = 150)
print(p8)

# Cluster profile summary
cluster_profile <- data %>%
  group_by(Cluster) %>%
  summarise(
    Avg_Frequency  = round(mean(Frequency), 2),
    Avg_Percentage = round(mean(Percentage), 2),
    Count          = n(),
    .groups = "drop"
  )
cat("\nCluster Profile:\n")
print(cluster_profile)

# ── 8. Model Comparison Summary ──────────────────────────────
cat("\n\n============================================\n")
cat("       MODEL EVALUATION SUMMARY\n")
cat("============================================\n")
cat(sprintf("%-30s %s\n", "Model", "Accuracy / Key Metric"))
cat(sprintf("%-30s %s\n", "-----", "---------------------"))
cat(sprintf("%-30s %.2f%%\n",
            "Decision Tree (Classifier)",
            dt_cm$overall["Accuracy"] * 100))
cat(sprintf("%-30s %.2f%%\n",
            "Random Forest (Classifier)",
            rf_cm$overall["Accuracy"] * 100))
cat(sprintf("%-30s R² = %.4f | RMSE = %.4f\n",
            "Linear Regression",
            lm_summary$r.squared,
            sqrt(mean(lm_model$residuals^2))))
cat(sprintf("%-30s k=3 | Between SS = %.2f\n",
            "K-Means Clustering",
            km_model$betweenss))
cat("============================================\n\n")

# ── 9. Save Final Annotated Dataset ──────────────────────────
write.csv(data, "phase4_final_dataset.csv", row.names = FALSE)
cat("Final dataset saved: phase4_final_dataset.csv\n")
cat("All figures saved as PNG files.\n")
cat("\nPhase 4 complete!\n")

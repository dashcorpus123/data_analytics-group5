# ============================================================
# MindTrack PH – Phase 5: R Shiny Dashboard
# IT 030 – Data Analytics | Group 6 | IT33S4
# Instructor: Ms. Nila D. Santiago
# ============================================================

# ── 0. Install & Load Packages ──────────────────────────────
packages <- c(
  "shiny", "shinydashboard", "tidyverse", "dplyr",
  "ggplot2", "plotly", "DT", "scales",
  "caret", "rpart", "rpart.plot",
  "randomForest", "cluster", "factoextra", "e1071"
)
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, dependencies = TRUE)
}

library(shiny)
library(shinydashboard)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(plotly)
library(DT)
library(scales)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(cluster)
library(factoextra)
library(e1071)

# ── 1. Load & Prepare Data ──────────────────────────────────
# Reads cleaned_nsmhw_data.csv (Phase 3 output).
# If the file is missing a built-in demo dataset is used
# so the dashboard runs without any external files.

load_data <- function() {
  if (file.exists("cleaned_nsmhw_data.csv")) {
    df <- read.csv("cleaned_nsmhw_data.csv", stringsAsFactors = FALSE)
  } else {
    df <- data.frame(
      Category = c(
        rep("Sex", 2),
        rep("Age Group (18 years old and above)", 4),
        rep("Marital Status", 4),
        rep("Education Level", 4),
        rep("Employment Status", 2),
        rep("Urbanization", 2)
      ),
      Group = c(
        "Male", "Female",
        "18-34 (Early adult)", "35-49 (Middle adult)",
        "50-64 (Late adult)", "65+ (Elderly)",
        "Single", "Married/Cohabiting",
        "Separated/Widowed", "Other",
        "No formal education", "Elementary/High School",
        "College", "Post-Graduate",
        "Employed", "Unemployed/Others",
        "Urban", "Rural"
      ),
      Frequency = c(
        3817, 6037, 4259, 2759, 2011, 825,
        3542, 5106, 890, 316,
        412, 4931, 3802, 709,
        5248, 4606, 6221, 3633
      ),
      Percentage = c(
        38.74, 61.26, 43.22, 28.00, 20.41, 8.37,
        35.94, 51.82, 9.03, 3.21,
        4.18, 50.02, 38.59, 7.20,
        53.24, 46.76, 63.13, 36.87
      ),
      stringsAsFactors = FALSE
    )
  }

  df$Category <- as.factor(df$Category)
  df$Group    <- trimws(df$Group)

  df$Risk_Level <- cut(
    df$Percentage,
    breaks         = c(0, 20, 40, 60, 100),
    labels         = c("Low", "Moderate", "High", "Very High"),
    include.lowest = TRUE
  )

  df$High_Burden     <- ifelse(df$Percentage >= 40, 1, 0)
  df$High_Burden_Fac <- as.factor(df$High_Burden)
  df
}

dat <- load_data()

# ── 2. Train Models Once at Startup ─────────────────────────
set.seed(42)
train_idx  <- createDataPartition(dat$High_Burden_Fac, p = 0.75, list = FALSE)
train_dat  <- dat[train_idx, ]
test_dat   <- dat[-train_idx, ]

dt_model <- rpart(
  High_Burden_Fac ~ Frequency + Percentage + Category,
  data    = train_dat, method = "class",
  control = rpart.control(minsplit = 2, cp = 0.01)
)

rf_model <- randomForest(
  High_Burden_Fac ~ Frequency + Percentage + Category,
  data = train_dat, ntree = 200, importance = TRUE
)

lm_model <- lm(Percentage ~ Frequency + Category, data = dat)

clust_scaled <- scale(dat[, c("Frequency", "Percentage")])
set.seed(42)
km_model     <- kmeans(clust_scaled, centers = 3, nstart = 25)
dat$Cluster  <- as.factor(km_model$cluster)

# Pre-compute evaluation metrics
dt_pred <- predict(dt_model, test_dat, type = "class")
dt_cm   <- confusionMatrix(dt_pred, test_dat$High_Burden_Fac)
dt_acc  <- round(dt_cm$overall["Accuracy"] * 100, 1)

rf_pred <- predict(rf_model, test_dat)
rf_cm   <- confusionMatrix(rf_pred, test_dat$High_Burden_Fac)
rf_acc  <- round(rf_cm$overall["Accuracy"] * 100, 1)

lm_r2   <- round(summary(lm_model)$r.squared, 4)
lm_rmse <- round(sqrt(mean(lm_model$residuals^2)), 4)

# ── 3. Colour Palette ────────────────────────────────────────
risk_pal <- c(
  "Low"       = "#27ae60",
  "Moderate"  = "#f39c12",
  "High"      = "#e67e22",
  "Very High" = "#c0392b"
)

clust_pal <- c("1" = "#3498db", "2" = "#e74c3c", "3" = "#2ecc71")

# ── 4. Shared plot layout helper ────────────────────────────
transparent_bg <- function(p) {
  p %>% layout(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = list(family = "Arial", size = 12)
  )
}

# ============================================================
# UI
# ============================================================
ui <- dashboardPage(
  skin = "blue",

  # ── Header ──────────────────────────────────────────────────
  dashboardHeader(
    title      = "MindTrack PH",
    titleWidth = 240
  ),

  # ── Sidebar ─────────────────────────────────────────────────
  dashboardSidebar(
    width = 240,
    sidebarMenu(
      id = "sidebar",
      menuItem("Overview",      tabName = "overview",     icon = icon("chart-pie")),
      menuItem("Demographics",  tabName = "demographics", icon = icon("users")),
      menuItem("Risk Analysis", tabName = "risk",         icon = icon("triangle-exclamation")),
      menuItem("Model Results", tabName = "models",       icon = icon("brain")),
      menuItem("Clustering",    tabName = "clustering",   icon = icon("circle-nodes")),
      menuItem("Data Table",    tabName = "datatable",    icon = icon("table"))
    ),
    tags$hr(style = "border-color:#4a6785; margin:10px 15px;"),
    tags$div(
      style = "padding: 0 15px 10px;",
      tags$p(
        style = "color:#b8c7ce; font-size:12px; margin-bottom:8px; font-weight:600;",
        icon("sliders"), "  FILTERS"
      ),
      selectInput(
        "flt_cat", "Category",
        choices  = c("All", levels(dat$Category)),
        selected = "All"
      ),
      sliderInput(
        "flt_pct", "Percentage Range (%)",
        min = 0, max = 100, value = c(0, 100), step = 1
      ),
      selectInput(
        "flt_risk", "Risk Level",
        choices  = c("All", "Low", "Moderate", "High", "Very High"),
        selected = "All"
      ),
      tags$p(
        style = "color:#8aa4bf; font-size:10px; margin-top:6px;",
        "Filters apply to Demographics & Risk tabs."
      )
    )
  ),

  # ── Body ────────────────────────────────────────────────────
  dashboardBody(
    tags$head(
      tags$style(HTML("
        /* General */
        body, .content-wrapper, .main-footer { background-color: #f4f6f9; }
        .box       { border-radius: 6px; box-shadow: 0 2px 6px rgba(0,0,0,.08); }
        .info-box  { border-radius: 6px; }
        .value-box { border-radius: 6px; }

        /* Sidebar */
        .skin-blue .main-sidebar { background-color: #1c2b3a; }
        .skin-blue .sidebar a    { color: #b8c7ce !important; }
        .skin-blue .sidebar .active a { color: #fff !important; }
        .skin-blue .main-sidebar .sidebar .sidebar-menu li.active > a {
          background-color: #2980b9;
        }

        /* Header */
        .skin-blue .main-header .logo { background-color: #1a252f; font-weight: 700; }
        .skin-blue .main-header .navbar { background-color: #1a252f; }

        /* Select / Slider labels */
        .control-label { color: #b8c7ce; font-size: 12px; }

        /* Tab box */
        .nav-tabs-custom > .nav-tabs > li.active > a { border-top-color: #2980b9; }
      "))
    ),

    tabItems(

      # ════════════════════════════════════════════════════════
      # TAB 1 – OVERVIEW
      # ════════════════════════════════════════════════════════
      tabItem(tabName = "overview",

        # Info boxes row
        fluidRow(
          infoBox(
            "Total Respondents",
            format(sum(dat$Frequency), big.mark = ","),
            icon  = icon("people-group"), color = "blue",  width = 3
          ),
          infoBox(
            "High-Burden Groups",
            paste0(sum(dat$High_Burden), " of ", nrow(dat)),
            icon  = icon("circle-exclamation"), color = "red",    width = 3
          ),
          infoBox(
            "Categories Covered",
            nlevels(dat$Category),
            icon  = icon("layer-group"), color = "green",  width = 3
          ),
          infoBox(
            "Largest Group",
            {
              g <- dat[which.max(dat$Percentage), ]
              paste0(g$Group, " — ", g$Percentage, "%")
            },
            icon  = icon("arrow-trend-up"), color = "yellow", width = 3
          )
        ),

        # Charts row
        fluidRow(
          box(
            title = "Frequency vs. Percentage by Risk Level",
            status = "primary", solidHeader = TRUE,
            width = 7, height = 430,
            plotlyOutput("ov_scatter", height = 370)
          ),
          box(
            title = "Risk Level Breakdown",
            status = "warning", solidHeader = TRUE,
            width = 5, height = 430,
            plotlyOutput("ov_donut", height = 370)
          )
        ),

        # About box
        fluidRow(
          box(
            title = "About MindTrack PH", width = 12,
            status = "info", collapsible = TRUE, collapsed = FALSE,
            tags$p(
              style = "font-size:14px; line-height:1.8; color:#333;",
              tags$b("MindTrack PH"), " is an interactive analytics dashboard exploring mental
              health trends in the Philippines using the National Survey on Mental Health and
              Well-being (NSMHW) Sociodemographic Profile dataset. Four machine learning models
              were developed in Phase 4 — Decision Tree, Random Forest, Linear Regression, and
              K-Means Clustering — and their results are visualised across this dashboard."
            ),
            tags$p(
              style = "font-size:14px; line-height:1.8; color:#333;",
              "Use the ", tags$b("sidebar filters"), " to slice data by category, percentage
              range, and risk level. Navigate the tabs above to explore demographics, risk
              patterns, model performance, and population clusters."
            )
          )
        )
      ),

      # ════════════════════════════════════════════════════════
      # TAB 2 – DEMOGRAPHICS
      # ════════════════════════════════════════════════════════
      tabItem(tabName = "demographics",
        fluidRow(
          box(
            title = "Top Groups by Number of Respondents",
            status = "primary", solidHeader = TRUE,
            width = 8, height = 460,
            plotlyOutput("dem_bar", height = 400)
          ),
          box(
            title = "Share of Respondents by Category",
            status = "success", solidHeader = TRUE,
            width = 4, height = 460,
            plotlyOutput("dem_donut", height = 400)
          )
        ),
        fluidRow(
          box(
            title = "Percentage Distribution by Category (Boxplot)",
            status = "warning", solidHeader = TRUE,
            width = 12, height = 400,
            plotlyOutput("dem_box", height = 340)
          )
        ),
        fluidRow(
          box(
            title = "Sociodemographic Profile",
            status = "info", solidHeader = TRUE, width = 12,
            DTOutput("dem_table")
          )
        )
      ),

      # ════════════════════════════════════════════════════════
      # TAB 3 – RISK ANALYSIS
      # ════════════════════════════════════════════════════════
      tabItem(tabName = "risk",
        fluidRow(
          box(
            title = "Bubble Chart – Frequency vs. Percentage (sized by % share)",
            status = "danger", solidHeader = TRUE,
            width = 7, height = 450,
            plotlyOutput("risk_bubble", height = 390)
          ),
          box(
            title = "Risk Level Count",
            status = "warning", solidHeader = TRUE,
            width = 5, height = 450,
            plotlyOutput("risk_bar", height = 390)
          )
        ),
        fluidRow(
          box(
            title = "Percentage Heatmap — All Groups",
            status = "primary", solidHeader = TRUE,
            width = 12, height = 380,
            plotlyOutput("risk_heat", height = 320)
          )
        )
      ),

      # ════════════════════════════════════════════════════════
      # TAB 4 – MODEL RESULTS
      # ════════════════════════════════════════════════════════
      tabItem(tabName = "models",

        # Value boxes
        fluidRow(
          valueBox(
            paste0(dt_acc, "%"), "Decision Tree Accuracy",
            icon = icon("tree"),     color = "green",  width = 3
          ),
          valueBox(
            paste0(rf_acc, "%"), "Random Forest Accuracy",
            icon = icon("seedling"), color = "teal",   width = 3
          ),
          valueBox(
            lm_r2,  "Linear Regression R²",
            icon = icon("chart-line"), color = "blue",   width = 3
          ),
          valueBox(
            lm_rmse, "Linear Regression RMSE",
            icon = icon("ruler"),    color = "yellow", width = 3
          )
        ),

        # Model detail tabs
        fluidRow(
          tabBox(
            id = "model_tabs", width = 12,
            title = "Model Details",

            tabPanel("Decision Tree",
              fluidRow(
                column(6,
                  h4("Variable Importance", style = "color:#2c3e50; font-weight:700;"),
                  plotlyOutput("dt_imp", height = 280)
                ),
                column(6,
                  h4("Confusion Matrix", style = "color:#2c3e50; font-weight:700;"),
                  verbatimTextOutput("dt_cm")
                )
              )
            ),

            tabPanel("Random Forest",
              fluidRow(
                column(6,
                  h4("Variable Importance (Mean Decrease Gini)",
                     style = "color:#2c3e50; font-weight:700;"),
                  plotlyOutput("rf_imp", height = 280)
                ),
                column(6,
                  h4("Confusion Matrix", style = "color:#2c3e50; font-weight:700;"),
                  verbatimTextOutput("rf_cm")
                )
              )
            ),

            tabPanel("Linear Regression",
              fluidRow(
                column(6,
                  h4("Residuals vs. Fitted Values",
                     style = "color:#2c3e50; font-weight:700;"),
                  plotlyOutput("lm_resid", height = 280)
                ),
                column(6,
                  h4("Model Summary", style = "color:#2c3e50; font-weight:700;"),
                  verbatimTextOutput("lm_summ")
                )
              )
            ),

            tabPanel("Model Comparison",
              fluidRow(
                column(12,
                  h4("Performance Summary", style = "color:#2c3e50; font-weight:700;"),
                  plotlyOutput("model_compare", height = 320)
                )
              )
            )
          )
        )
      ),

      # ════════════════════════════════════════════════════════
      # TAB 5 – CLUSTERING
      # ════════════════════════════════════════════════════════
      tabItem(tabName = "clustering",
        fluidRow(
          box(
            title = "K-Means Cluster Plot (k = 3)",
            status = "primary", solidHeader = TRUE,
            width = 7, height = 460,
            plotlyOutput("clust_plot", height = 400)
          ),
          box(
            title = "Elbow Method – Choosing Optimal k",
            status = "info", solidHeader = TRUE,
            width = 5, height = 460,
            plotlyOutput("elbow_plot", height = 400)
          )
        ),
        fluidRow(
          box(
            title = "Cluster Profiles",
            status = "success", solidHeader = TRUE, width = 12,
            DTOutput("clust_table")
          )
        )
      ),

      # ════════════════════════════════════════════════════════
      # TAB 6 – DATA TABLE
      # ════════════════════════════════════════════════════════
      tabItem(tabName = "datatable",
        fluidRow(
          box(
            title = "Full Dataset with Engineered Features",
            status = "primary", solidHeader = TRUE, width = 12,
            tags$div(
              style = "margin-bottom:10px;",
              downloadButton("dl_csv", " Download CSV",
                             icon = icon("download"),
                             class = "btn-success btn-sm")
            ),
            DTOutput("full_table")
          )
        )
      )

    ) # end tabItems
  )   # end dashboardBody
)     # end dashboardPage


# ============================================================
# SERVER
# ============================================================
server <- function(input, output, session) {

  # ── Reactive filtered dataset ────────────────────────────────
  fd <- reactive({
    d <- dat
    if (input$flt_cat != "All")
      d <- d %>% filter(Category == input$flt_cat)
    d <- d %>% filter(
      Percentage >= input$flt_pct[1],
      Percentage <= input$flt_pct[2]
    )
    if (input$flt_risk != "All")
      d <- d %>% filter(Risk_Level == input$flt_risk)
    d
  })

  # ════════════════════════════════════════════════════════════
  # OVERVIEW
  # ════════════════════════════════════════════════════════════

  output$ov_scatter <- renderPlotly({
    plot_ly(
      dat,
      x         = ~Frequency,
      y         = ~Percentage,
      color     = ~Risk_Level,
      colors    = risk_pal,
      type      = "scatter",
      mode      = "markers",
      marker    = list(size = 13, opacity = 0.85,
                       line = list(color = "white", width = 1)),
      text      = ~paste0(
        "<b>", Group, "</b><br>",
        "Category: ", Category, "<br>",
        "Frequency: ", format(Frequency, big.mark = ","), "<br>",
        "Percentage: ", Percentage, "%<br>",
        "Risk Level: ", Risk_Level
      ),
      hoverinfo = "text"
    ) %>%
      layout(
        xaxis  = list(title = "Number of Respondents", tickformat = ","),
        yaxis  = list(title = "Percentage (%)"),
        legend = list(title = list(text = "<b>Risk Level</b>"))
      ) %>%
      transparent_bg()
  })

  output$ov_donut <- renderPlotly({
    rl <- dat %>% count(Risk_Level)
    plot_ly(
      rl,
      labels    = ~Risk_Level,
      values    = ~n,
      type      = "pie",
      hole      = 0.45,
      marker    = list(
        colors = c("#27ae60", "#f39c12", "#e67e22", "#c0392b"),
        line   = list(color = "white", width = 2)
      ),
      textinfo  = "label+percent",
      hoverinfo = "label+value+percent"
    ) %>%
      transparent_bg()
  })

  # ════════════════════════════════════════════════════════════
  # DEMOGRAPHICS
  # ════════════════════════════════════════════════════════════

  output$dem_bar <- renderPlotly({
    d <- fd() %>% arrange(desc(Frequency)) %>% slice_head(n = 15)
    plot_ly(
      d,
      x         = ~Frequency,
      y         = ~reorder(Group, Frequency),
      color     = ~Category,
      type      = "bar",
      orientation = "h",
      text      = ~format(Frequency, big.mark = ","),
      textposition = "outside",
      hovertemplate = paste(
        "<b>%{y}</b><br>Respondents: %{x:,}<extra></extra>"
      )
    ) %>%
      layout(
        xaxis   = list(title = "Frequency", tickformat = ","),
        yaxis   = list(title = ""),
        barmode = "stack",
        legend  = list(title = list(text = "<b>Category</b>"))
      ) %>%
      transparent_bg()
  })

  output$dem_donut <- renderPlotly({
    d <- fd() %>%
      group_by(Category) %>%
      summarise(Total = sum(Frequency), .groups = "drop")
    plot_ly(
      d,
      labels    = ~Category,
      values    = ~Total,
      type      = "pie",
      hole      = 0.5,
      textinfo  = "label+percent",
      hoverinfo = "label+value"
    ) %>%
      transparent_bg()
  })

  output$dem_box <- renderPlotly({
    d <- fd()
    p <- ggplot(d, aes(
      x    = reorder(Category, Percentage, FUN = median),
      y    = Percentage,
      fill = Category,
      text = paste0("Group: ", Group, "\n", Percentage, "%")
    )) +
      geom_boxplot(alpha = 0.8, show.legend = FALSE, outlier.colour = "red") +
      geom_jitter(width = 0.15, alpha = 0.6, size = 2, colour = "#2c3e50") +
      coord_flip() +
      scale_fill_brewer(palette = "Set2") +
      labs(x = NULL, y = "Percentage (%)") +
      theme_minimal(base_size = 12)
    ggplotly(p, tooltip = "text") %>% transparent_bg()
  })

  output$dem_table <- renderDT({
    fd() %>%
      select(Category, Group, Frequency, Percentage, Risk_Level, High_Burden) %>%
      arrange(Category, desc(Percentage)) %>%
      datatable(
        rownames = FALSE,
        options  = list(pageLength = 10, scrollX = TRUE,
                        columnDefs = list(list(className = "dt-center",
                                               targets = 2:5))),
        colnames = c("Category", "Group", "Frequency",
                     "Percentage (%)", "Risk Level", "High Burden (1=Yes)")
      ) %>%
      formatRound("Percentage", digits = 2) %>%
      formatCurrency("Frequency", currency = "", digits = 0) %>%
      formatStyle(
        "Risk_Level",
        backgroundColor = styleEqual(
          c("Low", "Moderate", "High", "Very High"),
          c("#d5f5e3", "#fef9e7", "#fdebd0", "#fadbd8")
        )
      )
  })

  # ════════════════════════════════════════════════════════════
  # RISK ANALYSIS
  # ════════════════════════════════════════════════════════════

  output$risk_bubble <- renderPlotly({
    d <- fd()
    plot_ly(
      d,
      x         = ~Frequency,
      y         = ~Percentage,
      size      = ~Percentage,
      color     = ~Risk_Level,
      colors    = risk_pal,
      type      = "scatter",
      mode      = "markers",
      sizes     = c(15, 60),
      marker    = list(opacity = 0.75, sizemode = "area",
                       line = list(color = "white", width = 1)),
      text      = ~paste0(
        "<b>", Group, "</b><br>",
        "Frequency: ", format(Frequency, big.mark = ","), "<br>",
        "Percentage: ", Percentage, "%<br>",
        "Risk: ", Risk_Level
      ),
      hoverinfo = "text"
    ) %>%
      layout(
        xaxis  = list(title = "Frequency", tickformat = ","),
        yaxis  = list(title = "Percentage (%)"),
        legend = list(title = list(text = "<b>Risk Level</b>"))
      ) %>%
      transparent_bg()
  })

  output$risk_bar <- renderPlotly({
    d <- fd() %>% count(Risk_Level)
    plot_ly(
      d,
      x         = ~Risk_Level,
      y         = ~n,
      type      = "bar",
      color     = ~Risk_Level,
      colors    = risk_pal,
      text      = ~n,
      textposition = "outside",
      hovertemplate = "<b>%{x}</b><br>Groups: %{y}<extra></extra>",
      showlegend = FALSE
    ) %>%
      layout(
        xaxis = list(title = "Risk Level",
                     categoryorder = "array",
                     categoryarray = c("Low","Moderate","High","Very High")),
        yaxis = list(title = "Number of Groups")
      ) %>%
      transparent_bg()
  })

  output$risk_heat <- renderPlotly({
    d <- fd()
    plot_ly(
      d,
      x         = ~as.character(Category),
      y         = ~reorder(Group, Percentage),
      z         = ~Percentage,
      type      = "heatmap",
      colorscale = list(
        c(0,   "#d5f5e3"),
        c(0.25, "#f9e79f"),
        c(0.60, "#f0b27a"),
        c(1,   "#c0392b")
      ),
      text      = ~paste0(Group, "<br>", Percentage, "%"),
      hoverinfo = "text",
      colorbar  = list(title = "Percentage (%)")
    ) %>%
      layout(
        xaxis = list(title = "Category", tickangle = -20),
        yaxis = list(title = "")
      ) %>%
      transparent_bg()
  })

  # ════════════════════════════════════════════════════════════
  # MODEL RESULTS
  # ════════════════════════════════════════════════════════════

  # — Decision Tree —
  output$dt_imp <- renderPlotly({
    imp <- dt_model$variable.importance
    if (is.null(imp)) {
      df_imp <- data.frame(Variable = "Percentage", Importance = 1)
    } else {
      df_imp <- data.frame(Variable   = names(imp),
                           Importance = as.numeric(imp))
    }
    plot_ly(
      df_imp,
      x    = ~Importance,
      y    = ~reorder(Variable, Importance),
      type = "bar", orientation = "h",
      marker = list(color = "#27ae60",
                    line  = list(color = "#1e8449", width = 1))
    ) %>%
      layout(xaxis = list(title = "Importance Score"),
             yaxis = list(title = "")) %>%
      transparent_bg()
  })

  output$dt_cm <- renderPrint({ dt_cm })

  # — Random Forest —
  output$rf_imp <- renderPlotly({
    imp_df <- as.data.frame(importance(rf_model))
    imp_df$Variable <- rownames(imp_df)
    plot_ly(
      imp_df,
      x    = ~MeanDecreaseGini,
      y    = ~reorder(Variable, MeanDecreaseGini),
      type = "bar", orientation = "h",
      marker = list(color = "#2980b9",
                    line  = list(color = "#1a5276", width = 1))
    ) %>%
      layout(xaxis = list(title = "Mean Decrease Gini"),
             yaxis = list(title = "")) %>%
      transparent_bg()
  })

  output$rf_cm <- renderPrint({ rf_cm })

  # — Linear Regression —
  output$lm_resid <- renderPlotly({
    df <- data.frame(
      Fitted    = fitted(lm_model),
      Residuals = residuals(lm_model),
      Group     = dat$Group
    )
    plot_ly(
      df,
      x    = ~Fitted,
      y    = ~Residuals,
      type = "scatter", mode = "markers",
      marker    = list(color = "#8e44ad", size = 10, opacity = 0.8),
      text      = ~Group,
      hoverinfo = "text+x+y"
    ) %>%
      add_lines(
        x    = range(df$Fitted),
        y    = c(0, 0),
        line = list(color = "red", dash = "dash", width = 2),
        showlegend = FALSE,
        inherit    = FALSE
      ) %>%
      layout(
        xaxis = list(title = "Fitted Values"),
        yaxis = list(title = "Residuals")
      ) %>%
      transparent_bg()
  })

  output$lm_summ <- renderPrint({ summary(lm_model) })

  # — Model Comparison —
  output$model_compare <- renderPlotly({
    df <- data.frame(
      Model    = c("Decision Tree", "Random Forest"),
      Accuracy = c(dt_acc, rf_acc)
    )
    plot_ly(
      df,
      x         = ~Model,
      y         = ~Accuracy,
      type      = "bar",
      color     = ~Model,
      colors    = c("#27ae60", "#2980b9"),
      text      = ~paste0(Accuracy, "%"),
      textposition = "outside",
      hovertemplate = "<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>",
      showlegend = FALSE
    ) %>%
      layout(
        yaxis = list(title = "Accuracy (%)", range = c(0, 115)),
        xaxis = list(title = "")
      ) %>%
      transparent_bg()
  })

  # ════════════════════════════════════════════════════════════
  # CLUSTERING
  # ════════════════════════════════════════════════════════════

  output$clust_plot <- renderPlotly({
    df <- as.data.frame(clust_scaled)
    df$Cluster <- dat$Cluster
    df$Group   <- dat$Group

    plot_ly(
      df,
      x         = ~Frequency,
      y         = ~Percentage,
      color     = ~Cluster,
      colors    = clust_pal,
      type      = "scatter",
      mode      = "markers",
      marker    = list(size = 13, opacity = 0.85,
                       line = list(color = "white", width = 1.5)),
      text      = ~paste0(
        "<b>", Group, "</b><br>",
        "Cluster: ", Cluster, "<br>",
        "Freq (scaled): ", round(Frequency, 2), "<br>",
        "Pct (scaled): ", round(Percentage, 2)
      ),
      hoverinfo = "text"
    ) %>%
      layout(
        xaxis  = list(title = "Frequency (z-scaled)"),
        yaxis  = list(title = "Percentage (z-scaled)"),
        legend = list(title = list(text = "<b>Cluster</b>"))
      ) %>%
      transparent_bg()
  })

  output$elbow_plot <- renderPlotly({
    set.seed(42)
    wss_vals <- sapply(1:6, function(k) {
      kmeans(clust_scaled, centers = k, nstart = 25)$tot.withinss
    })
    df <- data.frame(k = 1:6, WSS = wss_vals)

    plot_ly(df, x = ~k, y = ~WSS,
            type = "scatter", mode = "lines+markers",
            line   = list(color = "#2980b9", width = 2.5),
            marker = list(color = "#c0392b", size = 9)) %>%
      add_segments(
        x  = 3, xend = 3,
        y  = 0, yend = max(wss_vals),
        line = list(color = "orange", dash = "dot", width = 2),
        showlegend = FALSE,
        inherit    = FALSE
      ) %>%
      layout(
        xaxis = list(title = "Number of Clusters (k)", dtick = 1),
        yaxis = list(title = "Total Within-Cluster SS"),
        annotations = list(
          list(x = 3.05, y = max(wss_vals) * 0.9,
               text = "Optimal k = 3", showarrow = FALSE,
               font = list(color = "orange", size = 12))
        )
      ) %>%
      transparent_bg()
  })

  output$clust_table <- renderDT({
    dat %>%
      group_by(Cluster) %>%
      summarise(
        `Groups`           = paste(Group, collapse = "; "),
        `Avg Frequency`    = round(mean(Frequency), 0),
        `Avg Percentage`   = round(mean(Percentage), 2),
        `No. of Members`   = n(),
        `Burden Profile`   = case_when(
          mean(Percentage) >= 40 ~ "High Burden",
          mean(Percentage) >= 20 ~ "Moderate Burden",
          TRUE                   ~ "Low Burden"
        ),
        .groups = "drop"
      ) %>%
      datatable(rownames = FALSE,
                options = list(pageLength = 5, scrollX = TRUE)) %>%
      formatCurrency("Avg Frequency", currency = "", digits = 0) %>%
      formatRound("Avg Percentage", digits = 2)
  })

  # ════════════════════════════════════════════════════════════
  # DATA TABLE
  # ════════════════════════════════════════════════════════════

  output$full_table <- renderDT({
    dat %>%
      select(Category, Group, Frequency, Percentage,
             Risk_Level, High_Burden, Cluster) %>%
      datatable(
        rownames = FALSE,
        filter   = "top",
        options  = list(pageLength = 10, scrollX = TRUE),
        colnames = c("Category", "Group", "Frequency",
                     "Percentage (%)", "Risk Level",
                     "High Burden (1=Yes)", "Cluster")
      ) %>%
      formatRound("Percentage", digits = 2) %>%
      formatCurrency("Frequency", currency = "", digits = 0) %>%
      formatStyle(
        "Risk_Level",
        backgroundColor = styleEqual(
          c("Low", "Moderate", "High", "Very High"),
          c("#d5f5e3", "#fef9e7", "#fdebd0", "#fadbd8")
        )
      )
  })

  output$dl_csv <- downloadHandler(
    filename = function() {
      paste0("mindtrack_ph_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(
        dat %>% select(Category, Group, Frequency, Percentage,
                       Risk_Level, High_Burden, Cluster),
        file, row.names = FALSE
      )
    }
  )

} # end server

# ── Run ──────────────────────────────────────────────────────
shinyApp(ui = ui, server = server)

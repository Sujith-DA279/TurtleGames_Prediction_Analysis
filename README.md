# Turtle Games : Predicting Future Outcomes

### 📋 Project Brief
I conducted an in-depth analysis of Turtle Games customer data to improve overall sales performance through enhanced understanding of customer behavior and loyalty patterns. Using Python, R, and advanced statistical methods, I delivered actionable insights that can drive enhanced marketing effectiveness, customer retention, and targeted product offerings to increase revenue.

### ✅ Objectives
The analysis addressed key business needs of
- Understanding customer behavior and purchasing patterns.
- Identifying key factors influencing customer loyalty.
- Developing predictive models to forecast future sales.

They were addressed through multiple advanced data techniques: 
- **Customer Loyalty Drivers Analysis**: Applied **linear regression** modeling to identify and quantify key factors influencing customer loyalty
- **Customer Segmentation**: Implemented **decision tree** and **K-means clustering** to develop behavior-driven customer personas
- **Sentiment Analysis**: Utilized **NLP techniques (VADER, TextBlob)** on 2,000 customer reviews to assess product satisfaction
- **Integrated Recommendation System**: Combined quantitative insights with sentiment analysis to create segment-specific strategies

### 🎯 Key Findings & Business Impact
The analysis revealed significant opportunities for enhanced customer engagement through strategic loyalty program redesign and targeted marketing. Spending score and remuneration emerged as *primary loyalty drivers* (**correlation scores of 0.67 and 0.62**), while *five distinct customer segments* were identified with significantly different loyalty behaviors (ranging from 275 to 3988 average loyalty points).

**Data-Driven Recommendations:**
- Implement a tiered loyalty program with differentiated benefits based on five identified customer segments
- Deploy predictive loyalty model to anticipate customer behavior and tailor marketing approaches
- Enhance product quality and gameplay mechanics based on sentiment analysis findings
- Develop targeted strategies for "Occasional Affluents" with high income but inconsistent spending patterns
---

### Project Overview:
<div>
  <table>
    <tr align="center">
      <th align="center">⏱️ Duration</th>
      <th align="center">🏆 Grade</th>
      <th align="center">🛠️ Technologies</th>
      <th align="center">🧠 ML Algorithms</th>
      <th align="center">📊 Datasets</th>
    </tr>
    <tr>
      <td align="center"><b>6 Weeks</b></td>
      <td align="center"><b>97% (High Distinction)</b></td>
      <td align="center">
        <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
        <img src="https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white">
        <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
        <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
        <img src="https://img.shields.io/badge/NLTK-00B7FF?style=for-the-badge&logo=python&logoColor=white">
        <img src="https://img.shields.io/badge/statsmodels-4169E1?style=for-the-badge&logo=python&logoColor=white">
      </td>
      <td align="center">
        <img src="https://img.shields.io/badge/Linear_Regression-005571?style=for-the-badge">
        <img src="https://img.shields.io/badge/WLS_Regression-005571?style=for-the-badge">
        <img src="https://img.shields.io/badge/Decision_Trees-005571?style=for-the-badge">
        <img src="https://img.shields.io/badge/K--means_Clustering-005571?style=for-the-badge">
        <img src="https://img.shields.io/badge/VADER_Sentiment-005571?style=for-the-badge">
        <img src="https://img.shields.io/badge/GridSearchCV-005571?style=for-the-badge">
      </td>
      <td align="center">
        <b>2,000</b> customer reviews<br>
        <b>6</b> customer attributes
      </td>
    </tr>
  </table>
</div>

---

### Analytical Approach:
#### 1. Data Preparation & Engineering

| Process | Highlights & Technical Implementation |
|---------|---------|
| **Data Import & Validation** | • Loaded customer reviews dataset with purchase and demographic data<br>• Performed dataset validation with `.info()`, `.describe()`, `.shape()` functions<br>• Confirmed no missing values in the dataset with `.isna().sum()` |
| **Data Cleaning** | • Identified and **retained potential outliers** in loyalty points to preserve dataset integrity<br>• Flagged 2.5% of **age** data showing **potential misreporting**<br>• Applied exploratory visualizations to assess data distributions and relationships |
| **Feature Analysis** | • Analyzed correlation matrix between quantitative variables<br>• **Visualized** relationships between **loyalty points and potential predictor** variables<br>• Assessed categorical variables (gender, education) for their impact on loyalty points |
| **Feature Engineering** | • Applied **log and square root transformations** to address *skewness* in loyalty points<br>• **One-hot encoded** categorical features for decision tree analysis<br>• Created normalized features for clustering algorithms |

#### 2. Linear Regression Analysis

| Process | Technical Implementation |
|---------|---------|
| **Simple Linear Regression** | • Applied **OLS** function in statsmodel to analyze relationships between loyalty points and individual predictors<br>• Developed multiple iterations of models to identify optimal variable combinations<br>• Implemented model validation through training/test split |
| **Multiple Linear Regression** | • Created **comprehensive models** incorporating spending score, remuneration and age<br>• Applied statistical tests for **normality (Shapiro-Wilk) and heteroscedasticity (Breusch-Pagan)** <br>• Evaluated model performance using adjusted R-squared (84.0%) |
| **Model Optimization** | • Tested variable transformations (log, square root) to improve model performance<br>• Implemented **Weighted Least Squares regression** to address heteroscedasticity<br>• Validated final model with VIF assessment to confirm absence of multicollinearity |

<div align="center">
  <img src="https://github.com/user-attachments/assets/570fc746-b6f3-4736-b353-c972e01241e8" alt="OLS_LoyaltyPoints" width="80%"/>
  <p><em>Simple and MultiLinear Regressions plotted</em></p>
</div>

#### 3. Customer Segmentation Analysis

| Component | Technical Implementation |
|-----------|---------|
| **Decision Tree Analysis** | • Applied `DecisionTreeRegressor` from sklearn to identify key loyalty drivers<br>• Implemented feature importance analysis to identify spending score and remuneration as primary factors<br>• Used **GridSearchCV** to optimize hyperparameters (max_depth=3, max_leaf_nodes=40)<br>• Achieved 91.04% R-squared score on the full dataset |
| **K-means Clustering** | • Used elbow method and silhouette scores to determine optimal number of clusters (k=5)<br>• Applied **K-means clustering** to segment customers based on spending score and remuneration<br>• Achieved **silhouette score of 0.604** indicating good cluster separation<br>• Characterized clusters through descriptive statistics and visualization |
| **Persona Development** | • Created detailed customer personas based on cluster analysis<br>• Developed **cross-tabulation analysis** to examine loyalty point distribution across segments<br>• Generated cluster visualization using scatter plots with color-coding<br>• Formulated **segment-specific marketing recommendations** |

<div align="center">
  <img src="https://github.com/user-attachments/assets/16b1f12e-5adb-46b1-92fa-49522ac5e00d" alt="ClassificationKNN" width="90%"/>
  <p><em>Scatterplot of Loyalty points showing unique customer plots</em></p>
</div>

#### 4. Sentiment Analysis

| Process | Technical Implementation |
|---------|---------|
| **Text Preprocessing** | • Tokenized and cleaned review and summary text using **NLTK**<br>• Verified no relevant duplicates in the review dataset<br>• Performed a **manual verification** on 20 rows to identify most reliable text source (summary or full review text) |
| **Sentiment Classification** | • Implemented **VADER SIA model** for *sentiment polarity* scoring<br>• Applied **TextBlob** for *polarity and subjectivity* analysis<br>• Conducted comparative analysis between VADER (72.5% accuracy) and TextBlob (22.5% accuracy)<br>• Created sentiment distribution visualizations |
| **Word Analysis** | • Generated word clouds for positive and negative sentiment reviews<br>• Extracted top 20 frequent terms by sentiment polarity<br>• Identified key positive terms (play, great, love, fun) and negative terms (anger, disappointed, boring)<br>• Conducted **product-level sentiment aggregation** |

<div align="center">
  <img src="https://github.com/user-attachments/assets/03ccf4e3-c8a3-47d1-a7cc-7db8a57f4279" alt="SentimentPolarity" width="75%"/>
  <p><em>Distribution showing Sentiment Polarity and Subjectivity scores with a rule-based classification of sentiment scores</em></p>
</div>

---

### 🗝️ Key Insights:

**Loyalty Drivers Analysis**
- Identified **spending score** (0.67) and **remuneration** (0.62) as strongest predictors of customer loyalty
- Determined that age has a weak negative correlation (-0.04) with loyalty points but remains statistically significant
- Developed robust **Weighted Least Squares** regression model explaining **82.1%** of loyalty point variance
- Created **predictive formula: Loyalty Points = -1944.89 + 31.87 × Spending Score + 31.39 × Remuneration + 10.57 × Age**
> [!IMPORTANT]
  > **Value Discovery: Every 1-point increase in spending score correlates with a 31.87-point increase in loyalty points, while each £1,000 in income corresponds to a 31.39-point increase**

**Customer Segmentation**
- Identified **five distinct customer segments** with clear behavior patterns and loyalty characteristics
- Labelled *"Premium Buyers"* (**17.8%** of customers) averaging 3988 loyalty points with high income and frequent purchases
- Classified *"Occasional Affluents"* (**16.5%** of customers) with high income but relatively low loyalty engagement (912 points)
- Categorized the **largest segment** as *"Regular Customers"* (**38.7%**) with mid-range income and consistent spending
> [!IMPORTANT]
> **Value Discovery: "Bargain Hunters" (13.4% of customers) show high spending despite lower income, representing opportunity for targeted value-based marketing**


**Sentiment Analysis**
- Determined that **90%** of customer reviews express **positive sentiment** toward Turtle Games products
- Validated that VADER sentiment analyzer (72.5% accuracy) significantly outperformed TextBlob (22.5%) for game product reviews
- Identified key positive review themes around gameplay experience, physical components, and family enjoyment
- Located **negative sentiment** clusters around *product quality issues, age-appropriateness, and gameplay mechanics*
> [!IMPORTANT]
> **Value Discovery: Premium Buyers show highest positive sentiment (90%) with only 5% negative reviews, indicating strong correlation between satisfaction and loyalty**

<div align="center">
  <img src="https://github.com/user-attachments/assets/b2ec51de-179f-44b7-bafa-3282d4a3192a" alt="Sentiment_Product_CustomerTrends" width="80%"/>
  <p><em>Average Sentiment Polarity for different products and Customer Personas</em></p>
</div>

---
### 📊 Data-Driven Recommendations:

Based on comprehensive data analysis, the following strategic recommendations were developed to enhance Turtle Games' business performance:

**Loyalty Program Optimization**
- Implement a **tiered, gamified loyalty program** with distinct tiers matching the five customer segments
- Offer VIP "Platinum" benefits to Premium Buyers, including exclusive products and early access
- Create customized "Gold" tier for Regular Customers with bundle offers and upselling opportunities
- Develop entry-level tiers with targeted incentives for Basic Buyers, Bargain Hunters, and Occasional Affluents
- Deploy the **WLS** predictive model to **anticipate loyalty behaviors** and tailor marketing approaches

**Customer Engagement Strategies**
- Develop **targeted strategies** for "Occasional Affluents" who have high income but inconsistent spending
- Create personalized incentives based on spending score and income level analysis
- Investigate and address lower engagement among younger customers with **age-appropriate offerings**
- Monitor **segment migration** patterns to evaluate effectiveness of marketing interventions
- Implement clear upgrade paths between segments with tailored incentives

**Product Development & Quality Enhancement**
- Address **product quality issues** identified through sentiment analysis, particularly for budget offerings
- Enhance gameplay mechanics and instructions based on negative sentiment patterns
- Develop additional family-oriented products based on positive sentiment around family gameplay
- Implement **feature-based rating system** to gather more structured feedback on product qualities
- Invest in advanced NLP techniques fine-tuned for Turtle Games' review data to extract deeper insights

**Technical & Data Recommendations**
- Enhance data collection to include **customer ID** information to reduce potential duplicate analysis
- Improve age data collection to address the 2.5% potential misreporting identified
- Clarify loyalty points mechanics including expiration dates and redemption structures
- Add timestamp information to purchase and review data to enable trend analysis
- Develop comprehensive **product category mappings** to enable more granular product performance analysis
---


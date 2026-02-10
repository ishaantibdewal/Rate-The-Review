# Rate-The-Review
Authors: Chris Chen, Ishaan Tibdewal

## Overview
This is a data science project analyzing Food.com recipe data, where our focus is to predict recipe ratings from review text using TF-IDF and logistic regression. It also includes exploratory data analysis, missingness assessment, and hypothesis testing on recipe nutrition and user ratings from Food.com data.

---

## Introduction
Food connects people, cultures, and everyday experiences. Whether we’re trying a new recipe, sharing a meal with friends, or scrolling through online reviews to decide what to cook next, food is a huge part of how we explore, learn, and communicate. Recipe platforms like Food.com don’t just host ingredients and instructions, they capture thousands of real user experiences through reviews and ratings.

Both of us love cooking and discovering new dishes, and we noticed how often people rely on these reviews to decide what’s worth making. That sparked our interest in digging deeper: What actually makes a review helpful? Do certain ingredients or nutritional qualities influence ratings? Can we predict how well a recipe will be received just from the text people write?

To answer these questions, we analyzed two large datasets from Food.com—one containing over 80,000 recipes and another containing over 700,000 of user interactions. Our project explores this ecosystem from multiple angles: understanding missingness patterns, performing exploratory data analysis on recipe characteristics, testing hypotheses about nutrition and ratings, and building a predictive model using TF-IDF and logistic regression to classify review sentiment.

Through this process, we aim to better understand the relationship between how people talk about food and how they choose to rate it.

The first dataset that we examined, `recipes`, contains 83,782 rows and 12 columns with each row corresponding to an individual recipe. The 12 columns include:

| Column | Description |
| ----------- | ----------- |
| `name` | Recipe name|
| `id` | Recipe ID |
| `minutes`| Minutes to prepare recipe |
| `contributor_id` | User ID who submitted this recipe |
| `submitted`| Date recipe was submitted|
| `tags` | Food.com tags for recipe|
| `nutrition` | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for "percentage of daily value"|
| `n_steps` | Number of steps in the recipe|
| `steps` | Text for recipe steps, in order|
| `description`| User-provided description of the recipe|
| `ingredients` | Text for recipe ingredients|
| `n_ingredients` | Number of ingredients in the recipe|


The Second dataset that we examined, `interactions`, contains 731,927 rows and 5 columns with each row corresponding to an individual review from a user. The 5 columns include :

| Column | Description |
| ----------- | ----------- |
| `user_id` | User ID who made the review|
| `recipe_id`| Recipe ID|
| `date`| Date of review|
| `rating`| Rating given on a 1-5 scale|
| `review`| Review text given|


---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning
Before we dive right into exploring and working with our data, we first had to perform a few cleaning steps to prepare our datasets for better and more efficient analysis.

1. We first mereged our `recipes` dataset with the `interactions` dataset on `id` and `recipe_id`, respecitvely.
   - The resulting dataset has a row for each unique review corresponding to the recipe. The resulting dataframe is `recipes_interactions`.
2. We then replaced all ratings of 0 with `np.nan`.
   - This makes sense for our dataset since ratings are only a scale of 1-5, so ratings of 0 are treated as missing ratings instead. This avoids our ratings being biased downwards when performing certain operations.
3. We then added a new column, `avg_rating`, which consists of the average rating for the recipe in the corresponding row.

4. We split the `nutrition` column into multiple seperate columns.
   - The original `nutrition` column contained what looked like a list (of several nutritional values), but was actually a string/object. We created a function that turned the `nutrition` column into a list of floats, then turned each individual nutrition fact into its own column. This allows us to do a lot more exploratory analysis with each nutrition fact.
5. We added a `review_length` column. 
   - This column just contains the length of the review text. We planned on doing some analysis with the review column, so it made logical sense to add this column.
  
#### Resulting Dataframe
After these cleaning steps, we are left with a dataframe which has 234,429 rows and 26 columns. Here are the first 5 rows of our cleaned dataframe with a few columns:

| name                                 |     id |   minutes |   rating |   calories(#) |   review_length |
|:-------------------------------------|-------:|----------:|---------:|--------------:|----------------:|
| 1 brownies in the world    best ever | 333281 |        40 |        4 |         138.4 |             254 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |        5 |         595.1 |             336 |
| 412 broccoli casserole               | 306168 |        40 |        5 |         194.8 |             469 |
| 412 broccoli casserole               | 306168 |        40 |        5 |         194.8 |             162 |
| 412 broccoli casserole               | 306168 |        40 |        5 |         194.8 |             188 |

More specifically, the 26 columns are:
`['name',
 'id',
 'minutes',
 'contributor_id',
 'submitted',
 'tags',
 'nutrition',
 'n_steps',
 'steps',
 'description',
 'ingredients',
 'n_ingredients',
 'user_id',
 'recipe_id',
 'date',
 'rating',
 'review',
 'avg_rating',
 'calories(#)',
 'total fat(pdv)',
 'sugar(pdv)',
 'sodium(pdv)',
 'protein(pdv)',
 'saturated fat(pdv)',
 'carbohydrates(pdv)',
 'review_length']`

### Univarite Analysis
For this part of our data analysis, we wanted to look more into the actual review texts generated by users. More specifically, we wanted to look at the length of reviews and its distribution in this dataframe.

<iframe src="assets/review_len_dist.html" 
   width="800"
   height="500"
   frameborder="0"
></iframe>
We can see that our data is skewed to the right with some reviews having drastically longer lengths than the majority of our data. To make the visualization of this graph better, we only inluded 99% of the data since there were some extremely large outliers. We also plotted the median and mean character lengths, which were 249 and 289 respectively-another sign that our data is right-skewed.

### Bivariate Analysis
For this section of our data analysis, we wanted to look at the distribution and statistics of review text length, in regards to the rating categories. We have seen some statistics on review length in our dataset overall, but how does it look when broken down into each rating category?
<iframe
  src="assets/avg_review_length_by_rating.html"
  width="800"
  height="550"
  frameborder="0"
></iframe>
<iframe
  src="assets/review_length_by_rating_violin.html"
  width="700"
  height="550"
  frameborder="0"
></iframe>
Here are two different graphs, one showing the mean review length and the other showing the distribution of review lengths, both dependent on the individual rating categories. In both graphs, we can see that reviews who have a rating of 3 tend to have the highest mean and median review lengths (336 and 272 respectively), and those value decreases as you go up or down in rating. Overall, we can see that reviews with ratings that are on either end of the spectrum (1 and 5), tend to have lower average and median review lengths, and this does make some logical sense since the ratings may speak more for themselves here. Ratings which are more in the middle (2, 3, and 4) may leave more room for some sort of explanation to back that choice.

### Interesting Aggregates
For this section, we wanted to see how some columns and variables in our dataset relate to the length of reviews. More specifically, we wanted to break down the review lengths into different categories and examine its relationship with columns such as `n_steps`, `n_ingredients`, `calories(#)`, `rating`, and `minutes`. We used `pd.cut()` to categorize review lenghts into a few simple categories, then examined the mean values for the previously listed columns. Here is our resulting grouped dataframe, where we can see a couple more interesting relationships regarding review length.
Average Rating, Steps, Ingredients, Calories, and Cooking Time by Review Length:

| review_category   |   Average Rating |   Count |   Avg Steps |   Avg Ingredients |   Avg Calories |   Avg Cooking Time (min) |
|:------------------|-----------------:|--------:|------------:|------------------:|---------------:|-------------------------:|
| Short             |             4.69 |   25910 |       10.29 |              8.93 |         427.42 |                    78.75 |
| Medium            |             4.69 |   82008 |        9.35 |              8.69 |         396.27 |                    99.12 |
| Long              |             4.69 |   84868 |        9.94 |              9.2  |         412.34 |                    93.88 |
| Very Long         |             4.62 |   26550 |       11.36 |              9.9  |         469.39 |                   171.77 |

(In this table, Short is defined as reviews with length 0-100, Medium: 101-250, Long: 251-500, Very Long: 500+)

---

## Assessment of Missingness
The three columns with the most missingness in our data set were `rating`, `avg_rating`, and `description`. We decided to look further into the missingness mechanism(s) the `description` column.

### NMAR Analysis
We believe that the missingness of the `description` column in our `recipes_interactions` dataframe is most likely NMAR due to the fact that only authors who actaully really care about the recipe and making the recipe look better on the site will leave a description. Those who don't care as much about their recipe, or if it's a simple and straightforward recipe, may decide that having a description is not a priority and will leave it out. For example, if you're an upcoming cook or someone who wants to make a cookbook, you'd probably include a nice description with your recipe. In contrast, if you're someone who’s lazy but still wants to share a recipe, you'd just post it without a description.

### Missingness Dependency (MAR Analysis)
In this section, we wanted to continue looking at the `description` column and its missingness, however, we will be examining its missingness dependency with other columns in our dataset. Specifically, we are going to be looking at whether the missingness of `description` is dependent on `n_steps` (the number of steps in the recipe), and `n_ingredients` (the number of ingredients in the recipe).

##### `n_steps` and the missingness of `description`
- **Null Hypothesis**: Missingness of description does not depend on n_steps
- **Alternate Hypothesis**: Missingness of description does depend on n_step
- **Test Statistic**: Difference of mean n_steps with missing description and mean n_steps with description present
- **Significance Level**: 0.05

<iframe
  src="assets/n_steps_missingness_kde.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>
<iframe
  src="assets/n_steps_permutation_test.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

For this MAR test we ran a permutation test with 2000 repetitions, each time randomly shuffling the missingness of `description` and taking the difference of mean `n_steps` (description missing - not missing). Our **observed statistic** was **0.995**, which is indicated by the red line on the distribution graph. This left us with a **p-value** of **0.1**, which is greater than **0.05** (our significance level), thus we **fail to reject** our null hypothesis and conclude that the missingness of `description` **does not** depend on `n_steps`.

---

##### `n_ingredients` and the missingness of `description`
- **Null Hypothesis**: Missingness of description does not depend on n_ingredients
- **Alternate Hypothesis**: Missingness of description does depend on n_ingredients
- **Test Statistic**: Absolute difference of mean n_ingredients (missing description) and mean n_ingredients (not missing description)
- **Significance Level**: 0.05

<iframe
  src="assets/n_ingredients_missingness_kde.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>
<iframe
  src="assets/n_ingredients_permutation_test.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

For this MAR test we ran a permutation test with 2000 repetitions, each time randomly shuffling the missingness of `description` and taking the absolute difference of mean `n_ingredients` (description missing - not missing). Our **observed statistic** was **0.1.472**, which is indicated by the red line on the distribution graph. This left us with a **p-value** of **0.002**, which is less than **0.05** (our significance level), thus in this test we **eject** our null hypothesis and conclude that the missingness of `description` **does** actually depend on `n_ingredients`.

---

## Hypothesis Testing
In this section, we wanted to dive deeper into the review length trends that we have seen in our dataset. More sepcifically, we wanted to see if there was some difference or correlation between the length of a review and the rating correlated with that review. To do this, we first had to do some prep with our data. We classified reviews as long vs short, based on whether they were above or below the median review length of our dataset. We took the median rather than the mean due to the fact that our data was right-skewed, and median is more robust to outliers. This leaves us with two similar sized groups to test on. We then ran a **permutation test** to futher analyze this question.
- **Null Hypothesis**: Long and short reviews recieve the same ratings on average
- **Alternate Hypothesis**: Long reviews recieve lower ratings on average compared to short reviews
- **Test Statistic**: Difference in means of long reviews and short reviews
- **Significance Level**: 0.05

We used a permutation test because it requires no distributional assumptions, which is important given our right-skewed review length data. Based on our alternate hypothesis, we believe that longer reviews will have lower ratings on average. We think this because we feel that if a recipe has a high rating, the rating may speak for itself and the user does not have to provide a lenghty description. This, compared to lower ratings where people may feel more obligated to talk about why the recipe deserved the rating it got. The difference in means of long and short reviews make sense since we want to break down the groups into long and short reviews, and compare the average rating recieved by each group.

<iframe
  src="assets/hypothesis_test.html"
  width="800"
  height="520"
  frameborder="0"
></iframe>

After running our permutation test with 10,000 repetitions, each time shuffling the long vs short labels and taking the difference in means of (long - short), we ended up with an **observed statistic** of **-0.0128** (red dotted line above), with the distribution of differences plotted above as well. Our negative observed statistic tells us that shorter reviews did have slightly higher ratings on average. Our **p-value** of **0.0001** tells us that our observed statistic is something is rarely ever seen when the long and short review labels are randomly distributed. With this p-value, we are able to **reject** our null hypothesis at a 0.05 significant level and can conclude that in general longer reviews are associated with lower ratings, but only by a small margin.

---

## Framing a Prediction Problem
Like we mentioned earlier in this report, we wanted to dive deeper into the review texts that were curated by the reviewers. We wanted to look more into review sentiment, and see if we could gain a better understanding about the relationship between the way people talk about a recipe and how they review it.

Our prediction task is a binary classification problem: determining whether a user’s review will be highly rated (rating ≥ 4) or low rated (rating < 4) based on the review text and relevant recipe attributes. The target variable, `highly_rated`, is a simple 0/1 indicator that captures overall sentiment—1 for positive reviews and 0 for more negative ones. We chose this target because it aligns closely with our platform’s goal of understanding how the content of a review relates to user satisfaction and how that information can support better recommendations and quality insights for both users and creators.

Because the dataset is extremely imbalanced (around 94% highly rated vs. 6% low rated), we evaluate our models using F1-macro instead of accuracy. Accuracy would make a weak model look good—for example, predicting “highly rated” for every review already yields 94% accuracy while completely ignoring low-rated reviews. F1-macro avoids this problem by computing the F1-score for each class separately and then averaging them, giving minority-class performance equal weight. This is especially important for our application, since detecting low-rated reviews provides meaningful insight into dissatisfaction, potential recipe issues, and areas where the platform can improve user experience.

<iframe
  src="assets/target_distribution.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

---
## Baseline Model
For our baseline model, we use a **logistic regression classifier** trained on an 80/20 stratified split of the data. The model incorporates two features:

- `review`: the user’s written review text
- `n_steps`: a numeric variable representing the number of steps in the recipe

For preprocessing, we convert the `review` text into TF-IDF vectors using a TfidfVectorizer with English stop words removed. This transforms each review into a high-dimensional numerical representation that highlights informative words while down-weighting very common ones. The `n_steps` feature is already quantitative, so it passes through without additional encoding. Because the dataset is highly imbalanced, we set class_weight="balanced" to ensure that both classes contribute equally to the learning process.

Our primary evaluation metric is F1-macro, which treats the low-rated and highly-rated classes with equal importance. The model achieves an overall **F1-macro** score of **0.7203**, with class-specific F1 scores of 0.72 for low-rated reviews and 0.72 for highly-rated reviews. This balanced performance indicates that the model is capturing meaningful patterns from both the text and recipe characteristics rather than defaulting to the majority class. While the baseline performs reasonably well, the F1-macro score also shows clear room for improvement, especially in capturing more nuance within the review text.

<iframe
  src="assets/baseline_confusion_matrix.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

## Final Model
Our final model expands the feature set used in the baseline model by incorporating additional recipe characteristics that may influence how users rate their cooking experiences.

`review`

This is the user’s written review text and remains the most important feature, as it directly reflects sentiment and user experience. We vectorize the text using TF-IDF with English stop words removed, converting each review into a numerical representation that highlights informative terms in the corpus.

`n_steps`

This quantitative feature represents the number of steps in the recipe. Since it is already numeric, it is passed through unchanged. We retain it because recipe complexity may influence user satisfaction.

`calories(#)`

This numeric column captures the total calorie count of the recipe. We include it because nutritional content may relate to user preferences—some users may gravitate toward lighter dishes, while others may prefer richer, higher-calorie meals. We standardize this feature using StandardScaler to ensure comparability across recipes and prevent scale differences from affecting model performance.

`minutes`

This quantitative feature measures total cooking time. Our exploratory analysis showed that cooking duration varies across rating categories, suggesting it may help distinguish highly rated recipes from lower-rated ones. Because cooking times are highly right-skewed, we apply a QuantileTransformer with output_distribution='normal' to normalize the distribution and allow the model to learn more effectively from this feature.

#### Modeling Algorithm and Hyperparameter Tuning

We use logistic regression, consistent with our baseline model, and tune its hyperparameters using GridSearchCV with 5-fold cross-validation and F1-macro scoring. The grid search includes:
- **C**: [1, 2, 3, 4, 5, 6]
- **penalty**: ['l1', 'l2']
- **solver**: ['lbfgs'] for L2, and ['liblinear'] for L1
- **max_iter**: [1500, 1600, 1700]
  
(These options for the hyperparamters were chosen after a couple test runs and narrowing down the range of values.)

The best hyperparameters selected were:
C = 5, penalty = 'l2', solver = 'lbfgs', max_iter = 1500.

#### Performance
We evaluate the final model using F1-macro. The model achieves an F1-macro score of 0.7302, an improvement of +0.0086 over the baseline (0.7216). The F1-weighted score also increases from 0.9208 to 0.9270, reflecting overall performance gains across classes. These improvements indicate that incorporating nutritional and cooking time features, along with targeted hyperparameter tuning enhanced the model’s ability to capture meaningful patterns that predict whether a review will be highly rated. Despite the strong class imbalance, the model maintains balanced performance across both classes, showing relatively measurable improvements over the baseline.

<iframe
  src="assets/final_model_confusion_matrix.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

---

## Fariness Analysis
For our fairness analysis, we compared the model’s performance across two groups based on cooking time:
- **Quick recipes**: cooking time ≤ 35 minutes (23,288 reviews)
- **Slow recipes**: cooking time > 35 minutes (20,580 reviews)

We selected 35 minutes as the threshold because it is the median cooking time in the dataset, which creates two balanced groups and ensures that any observed differences are not caused by disproportionate sample sizes. We chose to evaluate precision for the positive class (`highly_rated` = 1), since precision measures how often the model’s predictions of “highly rated” are correct. High precision is especially important for maintaining user trust—mislabeling a low-rated recipe as highly rated could mislead users toward a disappointing cooking experience. We used precision parity as our fairness criterion. A fair model should achieve similar precision across both groups, meaning it should be equally accurate when predicting positive ratings for quick and slow recipes.

- **Null Hypothesis**: The model is fair. Its precision for quick and slow recipes is approximately the same, and any differences are due to random chance.
- **Alternative Hypothesis**: The model is unfair. Its precision for quick recipes differs from its precision for slow recipes. (We use a two-sided test, because fairness requires detecting performance differences in either direction.)
- **Test Statistic**: Difference in precision: Quick – Slow
- **Significance Level**: 0.05

<iframe
  src="assets/fairness_test.html"
  width="800"
  height="520"
  frameborder="0"
></iframe>

The **observed difference** was **0.0012** (quick: 0.9875, slow: 0.9863), which is extremely small. To evaluate whether this difference could occur by chance, we conducted a permutation test with 10,000 repetitions, shuffling the group labels to generate the null distribution. The resulting **p-value** was **0.3123**, which is well above our chosen significance level of 0.05. Thus, we **fail to reject** the null hypothesis. There is no statistically significant difference in precision between quick and slow recipes. This suggests that the model performs comparably across both groups and does not exhibit evidence of unfairness with respect to cooking time.
  

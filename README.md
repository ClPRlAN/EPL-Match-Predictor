# EPL Match Predictor

A machine learning project exploring whether historical match data and recent team performance can be used to predict English Premier League match winners.

## Project Overview

This project investigates whether a machine learning model can identify Premier League teams that are likely to win an individual match.

The problem is formulated as a **binary classification task**:

- `1` = Win
- `0` = Draw or Loss

The main objective is to optimize for **precision**, measuring how often the team's predicted wins are actually wins.

The project focuses on a realistic prediction setting where only information available before a match can be used. Because football data is inherently chronological, particular attention is given to preventing future information from leaking into the predictors.

## Dataset

The dataset contains Premier League match-level observations from two seasons:

- 2020–21
- 2021–22

Each match is represented from the perspective of the participating team. Therefore, a single fixture can appear twice: once for the home team and once for the away team.

The original dataset contains:

- **1,389 observations**
- **27 columns**
- **23 unique teams**

The 2020–21 season is complete with 760 team-level observations.

The 2021–22 season is incomplete, with data available through **April 25, 2022**, resulting in 629 observations.

A notable data-quality issue is that Liverpool is missing from the 2021–22 portion of the dataset despite participating in that season. The dataset was retained as provided rather than artificially reconstructing or imputing the missing matches.

## Data Quality and Preparation

The initial exploration examined:

- Dataset dimensions
- Column structure
- Data types
- Missing values
- Date coverage
- Season coverage
- Team representation
- Match-result distribution

Several columns were removed during preparation because they were either metadata, redundant for the modelling task, or unsuitable as predictors.

The match date was converted to a proper datetime format, while the remaining relevant variables were retained for feature engineering and modelling.

The final modelling dataset contains **1,389 observations and 19 columns** before rolling-feature processing.

## Target Variable

The target variable was created from the original match result:

- `W` → `1`
- `D` → `0`
- `L` → `0`

This creates a binary classification problem focused specifically on identifying winning outcomes.

The final target distribution is:

- **526 wins (37.9%)**
- **863 non-wins (62.1%)**

## Baseline Model

The baseline model is a **Random Forest Classifier** using pre-match contextual features:

- Venue
- Opponent
- Match time
- Day of week

Categorical variables were encoded numerically before modelling.

The data was split chronologically:

- **Training:** 2020–21 season — 760 observations
- **Testing:** 2021–22 season — 629 observations

No random train/test split was used. Randomly mixing matches from different points in time could allow future information to influence model development and would not represent a realistic forecasting scenario.

### Baseline Performance

The baseline Random Forest achieved:

**45.5% precision**

On the 629 test observations, the model predicted 110 wins.

## Recent-Form Feature Engineering

The improved model extends the baseline by incorporating historical recent-form information.

Rolling averages were calculated from each team's previous matches for variables including:

- Goals scored
- Goals conceded
- Shots
- Shots on target
- Shot distance
- Free kicks
- Penalties
- Penalty attempts

The rolling features were shifted so that the current match was excluded from its own historical calculations.

This ensures that every prediction uses only information that would have been available **before the match took place**.

This chronological approach is critical for avoiding target leakage.

## Improved Model

A second Random Forest model was trained using the recent-form features together with the modelling framework described above.

The rolling-feature dataset contained one missing historical observation for each rolling variable at the beginning of the relevant team sequence. These rows were removed before model training, leaving:

- **759 training observations**
- **629 test observations**
- **12 rolling predictors**

The improved Random Forest achieved:

**53.0% precision**

This represents an improvement of:

**+7.5 percentage points**

compared with the baseline model.

## Match-Level Analysis

Because the source dataset represents each fixture from both teams' perspectives, team-level predictions were reconstructed into individual fixtures.

Home and away observations were matched using:

- Match date
- Home team
- Away team

The final validation found:

- **298 matched fixtures**
- **0 duplicated fixture groups**

A selective prediction strategy was then applied. A fixture was retained only when exactly one of the two teams was predicted to win.

This produced:

- **105 predicted winners**
- **58 correct predictions**
- **47 incorrect predictions**
- **55.2% precision**

The filtered match-level strategy therefore achieved higher precision than the team-level improved model, but it made substantially fewer predictions.

## Results

| Model | Precision | Predictions |
|---|---:|---:|
| Baseline Random Forest | 45.5% | 629 |
| Improved Random Forest | 53.0% | 629 |
| Filtered Match-Level Predictions | 55.2% | 105 |

The addition of recent-form features improved precision from **45.5% to 53.0%**.

The selective match-level strategy achieved **55.2% precision**, but only for 105 fixtures.

Therefore, the results demonstrate an important **precision-versus-coverage trade-off**. The highest precision was obtained by making fewer predictions rather than by improving precision uniformly across all test observations.

## Key Findings

The results suggest that recent team performance contains useful predictive information beyond basic match context.

The main progression was:

**45.5% → 53.0% → 55.2%**

However, these figures represent different prediction strategies.

The baseline and improved models produced predictions for all 629 test observations, while the filtered match-level strategy produced predictions for only 105 fixtures.

The 55.2% result should therefore not be interpreted as simply meaning that the final strategy is universally better. It represents a more selective approach that prioritizes precision over prediction coverage.

## Limitations

### Incomplete Test Season

The 2021–22 season is incomplete in the source dataset, with observations available only through April 25, 2022.

### Missing Liverpool Data

Liverpool is missing from the 2021–22 portion of the dataset despite participating in the Premier League that season. This appears to be a source-data or scraping issue.

The missing matches were not reconstructed or artificially imputed.

### Limited Historical Coverage

Only two Premier League seasons are represented. This limits the amount of historical information available for training and makes broader generalization difficult to assess.

### Team-Level Dataset Structure

Each fixture is represented from both participating teams' perspectives. Additional processing was therefore required to evaluate predictions at the actual match level.

### Precision vs. Coverage

Precision was selected as the primary evaluation metric. However, a model can increase precision by making fewer predictions.

The filtered match-level strategy demonstrates this trade-off clearly.

### Feature Scope

The model uses information available before each match but does not include potentially informative external variables such as:

- Player injuries
- Suspensions
- Player availability
- Betting market information
- League position
- Squad strength
- External team ratings

## Future Work

Potential extensions include:

- Adding additional Premier League seasons
- Incorporating historical league position and points
- Adding opponent-strength ratings
- Including player availability and injury information
- Comparing additional machine learning algorithms
- Performing time-aware hyperparameter tuning
- Evaluating recall, F1-score, ROC-AUC and probability calibration
- Developing probability-based rather than binary predictions
- Building an interactive prediction application

## Project Structure

```text
EPL-Match-Predictor/
│
├── data/
│   └── matches.csv
│
├── notebook/
│   └── EPL_Match_Predictor.ipynb
│
├── src/
│
├── .gitignore
└── README.md
```

## Reproducibility

The project was developed in **Python using Google Colab**.

The notebook contains the complete workflow from data loading and exploration through feature engineering, model training, evaluation, and match-level analysis.

The modelling workflow preserves chronological ordering and explicitly prevents future match information from being used when generating historical predictors.

All reported results are calculated directly from the dataset and model predictions.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab
- GitHub

## Author

**Ciprian Loghin**

This project was developed as a machine learning portfolio project demonstrating data exploration, data quality assessment, feature engineering, temporal modelling, model evaluation, and interpretation using real-world sports data.

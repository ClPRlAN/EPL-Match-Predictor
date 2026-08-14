# EPL Match Predictor

A machine learning project exploring whether historical match data and recent team performance can be used to predict English Premier League match winners.

## Project Overview

Predicting football match outcomes is challenging because team performance changes over time and match results are influenced by many contextual factors.

This project investigates whether historical match information and recent team form can provide useful predictive signal for identifying whether an EPL team will win a match.

The problem is formulated as a **binary classification task**:

- `1` = Win
- `0` = Draw or Loss

The primary evaluation metric is **precision**, reflecting the project's focus on making fewer incorrect positive predictions.

## Objectives

The project aims to:

- Explore and audit the underlying EPL match dataset
- Identify data quality and completeness issues
- Build a chronological baseline machine learning model
- Engineer recent-form features using rolling historical statistics
- Avoid future information leakage throughout the modelling process
- Compare baseline and improved model performance
- Reconstruct predictions at the individual fixture level
- Evaluate the trade-off between prediction precision and prediction coverage
- Present the results and limitations honestly

## Dataset

The dataset contains **1,389 team-level match observations** covering two EPL seasons:

- 2020–21
- 2021–22

Each match appears from the perspective of each participating team.

The 2020–21 season is complete with 760 team-level observations. The 2021–22 season is incomplete, with data available through April 25, 2022.

The dataset therefore contains fewer observations than the theoretical total of 1,520 observations across two complete seasons.

A notable data-quality issue is that Liverpool is missing from the 2021–22 portion of the dataset. The project retains the original data rather than attempting to reconstruct or impute missing matches.

### Main Variables

The dataset contains information including:

- Match date and time
- Home/away venue
- Opponent
- Match result
- Goals scored and conceded
- Expected goals (`xG`, `xGA`)
- Possession
- Shots and shots on target
- Free kicks
- Penalties
- Team
- Season

## Methodology

### 1. Data Exploration and Quality Audit

The dataset was first examined without modifying the underlying observations.

The audit investigated:

- Dataset dimensions
- Data types
- Missing values
- Date coverage
- Season coverage
- Team representation
- Match-result distribution
- Completeness by team and season

This identified the incomplete 2021–22 season and the missing Liverpool observations.

### 2. Data Cleaning

The cleaning stage included:

- Converting match dates to `datetime`
- Removing columns that were not appropriate predictors
- Retaining the original observations rather than imputing missing matches
- Checking remaining missing values in modelling predictors

Post-match information such as goals scored and match statistics was not used as a predictor for the pre-match classification models.

### 3. Target Definition

The prediction target was defined as:

```text
Win = 1
Draw/Loss = 0

#Goofy Ikem


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Load dataset
matches = pd.read_csv("matches.csv", index_col=0)

# Convert date column to datetime
matches["date"] = pd.to_datetime(matches["date"])

# Convert categorical columns to numerical codes
matches["h/a"] = matches["venue"].astype("category").cat.codes  # Home (1) / Away (0)
matches["opp"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)  # Extracting hour
matches["day"] = matches["date"].dt.dayofweek  # Day of the week as number

# Target variable: Win (1) / Not Win (0)
matches["target"] = (matches["result"] == "W").astype(int)

# Define training and testing data
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] >= '2022-01-01']  # Fixed condition

# Define predictors
predictors = ["h/a", "opp", "hour", "day"]

# Initialize and train Random Forest model
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])

# Make predictions
preds = rf.predict(test[predictors])

# Evaluate model
accuracy = accuracy_score(test["target"], preds)
precision = precision_score(test["target"], preds)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")

# Create DataFrame to compare actual vs predicted
combined = pd.DataFrame({"actual": test["target"], "prediction": preds})
result_table = pd.crosstab(combined["actual"], combined["prediction"])
print(result_table)

# Group matches by team
if "team" in matches.columns:
    grouped_matches = matches.groupby("team")
    group = grouped_matches.get_group("Manchester United").sort_values("date")
else:
    print("Column 'team' not found in matches DataFrame.")

# Rolling averages function
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    return group.dropna(subset=new_cols)

# Define columns for rolling averages
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

# Apply rolling averages to all teams
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team').reset_index(drop=True)  # Reset index

# Function to make predictions
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] >= '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame({"actual": test["target"], "prediction": preds}, index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

# Generate predictions with rolling averages
combined, precision = make_predictions(matches_rolling, predictors + new_cols)

# Merge actual match results
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

# Dictionary to handle team name variations
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

# Map teams to their simplified names
combined["new_team"] = combined["team"].map(mapping)

# Merge home and away team predictions correctly
merged = combined.merge(
    combined, left_on=["date", "new_team"], right_on=["date", "opponent"], suffixes=("_home", "_away")
)

# Display final merged results
print(merged.head())

# Save results to CSV (optional)
merged.to_csv("predictions.csv", index=False)


#Code inspired by DataQuest and Chatgpt for advanced debugging






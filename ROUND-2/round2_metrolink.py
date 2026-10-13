import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set visualization theme
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# ==============================================================================
# 0. DATA VERIFICATION & MOCK GENERATOR (if files do not exist yet)
# ==============================================================================
def verify_or_generate_datasets():
    required_files = ['trips.csv', 'boardings.csv', 'routes.csv', 'calendar.csv', 'operations.csv']
    if all(os.path.exists(f) for f in required_files):
        print(">> Found existing CSV datasets. Loading files...")
        return

    print(">> Datasets not detected in current directory. Generating realistic sample files...")
    np.random.seed(42)
    n_trips = 1200
    
    # 1. routes.csv
    routes_df = pd.DataFrame({
        'route_id': ['R101', 'R102', 'R103', 'R104', 'R105'],
        'route_name': ['Corridor North', 'Airport Express', 'Downtown Loop', 'Suburban Link', 'Tech Park Shuttle'],
        'distance_km': [18.5, 32.0, 12.2, 24.8, 15.0],
        'stop_count': [22, 14, 18, 28, 16],
        'corridor': ['North-South', 'East Express', 'Central', 'West-East', 'South-East']
    })
    routes_df.to_csv('routes.csv', index=False)
    
    # 2. calendar.csv
    dates = pd.date_range(start='2026-09-01', periods=30, freq='D')
    calendar_df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'weekday': dates.strftime('%A'),
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
        'is_holiday': [1 if d.day in [5, 19] else 0 for d in dates],
        'event': ['Tech Expo' if d.day == 12 else 'Marathon' if d.day == 20 else 'Music Fest' if d.day == 27 else 'None' for d in dates]
    })
    calendar_df.to_csv('calendar.csv', index=False)
    
    # 3. trips.csv (with inconsistent route naming)
    raw_route_names = ['r101', ' R101 ', 'R102', 'r-102', 'R103', 'R104', 'R105']
    trips_df = pd.DataFrame({
        'trip_id': [f'T{1000+i}' for i in range(n_trips)],
        'route_id': np.random.choice(raw_route_names, size=n_trips),
        'date': np.random.choice(calendar_df['date'], size=n_trips),
        'start_time': [f"{np.random.randint(5, 23):02d}:{np.random.choice([0, 15, 30, 45]):02d}" for _ in range(n_trips)]
    })
    trips_df.to_csv('trips.csv', index=False)
    
    # 4. boardings.csv (with missing stop IDs, duplicate rows, negative boardings)
    stop_ids = [f'STP_{i}' for i in range(1, 31)] + [None, np.nan]
    boardings_list = []
    for t_id in trips_df['trip_id']:
        n_stops = np.random.randint(3, 8)
        for _ in range(n_stops):
            b_count = np.random.randint(-2, 45) # intentional negatives
            a_count = max(0, b_count + np.random.randint(-5, 5))
            boardings_list.append({
                'trip_id': t_id,
                'stop_id': np.random.choice(stop_ids),
                'boarding': b_count,
                'alighting': a_count
            })
    boardings_df = pd.DataFrame(boardings_list)
    # inject duplicates
    boardings_df = pd.concat([boardings_df, boardings_df.iloc[:40]], ignore_index=True)
    boardings_df.to_csv('boardings.csv', index=False)
    
    # 5. operations.csv (with missing delay data)
    ops_df = pd.DataFrame({
        'trip_id': trips_df['trip_id'],
        'scheduled_trips': 1,
        'completed_trips': np.random.choice([1, 1, 1, 0], p=[0.94, 0.03, 0.02, 0.01], size=n_trips),
        'delay_minutes': [np.nan if np.random.rand() < 0.08 else max(0, int(np.random.exponential(6))) for _ in range(n_trips)]
    })
    ops_df.to_csv('operations.csv', index=False)
    print(">> Sample files created successfully.\n")

verify_or_generate_datasets()

# ==============================================================================
# 1. LOAD DATASETS
# ==============================================================================
print("--- [1] Loading Raw Datasets ---")
trips = pd.read_csv('trips.csv')
boardings = pd.read_csv('boardings.csv')
routes = pd.read_csv('routes.csv')
calendar = pd.read_csv('calendar.csv')
operations = pd.read_csv('operations.csv')

print(f"trips: {trips.shape}, boardings: {boardings.shape}, routes: {routes.shape}, calendar: {calendar.shape}, operations: {operations.shape}")

# ==============================================================================
# 2. DATA PREPROCESSING & DATA QUALITY RESOLUTION
# ==============================================================================
print("\n--- [2] Handling Data Quality Conditions ---")

# (A) Handle Inconsistent Route Names
trips['route_id'] = trips['route_id'].astype(str).str.strip().str.upper().str.replace('-', '')

# (B) Handle Duplicate Boarding Records
dup_before = boardings.duplicated().sum()
boardings = boardings.drop_duplicates()
print(f"Removed {dup_before} duplicate boarding records.")

# (C) Handle Missing Stop IDs
missing_stops = boardings['stop_id'].isna().sum()
boardings['stop_id'] = boardings['stop_id'].fillna('UNKNOWN_STOP')
print(f"Imputed {missing_stops} missing stop IDs with 'UNKNOWN_STOP'.")

# (D) Handle Negative Passenger Counts
neg_boardings = (boardings['boarding'] < 0).sum()
boardings['boarding'] = boardings['boarding'].apply(lambda x: 0 if x < 0 else x)
boardings['alighting'] = boardings['alighting'].apply(lambda x: 0 if x < 0 else x)
print(f"Rectified {neg_boardings} negative passenger counts to 0.")

# (E) Handle Incomplete Delay Data
missing_delays = operations['delay_minutes'].isna().sum()
operations['delay_minutes'] = operations['delay_minutes'].fillna(operations['delay_minutes'].median())
print(f"Imputed {missing_delays} missing delay entries with median delay.")

# ==============================================================================
# 3. MERGING & AGGREGATING DATA
# ==============================================================================
print("\n--- [3] Aggregating & Merging Data ---")

# Aggregate boardings per trip
trip_boardings = boardings.groupby('trip_id').agg(
    total_boardings=('boarding', 'sum'),
    total_alightings=('alighting', 'sum'),
    stops_serviced=('stop_id', 'nunique')
).reset_index()

# Combine all datasets at the Trip level
master_df = trips.merge(trip_boardings, on='trip_id', how='left')
master_df = master_df.merge(routes, on='route_id', how='left')
master_df = master_df.merge(calendar, on='date', how='left')
master_df = master_df.merge(operations, on='trip_id', how='left')

# Fill any residual nulls
master_df['total_boardings'] = master_df['total_boardings'].fillna(0)
master_df['delay_minutes'] = master_df['delay_minutes'].fillna(0)
master_df['distance_km'] = master_df['distance_km'].fillna(master_df['distance_km'].mean())
master_df['stop_count'] = master_df['stop_count'].fillna(master_df['stop_count'].median())

# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================
print("\n--- [4] Performing Feature Engineering ---")

# Extract Time Features
master_df['hour'] = pd.to_datetime(master_df['start_time'], format='%H:%M').dt.hour

# Create Rush Hour Indicator (Morning Peak: 8-10, Evening Peak: 17-19)
master_df['is_rush_hour'] = master_df['hour'].apply(lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 19) else 0)

# Create Event Indicator
master_df['has_special_event'] = master_df['event'].apply(lambda x: 0 if str(x).strip().lower() in ['none', 'nan', ''] else 1)

# Passenger Load per km
master_df['passenger_density'] = (master_df['total_boardings'] / master_df['distance_km']).round(2)

print(f"Engineered master dataset shape: {master_df.shape}")

# Save master dataset for Round 3 (Excel)
master_df.to_csv('metrolink_cleaned_master.csv', index=False)
print(">> Saved 'metrolink_cleaned_master.csv' for Round 3 Excel analysis.")

# ==============================================================================
# 5. EXPLORATORY DATA ANALYSIS (Answering the 6 Use-Case Questions)
# ==============================================================================
print("\n--- [5] Generating EDA Visualizations ---")

# Plot 1: Which routes carry the most passengers?
plt.figure(figsize=(9, 4))
route_pax = master_df.groupby('route_id')['total_boardings'].sum().sort_values(ascending=False)
sns.barplot(x=route_pax.index, y=route_pax.values, palette='Blues_r')
plt.title('Total Passenger Demand by Route')
plt.xlabel('Route ID')
plt.ylabel('Total Boardings')
plt.tight_layout()
plt.savefig('eda_1_route_demand.png')
plt.close()

# Plot 2: What are peak travel periods?
plt.figure(figsize=(9, 4))
hourly_pax = master_df.groupby('hour')['total_boardings'].mean()
sns.lineplot(x=hourly_pax.index, y=hourly_pax.values, marker='o', color='crimson')
plt.title('Average Passenger Boardings by Hour of Day')
plt.xlabel('Hour (24-hr)')
plt.ylabel('Average Boardings per Trip')
plt.xticks(range(5, 23))
plt.tight_layout()
plt.savefig('eda_2_peak_hours.png')
plt.close()

# Plot 3: Which stops have high boarding demand?
plt.figure(figsize=(9, 4))
top_stops = boardings[boardings['stop_id'] != 'UNKNOWN_STOP'].groupby('stop_id')['boarding'].sum().nlargest(10)
sns.barplot(x=top_stops.values, y=top_stops.index, palette='crest')
plt.title('Top 10 Busiest Bus Stops by Total Boardings')
plt.xlabel('Total Boardings')
plt.ylabel('Stop ID')
plt.tight_layout()
plt.savefig('eda_3_top_stops.png')
plt.close()

# Plot 4: How does weekend demand differ?
plt.figure(figsize=(6, 4))
sns.boxplot(x='is_weekend', y='total_boardings', data=master_df, palette='Set2')
plt.xticks([0, 1], ['Weekday', 'Weekend'])
plt.title('Ridership Comparison: Weekday vs Weekend')
plt.xlabel('Day Type')
plt.ylabel('Trip Total Boardings')
plt.tight_layout()
plt.savefig('eda_4_weekend_demand.png')
plt.close()

# Plot 5: Which routes have repeated delays?
plt.figure(figsize=(9, 4))
route_delay = master_df.groupby('route_id')['delay_minutes'].mean().sort_values(ascending=False)
sns.barplot(x=route_delay.index, y=route_delay.values, palette='Reds_r')
plt.title('Average Operational Delay by Route (Minutes)')
plt.xlabel('Route ID')
plt.ylabel('Average Delay (min)')
plt.tight_layout()
plt.savefig('eda_5_route_delays.png')
plt.close()

# Plot 6: How do events affect ridership?
plt.figure(figsize=(6, 4))
event_impact = master_df.groupby('has_special_event')['total_boardings'].mean()
sns.barplot(x=['No Event', 'Special Event'], y=event_impact.values, palette='magma')
plt.title('Impact of Events on Average Ridership')
plt.ylabel('Average Boardings per Trip')
plt.tight_layout()
plt.savefig('eda_6_event_impact.png')
plt.close()

print(">> Saved all 6 EDA charts (eda_1 to eda_6.png).")

# ==============================================================================
# 6. MACHINE LEARNING: PASSENGER DEMAND PREDICTION
# ==============================================================================
print("\n--- [6] Training Machine Learning Model ---")

# Define Features and Target
feature_cols = ['distance_km', 'stop_count', 'hour', 'is_rush_hour', 
                'is_weekend', 'is_holiday', 'has_special_event', 'delay_minutes']
target_col = 'total_boardings'

X = master_df[feature_cols]
y = master_df[target_col]

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions & Metrics
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"ML Algorithm: Random Forest Regressor")
print(f"Evaluation Metrics:")
print(f"  - RMSE : {rmse:.2f}")
print(f"  - MAE  : {mae:.2f}")
print(f"  - R²   : {r2:.4f}")

# Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)
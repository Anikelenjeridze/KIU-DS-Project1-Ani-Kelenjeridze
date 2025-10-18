# Project 1: Python & NumPy Fundamentals
# Introduction to Data Science with Python
# Student Name: [Ani kelenjeridze]
# Student ID: [388883399]
# Submission Date: October 19, 2025
# Honor Code: I certify that this work is my own and I have not plagiarized

import numpy as np
from typing import List, Dict, Tuple


# TASK 1: Python Data Structures & Control Flow


print("=" * 70)
print("TASK 1: STUDENT PERFORMANCE ANALYSIS")
print("=" * 70)

# Part A: Data Structure Creation
students = {
    "S001": {"name": "Emma Wilson", "scores": [78, 85, 92, 88], "attendance": 28},
    "S002": {"name": "James Chen", "scores": [92, 95, 89, 94], "attendance": 29},
    "S003": {"name": "Sofia Martinez", "scores": [65, 70, 68, 72], "attendance": 25},
    "S004": {"name": "Marcus Johnson", "scores": [88, 86, 90, 87], "attendance": 27},
    "S005": {"name": "Olivia Brown", "scores": [95, 98, 96, 97], "attendance": 30},
    "S006": {"name": "Liam Davis", "scores": [72, 75, 78, 74], "attendance": 26},
    "S007": {"name": "Ava Anderson", "scores": [58, 62, 55, 60], "attendance": 20},
    "S008": {"name": "Noah Taylor", "scores": [81, 79, 83, 85], "attendance": 28},
    "S009": {"name": "Isabella Moore", "scores": [45, 50, 48, 52], "attendance": 18},
    "S010": {"name": "Ethan White", "scores": [90, 88, 92, 91], "attendance": 29}
}

# Part B: Function Implementation

def calculate_average(scores: list) -> float:
    
    return round(sum(scores) / len(scores), 2)


def assign_grade(average: float) -> str:
    
    if average >= 90:
        return "A"
    elif average >= 80:
        return "B"
    elif average >= 70:
        return "C"
    elif average >= 60:
        return "D"
    else:
        return "F"


def check_eligibility(student_dict: dict, total_classes: int) -> tuple:
    
    avg = calculate_average(student_dict["scores"])
    attendance_rate = student_dict["attendance"] / total_classes
    
    if avg >= 60 and attendance_rate >= 0.75:
        return (True, f"Passed with {avg} average and {attendance_rate*100:.1f}% attendance")
    elif avg < 60 and attendance_rate < 0.75:
        return (False, f"Low average ({avg}) and insufficient attendance ({student_dict['attendance']}/{total_classes})")
    elif avg < 60:
        return (False, f"Low average ({avg})")
    else:
        return (False, f"Insufficient attendance ({student_dict['attendance']}/{total_classes})")


def find_top_performers(students: dict, n: int) -> list:
   
    averages = []
    for student_id, info in students.items():
        avg = calculate_average(info["scores"])
        averages.append((student_id, avg))
    
    # Sort by average in descending order
    averages.sort(key=lambda x: x[1], reverse=True)
    return averages[:n]


def generate_report(students: dict) -> dict:
    """
    Generate comprehensive course statistics.
    
    Parameters:
        students (dict): Dictionary of all students
    
    Returns:
        dict: Dictionary containing various course statistics
    """
    total_classes = 30
    passed = 0
    failed = 0
    all_averages = []
    total_attendance = 0
    
    for student_id, info in students.items():
        avg = calculate_average(info["scores"])
        all_averages.append(avg)
        total_attendance += info["attendance"]
        
        eligibility, _ = check_eligibility(info, total_classes)
        if eligibility:
            passed += 1
        else:
            failed += 1
    
    return {
        "total_students": len(students),
        "passed_count": passed,
        "failed_count": failed,
        "class_average": round(sum(all_averages) / len(all_averages), 2),
        "highest_score": max(all_averages),
        "lowest_score": min(all_averages),
        "average_attendance_rate": round((total_attendance / (len(students) * total_classes)) * 100, 2)
    }


# Part C: Analysis & Output

# Generate course statistics
report = generate_report(students)

print("\n=== COURSE STATISTICS ===")
print(f"Total Students: {report['total_students']}")
print(f"Passed: {report['passed_count']} ({report['passed_count']/report['total_students']*100:.1f}%)")
print(f"Failed: {report['failed_count']} ({report['failed_count']/report['total_students']*100:.1f}%)")
print(f"Class Average: {report['class_average']}")
print(f"Average Attendance Rate: {report['average_attendance_rate']}%")

# Top 5 performers
print("\n=== TOP 5 PERFORMERS ===")
top_performers = find_top_performers(students, 5)
for i, (student_id, avg) in enumerate(top_performers, 1):
    name = students[student_id]["name"]
    grade = assign_grade(avg)
    print(f"{i}. {student_id} - {name}: {avg} ({grade})")

# Students who failed
print("\n=== STUDENTS WHO FAILED ===")
total_classes = 30
for student_id, info in students.items():
    eligibility, reason = check_eligibility(info, total_classes)
    if not eligibility:
        print(f"{student_id} - {reason}")

# Grade distribution
print("\n=== GRADE DISTRIBUTION ===")
grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
for student_id, info in students.items():
    avg = calculate_average(info["scores"])
    grade = assign_grade(avg)
    grade_counts[grade] += 1

for grade, count in grade_counts.items():
    print(f"{grade}: {count} students")



# TASK 2: NumPy Arrays & Operations


print("\n" + "=" * 70)
print("TASK 2: NUMPY ARRAYS & OPERATIONS")
print("=" * 70)

# Set random seed for reproducibility
np.random.seed(42)

# Part A: Array Creation & Exploration

print("\n--- Part A: Array Creation ---")

# 1. Temperature Data
temperatures = np.random.uniform(-10.0, 40.0, size=(365, 5))
print(f"\nTemperature Array:")
print(f"Shape: {temperatures.shape}")
print(f"Dimensions: {temperatures.ndim}")
print(f"Data Type: {temperatures.dtype}")
print(f"Size: {temperatures.size}")

# 2. Sales Matrix
sales = np.random.randint(1000, 5001, size=(12, 4))
print(f"\nSales Matrix:")
print(f"Shape: {sales.shape}")
print(sales)

# 3. Special Arrays
identity = np.eye(5)
print(f"\nIdentity Matrix (5x5):")
print(identity)

evenly_spaced = np.linspace(0, 100, 50)
print(f"\nEvenly Spaced Array (50 values from 0 to 100):")
print(f"First 10 values: {evenly_spaced[:10]}")

# Part B: Array Manipulation & Indexing

print("\n--- Part B: Array Manipulation ---")

# 1.              Basic Slicing
january_data = temperatures[:31, :]
print(f"\nJanuary Data Shape: {january_data.shape}")
print(f"January Average Temp per City: {np.mean(january_data, axis=0).round(2)}")

summer_data = temperatures[151:243, :]
print(f"\nSummer Data Shape: {summer_data.shape}")
print(f"Summer Average Temp per City: {np.mean(summer_data, axis=0).round(2)}")

weekend_data = temperatures[4::7, :]
print(f"\nWeekend Data Shape: {weekend_data.shape}")

# 2.       Boolean Indexing
hot_days = np.any(temperatures > 35, axis=1)
print(f"\nDays with temperature > 35°C: {np.sum(hot_days)}")

freezing_days = np.sum(temperatures < 0, axis=0)
print(f"Freezing days per city: {freezing_days}")

comfortable_mask = (temperatures >= 15) & (temperatures <= 25)
comfortable_percentage = (np.sum(comfortable_mask) / temperatures.size) * 100
print(f"Percentage of comfortable readings: {comfortable_percentage:.2f}%")


temperatures_cleaned = temperatures.copy()
extreme_cold_count = np.sum(temperatures_cleaned < -5)
temperatures_cleaned[temperatures_cleaned < -5] = -5
print(f"Extreme cold values replaced: {extreme_cold_count}")

# 3.                 Fancy Indexing
specific_days = temperatures[[0, 100, 200, 300, 364], :]
print(f"\nSpecific Days Data Shape: {specific_days.shape}")
print(f"Temperatures on days [0, 100, 200, 300, 364]:\n{specific_days.round(2)}")

# Quarterly averages
q1 = temperatures[:91, :].mean(axis=0)
q2 = temperatures[91:182, :].mean(axis=0)
q3 = temperatures[182:273, :].mean(axis=0)
q4 = temperatures[273:, :].mean(axis=0)
quarterly = np.array([q1, q2, q3, q4])
print(f"\nQuarterly Averages (4 quarters x 5 cities):\n{quarterly.round(2)}")


annual_averages = temperatures.mean(axis=0)
sorted_indices = np.argsort(annual_averages)[::-1]
temperatures_sorted = temperatures[:, sorted_indices]
print(f"\nAnnual averages (descending): {annual_averages[sorted_indices].round(2)}")



print("\n--- Part C: Mathematical Operations ---")

# 1. Temperature Analysis
print("\nTemperature Statistics per City:")
for i in range(5):
    city_temps = temperatures[:, i]
    print(f"City {i+1}:")
    print(f"  Mean: {np.mean(city_temps):.2f}°C")
    print(f"  Median: {np.median(city_temps):.2f}°C")
    print(f"  Std Dev: {np.std(city_temps):.2f}°C")





hottest_idx = np.unravel_index(np.argmax(temperatures), temperatures.shape)
coldest_idx = np.unravel_index(np.argmin(temperatures), temperatures.shape)
print(f"\nHottest day: Day {hottest_idx[0]+1}, City {hottest_idx[1]+1}, Temp: {temperatures[hottest_idx]:.2f}°C")
print(f"Coldest day: Day {coldest_idx[0]+1}, City {coldest_idx[1]+1}, Temp: {temperatures[coldest_idx]:.2f}°C")





temp_ranges = temperatures.max(axis=0) - temperatures.min(axis=0)
print(f"\nTemperature Range per City: {temp_ranges.round(2)}")


correlation_matrix = np.corrcoef(temperatures.T)
print(f"\nCorrelation Matrix between Cities:\n{correlation_matrix.round(3)}")

# 2. Sales Analysis
print("\n--- Sales Analysis ---")
total_per_category = sales.sum(axis=0)
print(f"Total Sales per Category: {total_per_category}")

avg_monthly_per_category = sales.mean(axis=0)
print(f"Average Monthly Sales per Category: {avg_monthly_per_category.round(2)}")

best_month = np.argmax(sales.sum(axis=1))
print(f"Best Performing Month: Month {best_month + 1} (Total: {sales.sum(axis=1)[best_month]})")

best_category = np.argmax(total_per_category)
print(f"Best Performing Category: Category {best_category + 1} (Total: {total_per_category[best_category]})")

# 3. Advanced Computations
print("\n--- Advanced Computations ---")


moving_avg = np.array([temperatures[i:i+7, :].mean(axis=0) for i in range(len(temperatures) - 6)])
print(f"7-day Moving Average Shape: {moving_avg.shape}")
print(f"First week average: {moving_avg[0].round(2)}")


z_scores = np.zeros_like(temperatures)
for i in range(5):
    city_temps = temperatures[:, i]
    z_scores[:, i] = (city_temps - np.mean(city_temps)) / np.std(city_temps)
print(f"\nZ-scores calculated. Sample (Day 1): {z_scores[0].round(2)}")


print("\nPercentiles per City:")
for i in range(5):
    p25 = np.percentile(temperatures[:, i], 25)
    p50 = np.percentile(temperatures[:, i], 50)
    p75 = np.percentile(temperatures[:, i], 75)
    print(f"City {i+1}: 25th={p25:.2f}, 50th={p50:.2f}, 75th={p75:.2f}")



# TASK 3: Applied Data Analysis


print("\n" + "=" * 70)
print("TASK 3: FITNESS TRACKING DATA ANALYSIS")
print("=" * 70)

np.random.seed(42)



# Part A: Data Generation & Preparation

print("\n--- Part A: Data Generation ---")

# Create base dataset (100 users x 90 days x 4 metrics)
n_users = 100
n_days = 90
n_metrics = 4

# Generate realistic data with some variation
daily_steps = np.random.randint(2000, 15001, size=(n_users, n_days))
calories = np.random.randint(1500, 3501, size=(n_users, n_days))
active_minutes = np.random.randint(20, 181, size=(n_users, n_days))
avg_heart_rate = np.random.randint(60, 121, size=(n_users, n_days))

# Combine into single array
fitness_data = np.stack([daily_steps, calories, active_minutes, avg_heart_rate], axis=2)
print(f"Fitness Data Shape: {fitness_data.shape}")
print(f"(users x days x metrics)")

# Introduce missing values (5% NaN)
nan_mask = np.random.random(fitness_data.shape) < 0.05
fitness_data = fitness_data.astype(float)
fitness_data[nan_mask] = np.nan
print(f"\nMissing values introduced: {np.sum(np.isnan(fitness_data))}")

# Introduce outliers (2%)
outlier_mask = np.random.random(fitness_data.shape) < 0.02
outlier_values = np.random.choice([0, 50000, 200, 180], size=np.sum(outlier_mask))
fitness_data_flat = fitness_data.flatten()
fitness_data_flat[outlier_mask.flatten()] = outlier_values
fitness_data = fitness_data_flat.reshape(fitness_data.shape)
print(f"Outliers introduced: {np.sum(outlier_mask)}")

# User metadata
user_ids = np.arange(1, n_users + 1)
ages = np.random.randint(18, 71, size=n_users)
genders = np.random.randint(0, 2, size=n_users)  # 0=Female, 1=Male
user_metadata = np.column_stack([user_ids, ages, genders])
print(f"\nUser Metadata Shape: {user_metadata.shape}")








# Part B: Data Cleaning & Validation

print("\n--- Part B: Data Cleaning ---")

def handle_missing(data):
    """Replace NaN values with column mean."""
    result = data.copy()
    for metric_idx in range(data.shape[2]):
        metric_data = data[:, :, metric_idx]
        col_mean = np.nanmean(metric_data)
        mask = np.isnan(metric_data)
        result[:, :, metric_idx][mask] = col_mean
    return result

def remove_outliers(data, metric_index):
    """Remove outliers using IQR method."""
    result = data.copy()
    metric_data = result[:, :, metric_index].flatten()
    
    q1 = np.percentile(metric_data[~np.isnan(metric_data)], 25)
    q3 = np.percentile(metric_data[~np.isnan(metric_data)], 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_mask = (metric_data < lower_bound) | (metric_data > upper_bound)
    median_val = np.median(metric_data[~np.isnan(metric_data)])
    metric_data[outlier_mask] = median_val
    
    result[:, :, metric_index] = metric_data.reshape(result.shape[0], result.shape[1])
    return result, np.sum(outlier_mask)

# Apply cleaning pipeline
print("Removing outliers...")
outliers_removed = []
for metric_idx in range(4):
    fitness_data, count = remove_outliers(fitness_data, metric_idx)
    outliers_removed.append(count)
    print(f"  Metric {metric_idx}: {count} outliers removed")

print("\nHandling missing values...")
fitness_data = handle_missing(fitness_data)
print(f"Remaining NaN values: {np.sum(np.isnan(fitness_data))}")

# Part C: Comprehensive Analysis

print("\n--- Part C: Analysis ---")

# Extract cleaned metrics
daily_steps_clean = fitness_data[:, :, 0]
calories_clean = fitness_data[:, :, 1]
active_minutes_clean = fitness_data[:, :, 2]
heart_rate_clean = fitness_data[:, :, 3]

# 1. User Behavior Patterns
print("\n1. USER BEHAVIOR PATTERNS")

user_averages = fitness_data.mean(axis=1)
print(f"User Averages Shape: {user_averages.shape}")

# Calculate z-scores for all metrics
z_steps = (user_averages[:, 0] - user_averages[:, 0].mean()) / user_averages[:, 0].std()
z_calories = (user_averages[:, 1] - user_averages[:, 1].mean()) / user_averages[:, 1].std()
z_minutes = (user_averages[:, 2] - user_averages[:, 2].mean()) / user_averages[:, 2].std()
z_heart = (user_averages[:, 3] - user_averages[:, 3].mean()) / user_averages[:, 3].std()

combined_z = z_steps + z_calories + z_minutes - z_heart  # Lower heart rate is better
top_10_indices = np.argsort(combined_z)[-10:][::-1]
print(f"\nTop 10 Most Active Users (IDs): {user_ids[top_10_indices]}")

# Most consistent users
user_stds = fitness_data.std(axis=1)
consistency_score = user_stds.mean(axis=1)
most_consistent = np.argsort(consistency_score)[:5]
print(f"Most Consistent Users (IDs): {user_ids[most_consistent]}")

# Activity level classification
steps_percentile_25 = np.percentile(daily_steps_clean.mean(axis=1), 25)
steps_percentile_75 = np.percentile(daily_steps_clean.mean(axis=1), 75)
activity_levels = np.zeros(n_users, dtype=int)  # 0=Low, 1=Medium, 2=High
user_avg_steps = daily_steps_clean.mean(axis=1)
activity_levels[user_avg_steps < steps_percentile_25] = 0
activity_levels[(user_avg_steps >= steps_percentile_25) & (user_avg_steps <= steps_percentile_75)] = 1
activity_levels[user_avg_steps > steps_percentile_75] = 2

print(f"\nActivity Level Distribution:")
print(f"  Low: {np.sum(activity_levels == 0)} users")
print(f"  Medium: {np.sum(activity_levels == 1)} users")
print(f"  High: {np.sum(activity_levels == 2)} users")

# 2. Temporal Trends
print("\n2. TEMPORAL TRENDS")

# Population-wide 7-day rolling average
population_daily_avg = fitness_data.mean(axis=0)
rolling_7day = np.array([population_daily_avg[i:i+7].mean(axis=0) for i in range(n_days - 6)])
print(f"7-day Rolling Average Shape: {rolling_7day.shape}")
print(f"Week 1 avg steps: {rolling_7day[0, 0]:.0f}")
print(f"Week 12 avg steps: {rolling_7day[-1, 0]:.0f}")

# Weekly patterns (day of week)
day_of_week_avg = np.array([population_daily_avg[i::7].mean(axis=0) for i in range(7)])
print(f"\nAverage Steps by Day of Week:")
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i, day in enumerate(days):
    print(f"  {day}: {day_of_week_avg[i, 0]:.0f} steps")

# Trend analysis (comparing first vs last 30 days)
first_month = population_daily_avg[:30, 0].mean()
last_month = population_daily_avg[-30:, 0].mean()
trend_change = ((last_month - first_month) / first_month) * 100
print(f"\nActivity Trend: {trend_change:+.2f}% change from first to last month")

# 3. Correlations & Insights
print("\n3. CORRELATIONS & INSIGHTS")

# Correlation matrix for all metrics
user_avg_metrics = fitness_data.mean(axis=1)
correlation_metrics = np.corrcoef(user_avg_metrics.T)
print(f"\nCorrelation Matrix:")
metric_names = ['Steps', 'Calories', 'Active Min', 'Heart Rate']
print("\t\t" + "\t".join(metric_names))
for i, name in enumerate(metric_names):
    print(f"{name}\t", end="")
    for j in range(4):
        print(f"{correlation_metrics[i, j]:.3f}\t", end="")
    print()

# Age and activity relationship
age_groups = np.digitize(ages, bins=[18, 30, 45, 60, 70])
print(f"\nAverage Steps by Age Group:")
for age_group in range(1, 5):
    mask = age_groups == age_group
    if np.sum(mask) > 0:
        avg_steps = daily_steps_clean[mask].mean()
        print(f"  Group {age_group}: {avg_steps:.0f} steps")

# Gender comparison
print(f"\nGender-based Activity Comparison:")
female_mask = genders == 0
male_mask = genders == 1
print(f"  Female avg steps: {daily_steps_clean[female_mask].mean():.0f}")
print(f"  Male avg steps: {daily_steps_clean[male_mask].mean():.0f}")





# Health score formula
weights = np.array([0.3, 0.3, 0.3, 0.1])  # Steps, Calories, Minutes, Heart Rate
normalized_metrics = np.zeros_like(user_avg_metrics)
for i in range(4):
    if i == 3:  
        normalized_metrics[:, i] = 1 - (user_avg_metrics[:, i] - user_avg_metrics[:, i].min()) / (user_avg_metrics[:, i].max() - user_avg_metrics[:, i].min())
    else:
        normalized_metrics[:, i] = (user_avg_metrics[:, i] - user_avg_metrics[:, i].min()) / (user_avg_metrics[:, i].max() - user_avg_metrics[:, i].min())

health_scores = (normalized_metrics * weights).sum(axis=1)
top_health_users = np.argsort(health_scores)[-5:][::-1]
print(f"\nTop 5 Health Scores (User IDs): {user_ids[top_health_users]}")





# 4. Goal Achievement
print("\n4. GOAL ACHIEVEMENT")

# Define goals
GOAL_STEPS = 8000
GOAL_CALORIES = 2000
GOAL_MINUTES = 60

# Calculate achievement rates
steps_achieved = daily_steps_clean >= GOAL_STEPS
calories_achieved = calories_clean >= GOAL_CALORIES
minutes_achieved = active_minutes_clean >= GOAL_MINUTES

user_achievement_rates = np.zeros((n_users, 3))
user_achievement_rates[:, 0] = steps_achieved.mean(axis=1) * 100
user_achievement_rates[:, 1] = calories_achieved.mean(axis=1) * 100
user_achievement_rates[:, 2] = minutes_achieved.mean(axis=1) * 100

print(f"\nAverage Goal Achievement Rates:")
print(f"  Steps (8000/day): {user_achievement_rates[:, 0].mean():.1f}%")
print(f"  Calories (2000/day): {user_achievement_rates[:, 1].mean():.1f}%")
print(f"  Active Minutes (60/day): {user_achievement_rates[:, 2].mean():.1f}%")




# Users meeting all goals consistently
all_goals = steps_achieved & calories_achieved & minutes_achieved
consistent_achievers = all_goals.mean(axis=1) > 0.8
print(f"\nUsers meeting all goals >80% of days: {np.sum(consistent_achievers)} users")
if np.sum(consistent_achievers) > 0:
    print(f"Their IDs: {user_ids[consistent_achievers]}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# Part D: Report & Insights
print("\n" + "=" * 70)
print("TASK 3: WRITTEN ANALYSIS REPORT")
print("=" * 70)
print("\nNote: The detailed written report (300+ words) should be added as markdown")
print("cells in your Jupyter notebook, covering:")
print("- Executive Summary with key findings")
print("- Detailed Analysis of patterns observed")
print("- Recommendations for users and company")
print("- Limitations and future directions")
print("\nSee the full report template in the project documentation.")
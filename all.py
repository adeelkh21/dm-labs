Different types of imports
import pandas as pd  

# 1. Importing from Local Disk
from google.colab import files
uploaded = files.upload()  # Upload file manually
df_local = pd.read_csv(next(iter(uploaded)))  # Read the uploaded file
print("Local Disk Data:", df_local.head())

# 2. Importing from Colab’s Disk (Assuming file exists in a path)
df_colab = pd.read_csv('/content/sample_data/mnist_test.csv')  # Change path as needed
print("Colab Disk Data:", df_colab.head())

# 3. Importing from Google Drive
from google.colab import drive
drive.mount('/content/drive')  
df_drive = pd.read_csv('/content/drive/My Drive/dataset.csv')  # Change path accordingly
print("Google Drive Data:", df_drive.head())

# 4. Importing using a URL
url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv"
df_url = pd.read_csv(url)
print("URL Data:", df_url.head())

# 5. Importing from a GitHub Repository
github_url = "https://raw.githubusercontent.com/username/repository/main/dataset.csv"  # Replace with actual URL
df_github = pd.read_csv(github_url)
print("GitHub Data:", df_github.head())

--------------------------------------------------------------------------------------------------------------------------------

Inventory management system 
def add_item(inventory, item_name, quantity):
    """Add items to the inventory."""
    item_name = item_name.lower()
    if quantity <= 0:
        print("Quantity must be a positive integer.")
        return

    inventory[item_name] = inventory.get(item_name, 0) + quantity
    print(f"✅ Added {quantity} of {item_name} to inventory.")

def remove_item(inventory, item_name, quantity):
    """Remove items from the inventory if available."""
    item_name = item_name.lower()
    if item_name in inventory and inventory[item_name] >= quantity:
        inventory[item_name] -= quantity
        if inventory[item_name] == 0:
            del inventory[item_name]
        print(f"❌ Removed {quantity} of {item_name} from inventory.")
    else:
        print(f"⚠️ Cannot remove {quantity} of {item_name}. Not enough in inventory.")

def check_inventory(inventory):
    """Display inventory items."""
    if not inventory:
        print("📦 Inventory is empty.")
    else:
        print("\n📋 Current Inventory:")
        for item_name, quantity in inventory.items():
            print(f"- {item_name.capitalize()}: {quantity}")

def main():
    inventory = {}
    while True:
        print("\n📦 Inventory Management System")
        print("1️⃣ Add Item")
        print("2️⃣ Remove Item")
        print("3️⃣ Check Inventory")
        print("4️⃣ Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            item_name = input("Enter item name: ")
            try:
                quantity = int(input("Enter quantity: "))
                add_item(inventory, item_name, quantity)
            except ValueError:
                print("⚠️ Invalid input. Quantity must be a number.")

        elif choice == '2':
            item_name = input("Enter item name: ")
            try:
                quantity = int(input("Enter quantity: "))
                remove_item(inventory, item_name, quantity)
            except ValueError:
                print("⚠️ Invalid input. Quantity must be a number.")

        elif choice == '3':
            check_inventory(inventory)

        elif choice == '4':
            print("🚪 Exiting the system.")
            break

        else:
            print("⚠️ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()


------------------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1️⃣ Load the dataset (Titanic dataset from seaborn)
import seaborn as sns
df = sns.load_dataset('titanic')

# 2️⃣ Check basic information about the dataset
print("\n🔹 Basic Info:")
print(df.info())

print("\n🔹 First 5 Rows:")
print(df.head())

# 3️⃣ Check for missing data and clean it
print("\n🔹 Missing Values Before Cleaning:")
print(df.isnull().sum())

# Filling missing values
df['age'].fillna(df['age'].median(), inplace=True)  # Fill missing age with median
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)  # Fill missing categorical with mode
df.drop(columns=['deck'], inplace=True)  # Drop 'deck' due to excessive missing values

print("\n🔹 Missing Values After Cleaning:")
print(df.isnull().sum())

# 4️⃣ Find unique values, most common categories, and trends
print("\n🔹 Unique Values per Column:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n🔹 Most Common Categories:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"{col}: {df[col].value_counts().idxmax()} (Most Frequent)")

# 5️⃣ Normalize a numeric column (e.g., 'age' and 'fare') for better comparison
scaler = MinMaxScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

print("\n🔹 Normalized Numeric Columns (First 5 Rows):")
print(df[['age', 'fare']].head())

# Save cleaned dataset
df.to_csv('cleaned_titanic.csv', index=False)
print("\n✅ Data Cleaning & Exploration Complete. Cleaned dataset saved as 'cleaned_titanic.csv'")

-------------------------------------------------------------------------------------------------------------
Handling Missing Values:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = sns.load_dataset('titanic')

# Identify missing values
print("\n🔹 Missing Values Before Imputation:")
print(df.isnull().sum())

# Fill missing values
df['age'].fillna(df['age'].mean(), inplace=True)  # Mean for numerical
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)  # Mode for categorical
df['age'] = df['age'].interpolate()  # Interpolation for sequential data

# Visualizing before and after missing values handling
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Before Handling Missing Values")

plt.subplot(1, 2, 2)
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("After Handling Missing Values")
plt.show()

print("\n🔹 Missing Values After Imputation:")
print(df.isnull().sum())
------------------------------------------------------------------------------------------------------------------

Balancing Data with SMOTE & Undersampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Load dataset (Titanic 'survived' column is imbalanced)
X = df.drop(columns=['survived'])
y = df['survived']

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Check class distribution before balancing
print("\n🔹 Class Distribution Before Balancing:", Counter(y))

# Apply SMOTE (Oversampling)
smote = SMOTE(sampling_strategy=0.8)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("\n🔹 Class Distribution After SMOTE:", Counter(y_resampled))

# Apply Random Undersampling
undersample = RandomUnderSampler(sampling_strategy=0.8)
X_resampled, y_resampled = undersample.fit_resample(X, y)
print("\n🔹 Class Distribution After Undersampling:", Counter(y_resampled))


 Visualizing Data (Before & After Preprocessing)
import numpy as np

# Plot class distributions before & after balancing
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Before balancing
ax[0].bar(["Survived", "Not Survived"], np.bincount(y))
ax[0].set_title("Before Balancing")

# After SMOTE
ax[1].bar(["Survived", "Not Survived"], np.bincount(y_resampled))
ax[1].set_title("After SMOTE")

plt.show()

----------------------------------------------------------------------------------------------------

Correlation Analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Encode categorical features
df_encoded = df.select_dtypes(include=[np.number]).dropna()

# Compute correlation matrix
corr_matrix = df_encoded.corr()
print("\n🔹 Correlation Matrix:\n", corr_matrix)

# Identify highly correlated features (> 0.85)
threshold = 0.85
high_corr_pairs = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns if i != j and abs(corr_matrix.loc[i, j]) > threshold]
print("\n🔹 Highly Correlated Pairs:", high_corr_pairs)

# Drop one feature from each correlated pair
drop_features = set(j for _, j in high_corr_pairs)
df_reduced = df_encoded.drop(columns=drop_features)

# Recompute correlation matrix
corr_matrix_reduced = df_reduced.corr()

# Visualize before and after heatmaps
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Before Feature Dropping")

plt.subplot(1, 2, 2)
sns.heatmap(corr_matrix_reduced, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("After Feature Dropping")

plt.show()

------------------------------------------------------------------------------------------------
Feature Selection for a Classification Problem

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Prepare dataset
df = sns.load_dataset('titanic').dropna()
X = pd.get_dummies(df.drop(columns=['survived']), drop_first=True)
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SelectKBest (f_classif & mutual_info_classif)
k = 5
skb_f_classif = SelectKBest(score_func=f_classif, k=k).fit(X_train, y_train)
skb_mi = SelectKBest(score_func=mutual_info_classif, k=k).fit(X_train, y_train)

# RFE with Logistic Regression
logreg = LogisticRegression(max_iter=500)
rfe = RFE(estimator=logreg, n_features_to_select=k).fit(X_train, y_train)

# Model-based selection with RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
sfm = SelectFromModel(rf, max_features=k).fit(X_train, y_train)

# Get selected feature names
selected_features = {
    "SelectKBest_f_classif": X_train.columns[skb_f_classif.get_support()].tolist(),
    "SelectKBest_mutual_info": X_train.columns[skb_mi.get_support()].tolist(),
    "RFE_LogisticRegression": X_train.columns[rfe.get_support()].tolist(),
    "SelectFromModel_RandomForest": X_train.columns[sfm.get_support()].tolist()
}

print("\n🔹 Selected Features:")
for method, features in selected_features.items():
    print(f"{method}: {features}")

# Train Logistic Regression using selected features & evaluate performance
for method, features in selected_features.items():
    X_train_selected = X_train[features]
    X_test_selected = X_test[features]
    
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    print(f"\n🔹 Accuracy using {method}: {accuracy_score(y_test, y_pred):.4f}")




Recursive Feature Elimination with Cross-Validation (RFECV)
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# RFECV with Logistic Regression
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rfecv = RFECV(estimator=logreg, step=1, cv=cv, scoring='accuracy')
rfecv.fit(X_train, y_train)

# Optimal number of features
print(f"\n🔹 Optimal number of features: {rfecv.n_features_}")

# Plot cross-validation scores
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, marker='o')
plt.xlabel("Number of Features Selected")
plt.ylabel("Cross-Validation Score")
plt.title("RFECV Performance")
plt.show()

# Train Logistic Regression using optimal feature set
X_train_rfecv = X_train.loc[:, rfecv.support_]
X_test_rfecv = X_test.loc[:, rfecv.support_]

logreg.fit(X_train_rfecv, y_train)
y_pred_rfecv = logreg.predict(X_test_rfecv)

# Compare performance with full feature model and SelectKBest
logreg.fit(X_train, y_train)
y_pred_all = logreg.predict(X_test)

X_train_skb = X_train[selected_features["SelectKBest_f_classif"]]
X_test_skb = X_test[selected_features["SelectKBest_f_classif"]]
logreg.fit(X_train_skb, y_train)
y_pred_skb = logreg.predict(X_test_skb)

print("\n🔹 Model Performance Comparison:")
print(f"Full Feature Model Accuracy: {accuracy_score(y_test, y_pred_all):.4f}")
print(f"RFECV Selected Features Accuracy: {accuracy_score(y_test, y_pred_rfecv):.4f}")
print(f"SelectKBest Accuracy: {accuracy_score(y_test, y_pred_skb):.4f}")

--------------------------------------------------------------------------------------------

Experimenting with Different Threshold Values

import numpy as np
import matplotlib.pyplot as plt

# Sample data: Precision-Recall pairs for different thresholds
thresholds = np.linspace(0, 1, 10)
precision = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30]
recall = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision, marker='o', label="Precision")
plt.plot(thresholds, recall, marker='s', label="Recall")
plt.xlabel("Threshold Value")
plt.ylabel("Score")
plt.title("Impact of Threshold on Precision & Recall")
plt.legend()
plt.grid(True)
plt.show()

print("\n🔹 Best threshold depends on trade-off: High precision → low recall, and vice versa.")
---------------------------------------------------------------------------------------------------------------

Analyzing Fast Food Orders for Meal Combinations
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Manually creating a dataset
data = [
    ['Burger', 'Fries', 'Soda'],
    ['Pizza', 'Soda'],
    ['Burger', 'Pizza', 'Fries'],
    ['Burger', 'Fries'],
    ['Fries', 'Soda'],
    ['Pizza', 'Soda', 'Burger'],
    ['Fries', 'Soda'],
    ['Burger', 'Fries', 'Pizza'],
    ['Pizza', 'Soda'],
    ['Burger', 'Soda']
]

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv("FastFood_Orders.csv", index=False)

# Convert transactions to one-hot encoding
df_encoded = df.apply(lambda x: x.str.get_dummies(sep=',')).sum(level=0)

# Apply Apriori Algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True)

# Generate Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("\n🔹 Association Rules:\n", rules)

# Visualize Results
plt.figure(figsize=(8, 5))
rules['support'].plot(kind='bar', color='skyblue')
plt.title("Support of Frequent Itemsets")
plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.xticks(rotation=45)
plt.show()
------------------------------------------------------------------------------------------------------
Identifying Online Shopping Trends for a Clothing Store

import pandas as pd

# Simulated dataset: Online Clothing Store Transactions
transactions = [
    ['T-shirt', 'Jeans', 'Sneakers'],
    ['Dress', 'Heels'],
    ['Jeans', 'T-shirt', 'Sneakers'],
    ['T-shirt', 'Jeans'],
    ['Dress', 'Handbag'],
    ['Sneakers', 'Jeans'],
    ['T-shirt', 'Sneakers'],
    ['Dress', 'Heels', 'Handbag'],
    ['Jeans', 'Sneakers'],
    ['T-shirt', 'Jeans']
]

# Convert to DataFrame
df_clothing = pd.DataFrame(transactions)

# One-hot encoding
df_encoded = df_clothing.apply(lambda x: x.str.get_dummies(sep=',')).sum(level=0)

# Apply Apriori Algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

# Identify most frequent categories
top_items = frequent_itemsets.nlargest(5, 'support')

# Visualize top product associations
plt.figure(figsize=(8, 5))
plt.bar(top_items['itemsets'].astype(str), top_items['support'], color='purple')
plt.title("Top Clothing Product Associations")
plt.xlabel("Product Pairings")
plt.ylabel("Support")
plt.xticks(rotation=45)
plt.show()

print("\n🔹 Suggested Product Recommendations: Pair popular items like 'T-shirt & Sneakers' or 'Dress & Heels'.")

---------------------------------------------------------------------------------------------------

Experimenting with Different Threshold Values for FP-Growth Algorithm

import pandas as pd
import time
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

# Sample transaction dataset
data = [
    ['Burger', 'Fries', 'Soda'],
    ['Pizza', 'Soda'],
    ['Burger', 'Pizza', 'Fries'],
    ['Burger', 'Fries'],
    ['Fries', 'Soda'],
    ['Pizza', 'Soda', 'Burger'],
    ['Fries', 'Soda'],
    ['Burger', 'Fries', 'Pizza'],
    ['Pizza', 'Soda'],
    ['Burger', 'Soda']
]

# Convert transactions to DataFrame
df = pd.DataFrame(data)
df.to_csv("FastFood_Orders.csv", index=False)

# One-hot encoding
df_encoded = df.apply(lambda x: x.str.get_dummies(sep=',')).sum(level=0)

# Test different support thresholds for FP-Growth
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
for min_sup in thresholds:
    start_time = time.time()
    frequent_itemsets_fp = fpgrowth(df_encoded, min_support=min_sup, use_colnames=True)
    execution_time = time.time() - start_time
    print(f"\n🔹 FP-Growth Results for min_support={min_sup}:")
    print(frequent_itemsets_fp)
    print(f"⏱ Execution Time: {execution_time:.4f} seconds")

# Compare execution time with Apriori
start_time = time.time()
frequent_itemsets_apriori = apriori(df_encoded, min_support=0.2, use_colnames=True)
execution_time_apriori = time.time() - start_time
print(f"\n⏱ Apriori Execution Time: {execution_time_apriori:.4f} seconds")


Analyzing Fast Food Orders for Meal Combinations Using FP-Growth
# Apply FP-Growth Algorithm
frequent_itemsets_fp = fpgrowth(df_encoded, min_support=0.2, use_colnames=True)

# Generate Association Rules
rules_fp = association_rules(frequent_itemsets_fp, metric="lift", min_threshold=1.0)
print("\n🔹 FP-Growth Association Rules:\n", rules_fp)

# Visualizing results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
rules_fp['support'].plot(kind='bar', color='orange')
plt.title("Support of Frequent Itemsets (FP-Growth)")
plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.xticks(rotation=45)
plt.show()

-----------------------------------------------------------------------------------------------------
Identifying Online Shopping Trends for a Clothing Store Using FP-Growth

# Simulated dataset: Online Clothing Store Transactions
transactions = [
    ['T-shirt', 'Jeans', 'Sneakers'],
    ['Dress', 'Heels'],
    ['Jeans', 'T-shirt', 'Sneakers'],
    ['T-shirt', 'Jeans'],
    ['Dress', 'Handbag'],
    ['Sneakers', 'Jeans'],
    ['T-shirt', 'Sneakers'],
    ['Dress', 'Heels', 'Handbag'],
    ['Jeans', 'Sneakers'],
    ['T-shirt', 'Jeans']
]

# Convert to DataFrame
df_clothing = pd.DataFrame(transactions)

# One-hot encoding
df_encoded_clothing = df_clothing.apply(lambda x: x.str.get_dummies(sep=',')).sum(level=0)

# Apply FP-Growth Algorithm
frequent_itemsets_clothing = fpgrowth(df_encoded_clothing, min_support=0.2, use_colnames=True)

# Identify most frequent categories
top_items = frequent_itemsets_clothing.nlargest(5, 'support')

# Visualize top product associations
plt.figure(figsize=(8, 5))
plt.bar(top_items['itemsets'].astype(str), top_items['support'], color='purple')
plt.title("Top Clothing Product Associations (FP-Growth)")
plt.xlabel("Product Pairings")
plt.ylabel("Support")
plt.xticks(rotation=45)
plt.show()

print("\n🔹 Suggested Product Recommendations: Pair popular items like 'T-shirt & Sneakers' or 'Dress & Heels'.")

----------------------------------------------------------------------------------------------------------------

Classification using Decision Tree & Naive Bayes

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load dataset
df = sns.load_dataset('titanic').drop(columns=['deck'])  # Drop column with too many missing values

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
df['sex'] = encoder.fit_transform(df['sex'])
df['embark_town'] = encoder.fit_transform(df['embark_town'])
df['alone'] = encoder.fit_transform(df['alone'])

# Select features and target
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embark_town']]
y = df['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Evaluate Models
def evaluate_model(name, y_test, y_pred):
    print(f"\n🔹 {name} Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
    print(classification_report(y_test, y_pred))

# Compare Decision Tree and Naive Bayes
evaluate_model("Decision Tree", y_test, y_pred_dt)
evaluate_model("Naive Bayes", y_test, y_pred_nb)

# Plot Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap="Blues", ax=ax[0])
ax[0].set_title("Decision Tree Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap="Oranges", ax=ax[1])
ax[1].set_title("Naive Bayes Confusion Matrix")

plt.show()

-------------------------------------------------------------------------------------------------------
Classification using KNN & SVM

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load dataset
df = sns.load_dataset('titanic').drop(columns=['deck'])  # Drop column with too many missing values

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
df['sex'] = encoder.fit_transform(df['sex'])
df['embark_town'] = encoder.fit_transform(df['embark_town'])
df['alone'] = encoder.fit_transform(df['alone'])

# Select features and target
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embark_town']]
y = df['survived']

# Scale features (important for KNN & SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Train Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate Models
def evaluate_model(name, y_test, y_pred):
    print(f"\n🔹 {name} Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
    print(classification_report(y_test, y_pred))

# Compare KNN and SVM
evaluate_model("K-Nearest Neighbors", y_test, y_pred_knn)
evaluate_model("Support Vector Machine", y_test, y_pred_svm)

# Plot Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap="Blues", ax=ax[0])
ax[0].set_title("KNN Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap="Oranges", ax=ax[1])
ax[1].set_title("SVM Confusion Matrix")

plt.show()

-------------------------------------------------------------------------------------------------------------
Predicting Personal Loan Approvals

# Justification: SVM is generally better for high-dimensional data and works well for binary classification problems.
# However, KNN can be useful if the data is non-linearly separable. We'll implement both.

# Simulated Dataset (Example: Customers applying for a loan)
np.random.seed(42)
loan_data = pd.DataFrame({
    'income': np.random.randint(30000, 100000, 100),
    'credit_score': np.random.randint(300, 850, 100),
    'age': np.random.randint(20, 60, 100),
    'loan_approved': np.random.choice([0, 1], size=100)  # 0: Not Approved, 1: Approved
})

# Split dataset
X_loan = loan_data[['income', 'credit_score', 'age']]
y_loan = loan_data['loan_approved']

scaler = StandardScaler()
X_loan_scaled = scaler.fit_transform(X_loan)

X_train_loan, X_test_loan, y_train_loan, y_test_loan = train_test_split(X_loan_scaled, y_loan, test_size=0.2, random_state=42)

# Train SVM
svm_model_loan = SVC(kernel='linear')
svm_model_loan.fit(X_train_loan, y_train_loan)
y_pred_svm_loan = svm_model_loan.predict(X_test_loan)

# Evaluate SVM
evaluate_model("SVM for Loan Prediction", y_test_loan, y_pred_svm_loan)


------------------------------------------------------------------------------------------
 K-Means, K-Median, and Hierarchical Clustering

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: pip install scikit-learn-extra for KMedians
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None
    print("Warning: sklearn-extra not installed. K-Medians will be skipped.")


DATA_PATH = "banknote.csv"  # replace with your dataset path


df = pd.read_csv(DATA_PATH)
print("Initial shape:", df.shape)
print(df.info())
print(df.describe())


df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
print("Cleaned shape:", df.shape)


# Select only numerical columns for clustering
numeric_df = df.select_dtypes(include=[np.number])
print("Selected numeric features:", numeric_df.columns.tolist())


scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)


wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


k_optimal = 3  # Replace based on Elbow plot
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_data)
df['KMeans_Cluster'] = kmeans_labels


if KMedoids:
    kmedians = KMedoids(n_clusters=k_optimal, method='pam', random_state=42)
    kmedians_labels = kmedians.fit_predict(scaled_data)
    df['KMedian_Cluster'] = kmedians_labels
else:
    print("K-Median skipped due to missing sklearn-extra.")


linked = linkage(scaled_data, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Euclidean distances")
plt.show()


agg_cluster = AgglomerativeClustering(n_clusters=k_optimal)
agg_labels = agg_cluster.fit_predict(scaled_data)
df['Agglomerative_Cluster'] = agg_labels


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=kmeans_labels, palette='Set2')
plt.title("K-Means Clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()


print("Cluster interpretation:")
print(df.groupby('KMeans_Cluster').mean())

------------------------------------------------------------------------------------------------
# Outlier Detection including Z-Score, IQR, DBSCAN, Isolation 
Forest, Local Outlier Factor, and One-Class SVM—

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# ---- CONFIGURATION ----
DATA_PATH = "banknote.csv"  # replace with your dataset file path
COLUMN_FOR_STATS = None  # Optional: specify a column for Z-score and IQR. If None, the first numeric column is used.

# ---- LOAD & CLEAN DATA ----
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# ---- SELECT NUMERIC COLUMNS & SCALE ----
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.empty:
    raise ValueError("No numeric columns found in dataset.")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# ---- SELECT COLUMN FOR Z-SCORE AND IQR ----
col = COLUMN_FOR_STATS or numeric_df.columns[0]
print(f"\nUsing column '{col}' for Z-Score and IQR outlier detection.\n")

# ---- Z-SCORE METHOD ----
z_scores = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std())
z_outliers = z_scores > 3
print("Z-Score Outliers Detected:", np.sum(z_outliers))

# ---- IQR METHOD ----
Q1 = numeric_df[col].quantile(0.25)
Q3 = numeric_df[col].quantile(0.75)
IQR = Q3 - Q1
iqr_outliers = (numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))
print("IQR Outliers Detected:", np.sum(iqr_outliers))

# ---- PCA FOR 2D REDUCTION ----
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# ---- DBSCAN ----
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(pca_data)
db_outliers = np.sum(db_labels == -1)
print("DBSCAN Outliers Detected:", db_outliers)

# ---- DBSCAN PLOT ----
plt.figure(figsize=(8, 5))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=db_labels, palette="tab10", legend=None)
plt.title("DBSCAN Outlier Detection")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

# ---- ISOLATION FOREST ----
iso = IsolationForest(contamination=0.05, random_state=42)
iso_labels = iso.fit_predict(scaled_data)
iso_outliers = np.sum(iso_labels == -1)
print("Isolation Forest Outliers Detected:", iso_outliers)

# ---- LOCAL OUTLIER FACTOR ----
sample_size = min(10000, len(scaled_data))
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_labels = lof.fit_predict(scaled_data[:sample_size])
lof_outliers = np.sum(lof_labels == -1)
print("LOF Outliers Detected (sample of 10k or less):", lof_outliers)

# ---- ONE-CLASS SVM ----
svm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
svm_labels = svm.fit_predict(scaled_data)
svm_outliers = np.sum(svm_labels == -1)
print("One-Class SVM Outliers Detected:", svm_outliers)

# ---- COMPARISON TABLE ----
results = {
    "Method": ["Z-Score", "IQR", "DBSCAN", "Isolation Forest", "LOF (sample)", "One-Class SVM"],
    "Outliers Detected": [np.sum(z_outliers), np.sum(iqr_outliers), db_outliers, iso_outliers, lof_outliers, svm_outliers]
}
comparison_df = pd.DataFrame(results)
print("\nOutlier Detection Comparison:\n", comparison_df)

# ---- OPTIONAL: VISUALIZATION SUMMARY ----
comparison_df.set_index("Method").plot(kind="bar", figsize=(10, 5), legend=False)
plt.title("Outlier Detection - Method Comparison")
plt.ylabel("Outliers Detected")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

---------------------------------------------------------------------------------------------------
clustering k means k median

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None
    print("K-Median requires sklearn-extra. Falling back to Agglomerative Clustering.")

# --- CONFIG ---
DATA_PATH = "banknote.csv"  # Replace with path to your dataset

# --- LOAD + CLEAN ---
df = pd.read_csv(DATA_PATH)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Drop non-numeric or ID-like columns
df = df.select_dtypes(include=[np.number])
if df.empty:
    raise ValueError("No numeric columns found.")

# --- SCALE ---
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# --- ELBOW METHOD ---
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Clusters")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# --- KMEANS ---
k = 3  # Choose based on elbow plot
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["KMeans_Label"] = kmeans.fit_predict(scaled)

# --- SECOND METHOD ---
if KMedoids:
    model = KMedoids(n_clusters=k, method="pam", random_state=42)
    df["Second_Cluster"] = model.fit_predict(scaled)
    method_name = "K-Median"
else:
    model = AgglomerativeClustering(n_clusters=k)
    df["Second_Cluster"] = model.fit_predict(scaled)
    method_name = "Agglomerative"

# --- PCA & VISUALIZATION ---
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df["KMeans_Label"], palette="Set2")
plt.title("K-Means Clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df["Second_Cluster"], palette="Set1")
plt.title(f"{method_name} Clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()
plt.show()

# --- INTERPRETATION ---
print("\nKMeans Cluster Averages:\n", df.groupby("KMeans_Label").mean())
print(f"\n{method_name} Cluster Averages:\n", df.groupby("Second_Cluster").mean())

-------------------------------------------------------------------------------------------

classification of any two models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_PATH = "your_dataset.csv"
TARGET = "target_column"  # Replace with your target column

# --- LOAD + CLEAN ---
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# --- ENCODING ---
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# --- FEATURES / TARGET SPLIT ---
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# --- SCALE NUMERIC FEATURES ---
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# --- SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# --- MODEL TRAINING ---
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Evaluation:\n")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    results[name] = model.score(X_test, y_test)

# --- COMPARISON ---
best = max(results, key=results.get)
print(f"\nRecommended Model: {best} with accuracy {results[best]:.4f}")

---------------------------------------------------------------------------------------------------------

Association rule mining plus svm universal code

import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
DATA_PATH = "your_dataset.csv"  # Replace with your dataset
TARGET = "target_column"        # Replace for SVM section only
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.6

# --- LOAD DATA ---
df = pd.read_csv(DATA_PATH)
print("Initial Data:\n", df.head())

# --- UNIVERSAL PREPROCESSING ---
df = df.drop_duplicates().dropna()

# --- ASSOCIATION RULE MINING SECTION ---

def run_association_analysis(data):
    # Check if data is suitable for transactional encoding
    if not all(data.dtypes == bool) and not all(data.dtypes == int):
        print("\nTransforming for transaction format...")
        # One-hot encode categorical columns (use cautiously)
        trans_data = data.astype(str)
        trans_data = trans_data.apply(lambda x: x.name + "_" + x)
        df_encoded = pd.get_dummies(trans_data)
        df_encoded = df_encoded.groupby(level=0, axis=1).max()
    else:
        df_encoded = data.copy()

    print("\nRunning Apriori...")
    frequent_items_ap = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    rules_ap = association_rules(frequent_items_ap, metric="confidence", min_threshold=MIN_CONFIDENCE)
    print("\nTop Apriori Rules:\n", rules_ap[['antecedents', 'consequents', 'support', 'confidence']].head())

    print("\nRunning FP-Growth...")
    frequent_items_fp = fpgrowth(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    rules_fp = association_rules(frequent_items_fp, metric="confidence", min_threshold=MIN_CONFIDENCE)
    print("\nTop FP-Growth Rules:\n", rules_fp[['antecedents', 'consequents', 'support', 'confidence']].head())

# Only run ARM if dataset is suitable (lots of object/categorical values or transaction-style data)
if df.select_dtypes(include=['object', 'bool']).shape[1] > 0 or df.shape[1] <= 50:
    run_association_analysis(df.copy())

# --- SVM CLASSIFICATION SECTION ---

def run_svm_classifier(data, target_col):
    data = data.copy()
    print(f"\nPreparing for SVM classification on target: {target_col}")

    # Encode categorical features
    for col in data.select_dtypes(include='object').columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    # Split features/target
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Scale numeric features
    X_scaled = StandardScaler().fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

    # SVM Model
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    print("\nSVM Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Run SVM only if target exists
if TARGET in df.columns:
    run_svm_classifier(df, TARGET)
else:
    print(f"\nTarget column '{TARGET}' not found. Skipping SVM section.")


---------------------------------------------------------------------------------------------------------
Time series airline prediction Decomposition (Additive & Multiplicative) Stationarity check (ADF Test) Differencing ARIMA modeling Forecasting

import pandas as pd

# Example dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)

# Convert to datetime
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Handle missing values (if any)
df = df.interpolate()

import matplotlib.pyplot as plt

df.plot(title='Monthly Airline Passengers')
plt.ylabel('Passengers')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

result_add = seasonal_decompose(df, model='additive')
result_mul = seasonal_decompose(df, model='multiplicative')

result_add.plot().suptitle('Additive Decomposition')
plt.show()

result_mul.plot().suptitle('Multiplicative Decomposition')
plt.show()

from statsmodels.tsa.stattools import adfuller
import numpy as np

# Rolling stats
rolling_mean = df.rolling(window=12).mean()
rolling_std = df.rolling(window=12).std()

plt.plot(df, label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')
plt.legend()
plt.title('Rolling Mean & Std')
plt.show()

# ADF Test
result = adfuller(df['Passengers'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')


df_diff = df.diff().dropna()

df_diff.plot(title='First Order Differencing')
plt.show()

# Recheck stationarity
result_diff = adfuller(df_diff['Passengers'])
print(f'ADF Statistic after differencing: {result_diff[0]}')
print(f'p-value after differencing: {result_diff[1]}')


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# Forecast next 12 months
forecast = model_fit.forecast(steps=12)
forecast_index = pd.date_range(df.index[-1], periods=13, freq='MS')[1:]
forecast = pd.Series(forecast, index=forecast_index)

# Plot
plt.plot(df, label='Original')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.title('12-step Forecast')
plt.show()



-------------------------------------------------------------------------------------
#Universal time series analysis code
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# ---- CONFIG ----
DATA_PATH = "your_timeseries.csv"     # or a URL
DATE_COL = "Date"                     # column to parse as datetime
TARGET_COL = "Value"                 # numeric time series column

# ---- LOAD DATA ----
df = pd.read_csv(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df.set_index(DATE_COL, inplace=True)
df.sort_index(inplace=True)
df = df[[TARGET_COL]]

# ---- HANDLE MISSING ----
df = df.interpolate()

# ---- PLOT ORIGINAL SERIES ----
df.plot(title='Original Time Series')
plt.ylabel(TARGET_COL)
plt.grid(True)
plt.show()

# ---- DECOMPOSITION ----
try:
    result_add = seasonal_decompose(df, model='additive', period=12)
    result_mul = seasonal_decompose(df, model='multiplicative', period=12)

    result_add.plot().suptitle('Additive Decomposition', fontsize=16)
    plt.show()

    result_mul.plot().suptitle('Multiplicative Decomposition', fontsize=16)
    plt.show()
except Exception as e:
    print("Decomposition error:", e)

# ---- ROLLING STATISTICS ----
rolling_mean = df.rolling(window=12).mean()
rolling_std = df.rolling(window=12).std()

plt.plot(df, label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')
plt.legend()
plt.title('Rolling Mean & Standard Deviation')
plt.grid(True)
plt.show()

# ---- ADF TEST ----
result = adfuller(df[TARGET_COL])
print("ADF Test (Original Data):")
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
print()

# ---- DIFFERENCING ----
df_diff = df.diff().dropna()
df_diff.plot(title='First Order Differencing')
plt.grid(True)
plt.show()

# ---- ADF TEST AFTER DIFFERENCING ----
result_diff = adfuller(df_diff[TARGET_COL])
print("ADF Test (After Differencing):")
print(f"ADF Statistic: {result_diff[0]:.4f}")
print(f"p-value: {result_diff[1]:.4f}")
print()

# ---- ARIMA MODEL ----
try:
    model = ARIMA(df, order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())
except Exception as e:
    print("ARIMA fitting error:", e)

# ---- FORECASTING ----
forecast_steps = 12
try:
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(),
                                   periods=forecast_steps, freq='MS')
    forecast_series = pd.Series(forecast, index=forecast_index)

    # ---- PLOT FORECAST ----
    plt.plot(df, label='Original')
    plt.plot(forecast_series, label='Forecast', color='red')
    plt.legend()
    plt.title(f'{forecast_steps}-Step Forecast')
    plt.grid(True)
    plt.show()
except Exception as e:
    print("Forecasting error:", e)
-----------------------------------------------------------------------------------------------


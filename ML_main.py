import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train.astype(float), X_test.astype(float), y_train.astype(float), y_test.astype(float)


def get_model(model_name):
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0),
        #'KNN': KNeighborsClassifier(n_neighbors=5),
        #'SVC': SVC()
    }
    return models[model_name]


train_user = pd.read_csv("X_train.csv" , index_col='id')


# Drop the target column from train_user
X = train_user.drop('country_destination', axis=1)
y = train_user['country_destination']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Train and evaluate the models
models = ['Decision Tree', 'Random Forest']
results = {}
for model_name in models:
    # Get model
    model = get_model(model_name)
    print("{model} : Strat....")
    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    results[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    
# Store the results in a pandas DataFrame
results_df = pd.DataFrame(results)
print(results_df)
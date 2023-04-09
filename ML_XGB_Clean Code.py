import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score


def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, label_encoder

def train_model(X, y, params):
    xgb_model = XGBClassifier(tree_method='gpu_hist')
    grid_search = GridSearchCV(xgb_model, param_grid=params, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_params_

def predict(model, X, label_encoder):
    y_pred = model.predict(X)
    return label_encoder.inverse_transform(y_pred)

def predict_top_k_countries(model, X, test_ids, label_encoder, k=5):
    pred_prob = model.predict_proba(X)
    ids = []
    cts = []
    for i, test_id in enumerate(test_ids):
        idx = test_id
        ids += [idx] * k
        cts += label_encoder.inverse_transform(pred_prob[i].argsort()[::-1][:k]).tolist()

        # 計算自訂模型評估指標


    return ids, cts

def generate_submission(ids, cts, file_path):
    sub = pd.DataFrame({'id': ids, 'country': cts})
    sub.to_csv(file_path, index=False)

# 載入資料
X_train = load_data('X_train.csv')
X_test = load_data('X_test.csv')
y_train = load_data('y_train.csv')['country_destination']
test_df = load_data('./airbnb-recruiting-new-user-bookings/test_users.csv.zip')
test_ids = test_df['id'].values

# 資料前處理
X_train, y_train_encoded, label_encoder = preprocess_data(X_train, y_train)
X_test, _, _ = preprocess_data(X_test, y_train)

# 調整參數
params = {'learning_rate': [0.1],
          'max_depth': [6],
          'subsample': [0.5],
          'colsample_bytree': [0.5],
          'n_estimators': [25]}
best_params = train_model(X_train, y_train_encoded, params)

# 建立最佳模型並預測
model = XGBClassifier(**best_params)
model.fit(X_train, y_train_encoded)
y_pred = predict(model, X_test, label_encoder)
ids, cts = predict_top_k_countries(model, X_test, test_ids, label_encoder, k=5)

# 輸出最佳參數和預測結果
print(f'Best parameters found: {best_params}')
print(f'XGBoost predictions: {y_pred.shape}')

# 產生預測結果並提交
#generate_submission(ids, cts, 'sub.csv')

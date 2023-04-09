# # 計算五個分類模型的評估指標
# models = [lr, dt, rf]
# model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
# accuracy_scores = []
# precision_scores = []
# recall_scores = []
# f1_scores = []
# confusion_matrices = []

# for model, model_name in zip(models, model_names):
#     try:
#         y_pred = model.predict(X_train)
#         accuracy_scores.append(accuracy_score(y_train, y_pred))
#         precision_scores.append(precision_score(y_train, y_pred, average='weighted'))
#         recall_scores.append(recall_score(y_train, y_pred, average='weighted'))
#         f1_scores.append(f1_score(y_train, y_pred, average='weighted'))
#         confusion_matrices.append(confusion_matrix(y_train, y_pred))
#     except Exception as e:
#         print(f"Error in {model_name}: {e}")


# try:
#     # 預測測試集
#     y_pred = xgb.predict(X_train)

#     # 計算評估指標
#     accuracy = accuracy_score(y_train, y_pred)
#     precision = precision_score(y_train, y_pred, average='weighted')
#     recall = recall_score(y_train, y_pred, average='weighted')
#     f1 = f1_score(y_train, y_pred, average='weighted')
#     confusion_matrix = confusion_matrix(y_train, y_pred)

#     # 顯示評估結果和混淆矩陣
#     print(f"Accuracy: {accuracy}")
#     print(f"Precision: {precision}")
#     print(f"Recall: {recall}")
#     print(f"F1 score: {f1}")
#     print("Confusion Matrix:\n", confusion_matrix)

# except Exception as e:
#         print(f"Error in {model_name}: {e}")


# try:
#     # 預測測試集
#     y_pred = lgbm.predict(X_test)

#     # 計算評估指標
#     accuracy = accuracy_score(y_train, y_pred)
#     precision = precision_score(y_train, y_pred, average='weighted')
#     recall = recall_score(y_train, y_pred, average='weighted')
#     f1 = f1_score(y_train, y_pred, average='weighted')
#     confusion_matrix = confusion_matrix(y_train, y_pred)

#     # 顯示評估結果和混淆矩陣
#     print(f"Accuracy: {accuracy}")
#     print(f"Precision: {precision}")
#     print(f"Recall: {recall}")
#     print(f"F1 score: {f1}")
#     print("Confusion Matrix:\n", confusion_matrix)

# except Exception as e:
#         print(f"Error in {model_name}: {e}")
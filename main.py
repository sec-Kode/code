# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# 读取数据
train_df = pd.read_csv('./zhengqi_train.txt', sep='\t', encoding='utf-8')
test_df = pd.read_csv('./zhengqi_test.txt', sep='\t', encoding='utf-8')

# 去除异常值
def remove_train_outliers(dataset):
    for feature_col in dataset.columns[:-1]:
        Q1 = dataset.loc[:, feature_col].quantile(0.25)
        Q3 = dataset.loc[:, feature_col].quantile(0.75)
        IQR = Q3 - Q1

        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR

        outliers = (dataset[feature_col] > upper_bound) | (dataset[feature_col] < lower_bound)
        dataset = dataset[~outliers]

    return dataset

train_df = remove_train_outliers(train_df)

# 特征选择
new_train_df = train_df.drop(['V5', 'V9', 'V11', 'V14', 'V17', 'V21', 'V22', 'V27', 'V35'], axis=1)
new_test_df = test_df.drop(['V5', 'V9', 'V11', 'V14', 'V17', 'V21', 'V22', 'V27', 'V35'], axis=1)

# 相关性分析
threshold = 0.45
corr_matrix = new_train_df.corr().abs()
drop_cols = corr_matrix[corr_matrix['target'] < threshold].index
new_train_df.drop(drop_cols, axis=1, inplace=True)
new_test_df.drop(drop_cols, axis=1, inplace=True)

# 数据分割
train_label, val_label, train_target, val_target = train_test_split(
    new_train_df[new_train_df.columns[:-1]],
    new_train_df['target'],
    test_size=0.2,
    shuffle=True,
    random_state=2023
)

# 模型参数网格
param_grid_rf = {
    'n_estimators': [80, 100, 160, 240, 300],
    'max_depth': [3, 5, 8, 10],
    'max_features': [2, 3, 4, 5]
}
param_grid_gb = {
    'n_estimators': [80, 100, 160, 240, 300],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'max_depth': [3, 5, 8, 10],
    'max_features': [2, 3, 4, 5]
}
param_grid_ab = {
    'n_estimators': [80, 100, 160, 240, 300],
    'learning_rate': [0.005, 0.01, 0.05, 0.1]
}
param_grid_bag = {
    "estimator": [DecisionTreeRegressor(), SVR(), RandomForestRegressor(), GradientBoostingRegressor()],
    'n_estimators': [80, 100, 160, 240, 300],
    'max_features': [2, 3, 4, 5]
}
param_grid_lgb = {
    'n_estimators': [80, 100, 160, 240, 300],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'reg_lambda': [0.8, 1, 1.5, 2]
}
param_grid_xgb = {
    'n_estimators': [80, 100, 160, 240, 300],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'reg_lambda': [0.8, 1, 1.5, 2]
}

# 定义模型列表
models = [
    ('RandomForestRegressor', RandomForestRegressor(), param_grid_rf),
    ('GradientBoostingRegressor', GradientBoostingRegressor(), param_grid_gb),
    ('AdaBoostRegressor', AdaBoostRegressor(), param_grid_ab),
    ('LGBMRegressor', lgb.LGBMRegressor(device='cpu'), param_grid_lgb),
    ('BaggingRegressor', BaggingRegressor(n_jobs=-1), param_grid_bag),  # 使用多核 CPU
    ('XGBRegressor', xgb.XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', device='cuda:0'), param_grid_xgb)
]

# 模型训练与预测
y_preds = []
for model_name, model, param_grid in tqdm(models):
    print(f"Training and predicting with {model_name}...")

    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(train_label, train_target)

    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

    best_model = model.set_params(**best_params)
    best_model.fit(train_label, train_target)

    y_train_pred = best_model.predict(train_label)
    train_mse = mean_squared_error(train_target, y_train_pred)
    print(f"train loss: {train_mse}")

    y_val = best_model.predict(val_label)
    val_mse = mean_squared_error(val_target, y_val)
    print(f"val loss: {val_mse}")

    # 使用 predict 替代 inplace_predict
    y_pred = best_model.predict(new_test_df)
    y_preds.append(y_pred)

    print("-------------------------------------")

# 保存预测结果
df_result = pd.DataFrame(y_preds[0])
df_result.to_csv("predictions.txt", header=None, index=False, sep="\t")

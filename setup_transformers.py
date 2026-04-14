import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, SequentialFeatureSelector, SelectFromModel
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('cleaned_owid_energy_data1.csv')
target = 'renewables_share_energy'
X = df.drop(columns=[target, 'country', 'iso_code'])
y = df[target]

print("Fitting scaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_df = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

print("Fitting SelectKBest...")
skb = SelectKBest(f_regression, k=10)
skb.fit(X_train, y_train)

print("Fitting SequentialFeatureSelector...")
sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=10, direction='forward')
sfs.fit(X_train, y_train)

print("Fitting LassoCV...")
lasso_sel = SelectFromModel(LassoCV(), threshold='median')
lasso_sel.fit(X_train, y_train)

print("Fitting PCA...")
pca = PCA(n_components=10)
pca.fit(X_train)

transformers = {
    'scaler': scaler,
    'skb': skb,
    'sfs': sfs,
    'lasso': lasso_sel,
    'pca': pca,
    'columns': X.columns.tolist()
}

joblib.dump(transformers, 'transformers.joblib')
print("Transformers saved successfully to transformers.joblib.")

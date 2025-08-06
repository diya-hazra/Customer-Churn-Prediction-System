import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from preprocess import load_and_preprocess

df = load_and_preprocess('data/Telco-Customer-Churn.csv')

X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'xgb_model.pkl')

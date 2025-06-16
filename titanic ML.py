import seaborn as sns
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


df = sns.load_dataset("titanic")

df = df.dropna(subset=['age', 'embarked', 'fare', 'sex'])
df = df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'])

df['family_size'] = df['sibsp'] + df['parch']
df = df.drop(columns=['sibsp', 'parch'])
df['sex'] = df['sex'].map({'male':0, 'female':1})
df['embarked'] = df['embarked'].map({'S':0, 'C':1, 'Q':2})

X = df[['pclass', 'sex', 'age', 'fare', 'embarked', 'family_size']]
Y = df['survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=65)

#logistics regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, Y_train)
log_preds = log_model.predict(X_test)
log_acc = accuracy_score(Y_test, log_preds)
print("logistic regression accuracy:", round(log_acc * 100, 2), "%")

# random forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, Y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(Y_test, rf_preds)
print("Random forest accuracy:", round(rf_acc * 100, 2), "%")
importance = rf_model.feature_importances_
feature_names = X.columns
joblib.dump(rf_model, 'titanic_rf_model.pkl')
print("random forest model saved as 'titanic_rf_model.pkl")
model_cv = joblib.load('titanic_rf_model.pkl')
new_passenger = pd.DataFrame({'pclass': [2],'sex': [1], 'age': [29], 'fare': [30.0], 'embarked': [0], 'family_size': [1]})
prediction = model_cv.predict(new_passenger)
print("prediction (1 = survived, 0 = died:", prediction[0])

# decision tree
model_cv = DecisionTreeClassifier(max_depth=3, random_state=45)
scores = cross_val_score(model_cv, X, Y, cv=5)
model_cv.fit(X_train, Y_train)

Y_pred = model_cv.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)

print("Accuracy of decision tree:", round(acc *100, 2), "%")
print(f"Accuracy: {acc * 100:.2f}%")

importance_df = pd.DataFrame({'feature' : feature_names, 'importance' : importance})
importance_df = importance_df.sort_values(by='importance', ascending=False)
# sns.countplot(x = 'sex', hue = 'survived', color='purple', data=df)
plt.figure(figsize=(8, 5))
plt.bar(importance_df['feature'], importance_df['importance'], color='pink')
plot_tree(model_cv, feature_names=X.columns, class_names=["died", "survived"], filled=True)
# plt.title('Decision tree visualization')
plt.title('feature importance from random forest')
plt.xlabel('feature')
plt.ylabel('importance')
plt.xticks([0,1], ['male', 'female'])
plt.legend(title = 'survived', labels = ['no', 'yes'])
plt.show()

print("cross-validation scores:", scores)
print("average CV Accuracy:", round(scores.mean() *100, 2), "%")

print("training data shape:", X_train.shape)
print("testing data shape:", X_test.shape)

sns.set(style="whitegrid")
# bar plot
sns.barplot(x="sex", y="survived", data=df, palette="pastel")
plt.title("survival rate by gender")
plt.ylabel("survival rate")
plt.xticks([0,1], ['male', 'female'])
plt.show()

# bar plot 
sns.barplot(x="pclass", y="survived", data=df, palette="cool")
plt.title("survival rate by passenger class")
plt.xlabel("passenger class (1 = rich, 3 = poor")
plt.ylabel("survival rate")
plt.show()

# box plot
sns.boxplot(x="survived", y="fare", data=df, palette="muted")
plt.title("fare paid vs survival")
plt.xlabel("0 = died, 1 = survived")
plt.ylabel("fare")
plt.show()

# heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="pink", linewidths=0.5)
plt.title("correlation heatmap")
plt.show()



print(df['survived'].value_counts())
print(df['pclass'].value_counts())
print("\nRemaining missing value:\n", df.isnull().sum())

print("predicted:", Y_pred[:10])
print("Actual:", Y_test.values[:10])

print(df.head())
print("n\Shape:", df.shape)
print("\nInfo:")
print(df.info())
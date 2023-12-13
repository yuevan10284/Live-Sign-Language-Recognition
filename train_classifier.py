import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# load data
data_dict = pickle.load(open('data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# assign the labels to numeric values 
"""
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
"""
# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

#Use GridSearchCV for hyperparameter tuning (can be commented out if not needed)
'''
parameters = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
grid_search = GridSearchCV(model, parameters, cv=3, scoring='accuracy')
grid_search.fit(x_train, y_train)


best_model = grid_search.best_estimator_


best_model.fit(x_train, y_train)
'''

model.fit(x_train,y_train)
#y_pred = best_model.predict(x_test)
y_predict = model.predict(x_test)
"""
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the trained model and the label encoder

with open('model.pkl', 'wb') as model_file:
    pickle.dump({'model': best_model, 'label_encoder': label_encoder}, model_file)
    """
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
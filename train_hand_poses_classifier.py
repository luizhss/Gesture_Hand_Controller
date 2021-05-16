import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import pickle
import argparse
from hand_pose_transform import HandPoseTransform

parser = argparse.ArgumentParser(description="List Parameters.")
parser.add_argument("-d", "--dataset_path", type=str, default='dataset_train.csv',
                    help='Dataset filename path')
parser.add_argument("-s", "--save_path", type=str, default='handsPoseClassifier.pkl',
                    help='Path to save trained SVC model')
parser.add_argument("--test_size", type=float, default=0.2,
                    help='Float % Test Size')
args = parser.parse_args()

path_save_model = args.save_path
if not path_save_model.endswith('.pkl'):
    path_save_model += '.pkl'

dataset_path = args.dataset_path
test_size = args.test_size

# Load Dataset
print(f'Loading {dataset_path} dataset...')
df = pd.read_csv(dataset_path)
print(f'Dataset with shape={df.shape} Loaded', end='\n\n')

# Dataset Info
print('Qty of tuples per class')
len_per_class = df.groupby('class').count()['WRIST_x']
print([(key, len_per_class[key]) for key in len_per_class.keys()], end='\n\n')

# train test split
Y = df['class'].values
xcols = list(df.columns)
xcols.remove('hand')
xcols.remove('class')
X = df[xcols].values
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, random_state=42, stratify=Y)
print(f'Train Size {len(X_train)}')
print(f'Test Size {len(X_test)}')

# Train SVC inside of a Pipeline
scale_hands = True
if scale_hands:
    svc_model = Pipeline([
        ('scaler', HandPoseTransform()),
        ('svc', SVC(C=1, kernel='linear', random_state=42))
    ])
else:
    svc_model = SVC(C=1, kernel='linear', random_state=42)

print(f'Training SVC with {len(X_train)} tuples...')
svc_model.fit(X_train, Y_train)
print('Train Completed', end='\n\n')

# Test SVC
print(f'Testing SVC with {len(X_test)} tuples...', end='\n\n')
predict_svc_model = svc_model.predict(X_test)
print(classification_report(Y_test, predict_svc_model), end='\n\n')
print(accuracy_score(Y_test, predict_svc_model), end='\n\n')

# Grid Search SVC - Train/Validation
print('Grid Search Training Starting...')
# svc_model = SVC(random_state=42)
grid_svc = {'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
if scale_hands:  # For a Pipeline with multiple steps, we need to specify where param belongs
    for key in grid_svc.keys():
        grid_svc['svc__' + key] = grid_svc.pop(key)

svc_model_optimized = RandomizedSearchCV(estimator=svc_model, param_distributions=grid_svc,
                                         n_iter=10, cv=3, random_state=42)
svc_model_optimized.fit(X_train, Y_train)
print('Grid Search Training Completed', end='\n\n')

# Grid Search SVC - Test
print('Grid Search Test Starting...')
predict_optimized_svc_model = svc_model_optimized.predict(X_test)
print(classification_report(Y_test, predict_optimized_svc_model), end='\n\n')
print(accuracy_score(Y_test, predict_optimized_svc_model), end='\n\n')

# Get Best Parameters
best_parameters_svc = svc_model_optimized.best_params_
del svc_model_optimized, predict_optimized_svc_model
print('Best Parameters:')
print(best_parameters_svc, end='\n\n')

# Generate SVC Model - Best Parameters founded in Grid Search
print('Training Final SVC Model with Best Parameters')

# Add in that we want to predict probability instead of class (for multi-class estimates)
best_parameters_svc['svc__probability' if scale_hands else 'probability'] = True
svc_model.set_params(**best_parameters_svc)
svc_model.fit(X, Y)
print('Final training completed successfully', end='\n\n')

# Saving SVC Best Model
print('Saving Final SVC Model...')
pickle.dump(svc_model, open(path_save_model, "wb"))
print(f'Model Saved in {path_save_model}', end='\n\n')

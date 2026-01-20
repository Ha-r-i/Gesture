import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = './data'

def train_model():
    data = []
    labels = []
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} does not exist. Please run collect_data.py first.")
        return

    # Load data
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.pickle'):
            class_name = filename.split('.')[0]
            file_path = os.path.join(DATA_DIR, filename)
            
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)
            
            for sample in dataset:
                data.append(sample)
                labels.append(class_name)
    
    if len(data) == 0:
        print("No data found. Please run collect_data.py to collect gesture data.")
        return

    data = np.asarray(data)
    labels = np.asarray(labels)

    # Train/Test split
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    # Initialize model
    model = RandomForestClassifier()

    # Train
    print("Training model...")
    model.fit(x_train, y_train)

    # Predict
    y_predict = model.predict(x_test)

    # Evaluate
    score = accuracy_score(y_predict, y_test)
    print(f'{score * 100:.2f}% of samples were classified correctly !')

    # Save model
    with open('model.p', 'wb') as f:
        pickle.dump({'model': model}, f)
    
    print("Model saved to model.p")

if __name__ == "__main__":
    train_model()

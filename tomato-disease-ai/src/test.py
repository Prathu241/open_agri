def load_model(model_path):
    from keras.models import load_model
    return load_model(model_path)

def load_test_data(test_data_path):
    import pandas as pd
    return pd.read_csv(test_data_path)

def make_predictions(model, test_data):
    return model.predict(test_data)

def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return accuracy, report

def main():
    model_path = 'path/to/your/model.h5'  # Update with your model path
    test_data_path = 'path/to/your/test_data.csv'  # Update with your test data path

    model = load_model(model_path)
    test_data = load_test_data(test_data_path)

    # Assuming the test data has features and labels
    X_test = test_data.drop('label', axis=1)
    y_true = test_data['label']

    y_pred = make_predictions(model, X_test)
    
    accuracy, report = calculate_metrics(y_true, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

if __name__ == '__main__':
    main()
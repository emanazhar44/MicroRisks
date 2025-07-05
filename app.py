import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import gradio as gr

# --- Model Training ---
# 1. Synthetic Data Generation (6 features)
def create_synthetic_data(num_samples=200):
    data = pd.DataFrame({
        'Bacteroides': np.random.rand(num_samples) * np.random.choice([0.1, 0.8], num_samples, p=[0.7, 0.3]), # Often low, sometimes high
        'Firmicutes': np.random.rand(num_samples) * np.random.choice([0.1, 0.8], num_samples, p=[0.3, 0.7]),  # Often high, sometimes low
        'Actinobacteria': np.random.rand(num_samples) * 0.5, # Generally moderate
        'Proteobacteria': np.random.rand(num_samples) * np.random.choice([0.5, 0.1], num_samples, p=[0.6, 0.4]), # More often high in disease
        'Lactobacillus': np.random.rand(num_samples) * np.random.choice([0.8, 0.1], num_samples, p=[0.7, 0.3]), # More often high in health
        'Prevotella': np.random.rand(num_samples) * np.random.choice([0.5, 0.1], num_samples, p=[0.5, 0.5]),   # Variable
    })
    # Simple rule for labels (example, replace with real logic/data)
    # Diseased (1) if Firmicutes is high and Lactobacillus is low, or Proteobacteria is very high
    labels = np.zeros(num_samples)
    labels[((data['Firmicutes'] > 0.5) & (data['Lactobacillus'] < 0.2)) | (data['Proteobacteria'] > 0.35)] = 1
    data['label'] = labels.astype(int)
    return data

synthetic_df = create_synthetic_data(num_samples=300) # Increased samples
X = synthetic_df.drop('label', axis=1)
y = synthetic_df['label']

# 2. Train-test split
# Ensure that both classes are present in the training and testing sets if possible
# If the dataset is small or imbalanced, this might require stratification
if len(np.unique(y)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    # Handle the case where there's only one class in y (though synthetic data should prevent this)
    X_train, X_test, y_train, y_test = X, pd.DataFrame(), y, pd.Series() # Or handle error
    print("Warning: Dataset contains only one class. Model training might be trivial or fail.")


# 3. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
if not X_train.empty and len(np.unique(y_train)) > 1 :
    model.fit(X_train, y_train)
    # Evaluate (optional, for developer info)
    if not X_test.empty:
        y_pred_test = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    else:
        print("Test set is empty, skipping evaluation.")
else:
    print("Training data is empty or contains only one class. Model not trained.")
    accuracy = 0.0 # Default accuracy if model not trained


feature_names = ['Bacteroides', 'Firmicutes', 'Actinobacteria', 'Proteobacteria', 'Lactobacillus', 'Prevotella']

# --- Prediction Functions ---
def predict_single(*args):
    # args will be in the order of feature_names
    input_data_dict = {name: val for name, val in zip(feature_names, args)}
    input_df = pd.DataFrame([input_data_dict])
    
    # Ensure column order matches training
    input_df = input_df[X_train.columns] 

    if hasattr(model, 'classes_'): # Check if model is trained
        prediction_proba = model.predict_proba(input_df)[0]
        # Create a dictionary of class probabilities
        confidences = {model.classes_[i]: prob for i, prob in enumerate(prediction_proba)}
        predicted_label_code = np.argmax(prediction_proba)
        predicted_label = "Diseased" if model.classes_[predicted_label_code] == 1 else "Healthy"
        return predicted_label, confidences
    else:
        return "Model not trained", {}


def predict_batch(file_path_obj):
    if file_path_obj is None:
        return "No file uploaded.", None
    
    try:
        # Gradio gr.File provides a NamedTemporaryFile object, access its path via .name
        df = pd.read_csv(file_path_obj.name)
    except Exception as e:
        return f"Error reading CSV: {e}", None

    # Ensure all required features are present
    if not all(f_name in df.columns for f_name in feature_names):
        missing = set(feature_names) - set(df.columns)
        return f"CSV is missing required columns: {', '.join(missing)}. Required: {', '.join(feature_names)}", None

    df_features = df[feature_names] # Select and order features
    
    if hasattr(model, 'classes_'): # Check if model is trained
        predictions_codes = model.predict(df_features)
        predictions_labels = ['Diseased' if p == 1 else 'Healthy' for p in predictions_codes]
        df['Prediction'] = predictions_labels
        
        # Save predictions to a new CSV
        output_filename = "predictions_output.csv"
        df.to_csv(output_filename, index=False)
        return output_filename, f"Predictions saved to {output_filename}"
    else:
        return "Model not trained. Cannot make batch predictions.", None


# --- Gradio Interface ---
# Inputs for single prediction
single_sample_inputs = [
    gr.Slider(minimum=0, maximum=1, step=0.01, label=name) for name in feature_names
]

# Outputs for single prediction
single_sample_outputs = [
    gr.Textbox(label="Prediction Result"),
    gr.Label(label="Confidence Scores") 
]

# Interface for single sample prediction
single_tab = gr.Interface(
    fn=predict_single,
    inputs=single_sample_inputs,
    outputs=single_sample_outputs,
    title="🔬 Microbiome Disease Predictor (Single Sample)",
    description=f"Enter microbiome abundances to predict health status. Model Accuracy: {accuracy*100:.2f}% (on synthetic test data)"
)

# Interface for batch prediction
batch_tab = gr.Interface(
    fn=predict_batch,
    inputs=gr.File(label="Upload CSV with microbiome data", file_types=['.csv']),
    outputs=[gr.File(label="Download CSV with Predictions"), gr.Textbox(label="Status")],
    title="📊 Batch Predictor (Upload CSV)",
    description=f"Upload a CSV file with columns: {', '.join(feature_names)}."
)

# Combine interfaces into tabs
app = gr.TabbedInterface(
    [single_tab, batch_tab],
    tab_names=["Single Sample Prediction", "Batch CSV Prediction"]
)

if __name__ == '__main__':
    app.launch()

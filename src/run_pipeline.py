from data_preprocessing import load_data, preprocess_data
from feature_engineering import scale_features
from model_training import train_random_forest, save_model
from evaluation import evaluate_model


df = load_data("../data/MinoAI_dataset.csv")
df_clean = preprocess_data(df)
df_encoded = scale_features(df_clean)

model, X_test, y_test = train_random_forest(df_encoded)
metrics = evaluate_model(model, X_test, y_test)

save_model(model, "../models/random_forest_model.pkl")

print(metrics)

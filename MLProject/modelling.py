import sys
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def main(n_estimators, max_depth, dataset_path):
    TARGET_COLUMN = "churned"
    

    with mlflow.start_run():
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File dataset tidak ditemukan di: {dataset_path}")

        print(f"Loading data from {dataset_path}...")
        df = pd.read_csv(dataset_path)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training with n_estimators={n_estimators}, max_depth={max_depth}...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")

        print(f"Accuracy: {acc:.4f}")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset_source", dataset_path)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature
        )

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix (Acc: {acc:.2f})")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        plt.figure(figsize=(8, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X.columns
        sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
        plt.title("Feature Importance")
        plt.tight_layout()
        
        fi_path = "feature_importance.png"
        plt.savefig(fi_path)
        plt.close()
        mlflow.log_artifact(fi_path)

        if os.path.exists(cm_path): os.remove(cm_path)
        if os.path.exists(fi_path): os.remove(fi_path)

if __name__ == "__main__":
    main(
        n_estimators=int(sys.argv[1]),
        max_depth=int(sys.argv[2]),
        dataset_path=sys.argv[3]
    )
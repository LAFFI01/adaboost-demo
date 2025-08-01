# === Importing libraries ===
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import numpy as np
import time

# === Streamlit page config ===
st.set_page_config(layout="wide")
st.title("⚙️ AdaBoost Hyperparameter Tuner (Classifier + Regressor)")

# === Sidebar: Choose between classification or regression ===
model_type = st.sidebar.radio("Select AdaBoost Type", ["Classifier", "Regressor"])

# === Load appropriate dataset ===
if model_type == "Classifier":
    data = load_iris()
    X = data.data[:, :2]  # use only 2 features for easy visualization
    y = data.target
else:
    data = fetch_california_housing()
    X = data.data[:, :2]  # use only 2 features for plotting
    y = data.target

# === Create 2 columns for layout ===
col1, col2 = st.columns(2)

# === Right column: Show hyperparameter tuning options ===
with col2:
    st.header("🔧 Hyperparameters")

    # Basic tuning parameters
    n_estimators = st.slider("n_estimators", 10, 500, 50, step=10)
    learning_rate = st.slider("learning_rate", 0.01, 2.0, 1.0, step=0.01)
    max_depth = st.slider("base_estimator max_depth", 1, 10, 1)
    test_size = st.slider("Test Size", 0.1, 0.5, 0.3, step=0.05)
    random_state = st.number_input("random_state (optional)", min_value=0, value=42)

    # Conditional tuning options for Classifier/Regressor
    if model_type == "Classifier":
        algorithm = st.selectbox("algorithm", [ "SAMME"])
    else:
        loss = st.selectbox("loss", ["linear", "square", "exponential"])

    # Run button
    run_model = st.button("🚀 Train Model")

# === Train the model if button is clicked ===
if run_model:
    st.success("Training started...")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # === Create AdaBoost Classifier or Regressor ===
    if model_type == "Classifier":
        base = DecisionTreeClassifier(max_depth=max_depth)
        model = AdaBoostClassifier(
            estimator=base,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state
        )
    else:
        base = DecisionTreeRegressor(max_depth=max_depth)
        model = AdaBoostRegressor(
            estimator=base,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=random_state
        )

    # Train model and record time
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    # Predict on test set
    y_pred = model.predict(X_test)

    # === Left column: Show results ===
    with col1:
        st.header("📊 Results")
        st.write(f"🕒 Training Time: `{end - start:.2f} sec`")

        # Classification metrics
        if model_type == "Classifier":
            acc = accuracy_score(y_test, y_pred)
            st.write(f"✅ Accuracy: `{acc:.2f}`")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        # Regression metrics
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"📉 Mean Squared Error: `{mse:.2f}`")
            st.write(f"📈 R2 Score: `{r2:.2f}`")

        # === Function: Plot decision or regression surface ===
        def plot_boundary(X, y, model, is_classifier=True):
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                 np.linspace(y_min, y_max, 200))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(5, 4))
            if is_classifier:
                # For classification, show contour map
                plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
                plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
            else:
                # For regression, show gradient
                cp = plt.contourf(xx, yy, Z, cmap="viridis")
                plt.colorbar(cp)
                plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="viridis")

            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Decision Surface" if is_classifier else "Regression Surface")
            st.pyplot(plt)

        # Call the plot function
        plot_boundary(X_test, y_test, model, is_classifier=(model_type == "Classifier"))

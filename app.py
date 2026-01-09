import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import r2_score, accuracy_score, make_scorer

st.set_page_config(page_title="Advanced AutoML Pipeline", layout="wide")
st.title("🚀 Advanced AutoML Pipeline 🤖")
st.markdown(
    "Upload a dataset → Select target → Auto clean → Auto train → Download best model"
)
# ================= Upload =================
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select Target Column", df.columns)
        # ================= Cleaning =================
    if target_col:
        before = df.shape[0]
        df = df.dropna(subset=[target_col])
        st.info(f"Removed {before - df.shape[0]} Rows with null values in target colunm")

        # ================= Detect Problem =================
        if df[target_col].dtype in ["int64", "float64"] and df[target_col].nunique() > 10:
            problem_type = "Regression"
        else:
            problem_type = "Classification"

        st.success(f"Detected: **{problem_type}**")


        num_cols_all = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if target_col in num_cols_all:
            num_cols_all.remove(target_col)

        def remove_outliers_iqr(data, cols):
            for c in cols:
                Q1, Q3 = data[c].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                data = data[(data[c] >= Q1 - 1.5*IQR) & (data[c] <= Q3 + 1.5*IQR)]
            return data

        before = df.shape[0]
        df = remove_outliers_iqr(df, num_cols_all)
        st.info(f"Removed {before - df.shape[0]} outlier rows in target colunm")
        # ================= Download Cleaned CSV =================
        cleaned_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned CSV (Removed rows with null values,duplicates and outliers in target)",
            data=cleaned_csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
        # ================= Split X & y =================
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if problem_type == "Classification":
            y = LabelEncoder().fit_transform(y)

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        st.write("🔢 Numeric Columns:", num_cols)
        st.write("🔤 Categorical Columns:", cat_cols)

        # ================= Preprocessing =================
        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ================= Models =================
        if problem_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "SVM": SVR()
            }
            scoring = make_scorer(r2_score)
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC(),
                "Naive Bayes": GaussianNB()
            }
            scoring = make_scorer(accuracy_score)

        # ================= CV Evaluation =================
        st.subheader("📊 Model Comparison")

        best_score, best_pipe, best_name = -np.inf, None, None

        for name, model in models.items():
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            try:
                cv = min(5, len(y_train))
                if problem_type == "Classification":
                    cv = min(cv, np.bincount(y_train).min())

                scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
                score = scores.mean()
                st.write(f"**{name}** → {score:.4f}")

                if score > best_score:
                    best_score, best_pipe, best_name = score, pipe, name
            except:
                st.warning(f"{name} failed CV")

        st.success(f"🏆 Best Base Model: **{best_name}**")

        # ================= Hyperparameter Tuning =================
        st.subheader("⚙️ Hyperparameter Tuning")

        param_grids = {
            "Random Forest": {"model__n_estimators": [50, 100], "model__max_depth": [None, 10]},
            "Decision Tree": {"model__max_depth": [None, 5, 10]},
            "SVM": {"model__C": [0.1, 1]}
        }

        if best_name in param_grids:
            grid = GridSearchCV(
                best_pipe,
                param_grids[best_name],
                cv=3,
                scoring=scoring
            )
            grid.fit(X_train, y_train)
            best_pipe = grid.best_estimator_
            st.info(f"Best Params⚙️: {grid.best_params_}")
        else:
            best_pipe.fit(X_train, y_train)

 # ================= Save Model =================
joblib.dump(best_pipe, "best_model.pkl")
st.download_button(
            "⬇️ Download Trained Model (.pkl)",
            data=open("best_model.pkl", "rb"),
            file_name="best_model.pkl"
        )

st.success("✅ Advanced AutoML Pipeline Completed")
# =========================================================
# ============== ADVANCED PREDICTION UI ===================
# =========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Prediction")

if st.sidebar.checkbox("Open Prediction Panel"):
    st.title("🔮 Smart Prediction System")

    model_file = st.file_uploader("📦 Upload Trained Model (.pkl)", type=["pkl"])
    ref_csv = st.file_uploader("📄 Upload Training CSV (same structure)", type=["csv"])

    if model_file and ref_csv:
        model = joblib.load(model_file)
        ref_df = pd.read_csv(ref_csv)

        st.success("Model and reference dataset loaded successfully")

        feature_names = model.named_steps["prep"].feature_names_in_
        input_data = {}

        st.subheader("🧾 Enter Feature Values")

        for col in feature_names:
            if ref_df[col].dtype in ["int64", "float64"]:
                min_val = float(ref_df[col].min())
                max_val = float(ref_df[col].max())
                mean_val = float(ref_df[col].mean())

                input_data[col] = st.slider(
                    col,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val
                )
            else:
                categories = ref_df[col].dropna().unique().tolist()
                input_data[col] = st.selectbox(col, categories)

        input_df = pd.DataFrame([input_data])
        st.dataframe(input_df)

        if st.button("🚀 Predict"):
            prediction = model.predict(input_df)
            st.success(f"🎯 Prediction Result: {prediction[0]:.2f}")

            if hasattr(model.named_steps["model"], "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                proba_percent = np.round(proba * 100, 2)
                classes = model.named_steps["model"].classes_

                prob_df = pd.DataFrame({
                    "Class": classes,
                    "Probability (%)": proba_percent
                }).sort_values("Probability (%)", ascending=False)

                confidence = prob_df.iloc[0]["Probability (%)"]

                st.metric("🔐 Prediction Confidence", f"{confidence}%")
                st.subheader("📊 Probability Comparison")
                st.bar_chart(prob_df.set_index("Class"))

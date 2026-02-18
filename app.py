from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import os
import io
import base64
import requests
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────
ENCODED_KEY = "c2stb3ItdjEtNTlmYTdhZDE2MzQ1OGM3ZGY2OGRhYjAyNzY5NDAzYTU1MDMwMWM1MzQ5NjRkYzM3NzI5NTYyZWU3NWQ2YzVhMw=="
OPENROUTER_API_KEY = base64.b64decode(ENCODED_KEY).decode()
MODEL = "openai/gpt-3.5-turbo"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "diabetes.csv")

# ─── DATA & MODEL SETUP ────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH)

# Replace 0s with NaN for biological columns & impute with median
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df = df_raw.copy()
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
for col in cols_to_fix:
    df[col].fillna(df[col].median(), inplace=True)

FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[FEATURES]
y = df['Outcome']

# Impute NaN values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train multiple models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

trained_models = {}
model_metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    trained_models[name] = model
    model_metrics[name] = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_prob) * 100, 2),
        "cv_score": round(float(cross_val_score(model, X_scaled, y, cv=5).mean()) * 100, 2)
    }

# Best model
best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['accuracy'])
best_model = trained_models[best_model_name]
rf_model = trained_models["Random Forest"]
rf_importances = rf_model.feature_importances_

# ─── HELPER: chart to base64 ───────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

# ─── ROUTES ────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html",
                           model_names=list(trained_models.keys()),
                           best=best_model_name)


@app.route("/explore")
def explore():
    return render_template("explore.html")


@app.route("/chat")
def chat():
    return render_template("chat.html")


@app.route("/models")
def models_page():
    return render_template("models.html", metrics=model_metrics, best=best_model_name)


# ─── API: Dashboard stats ──────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    diabetic = int(df['Outcome'].sum())
    total = len(df)
    avg_glucose = round(df['Glucose'].mean(), 1)
    avg_bmi = round(df['BMI'].mean(), 1)
    avg_age = round(df['Age'].mean(), 1)

    return jsonify({
        "total": total,
        "diabetic": diabetic,
        "non_diabetic": total - diabetic,
        "prevalence": round(diabetic / total * 100, 1),
        "avg_glucose": avg_glucose,
        "avg_bmi": avg_bmi,
        "avg_age": avg_age,
        "best_model": best_model_name,
        "best_accuracy": model_metrics[best_model_name]["accuracy"]
    })


@app.route("/api/chart/outcome_dist")
def chart_outcome():
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    counts = df['Outcome'].value_counts()
    colors = ['#22d3ee', '#f43f5e']
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=['Non-Diabetic', 'Diabetic'],
        autopct='%1.1f%%', colors=colors, startangle=90,
        textprops={'color': 'white', 'fontsize': 12},
        wedgeprops={'edgecolor': '#0f172a', 'linewidth': 2})
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color('white')
    ax.set_title('Outcome Distribution', color='white', fontsize=14, pad=15)
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/age_dist")
def chart_age():
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    df[df['Outcome'] == 0]['Age'].hist(ax=ax, bins=20, alpha=0.8,
                                        color='#22d3ee', label='Non-Diabetic')
    df[df['Outcome'] == 1]['Age'].hist(ax=ax, bins=20, alpha=0.8,
                                        color='#f43f5e', label='Diabetic')
    ax.set_xlabel('Age', color='#94a3b8')
    ax.set_ylabel('Count', color='#94a3b8')
    ax.set_title('Age Distribution by Outcome', color='white', fontsize=13)
    ax.tick_params(colors='#94a3b8')
    ax.legend(facecolor='#1e293b', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/feature_importance")
def chart_importance():
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    sorted_idx = np.argsort(rf_importances)
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(FEATURES)))
    ax.barh([FEATURES[i] for i in sorted_idx],
            rf_importances[sorted_idx], color=colors)
    ax.set_xlabel('Importance', color='#94a3b8')
    ax.set_title('Feature Importance (RF)', color='white', fontsize=13)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/glucose_box")
def chart_glucose():
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    data0 = df[df['Outcome'] == 0]['Glucose']
    data1 = df[df['Outcome'] == 1]['Glucose']
    bp = ax.boxplot([data0, data1], patch_artist=True,
                    labels=['Non-Diabetic', 'Diabetic'],
                    medianprops={'color': 'white', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], ['#22d3ee', '#f43f5e']):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_ylabel('Glucose Level (mg/dL)', color='#94a3b8')
    ax.set_title('Glucose by Outcome', color='white', fontsize=13)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/corr")
def chart_corr():
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    corr = df[FEATURES + ['Outcome']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, linewidths=0.5, linecolor='#0f172a',
                annot_kws={'size': 8, 'color': 'white'},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', color='white', fontsize=13, pad=10)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/roc")
def chart_roc():
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    colors_roc = ['#22d3ee', '#f43f5e', '#a78bfa', '#34d399']
    for (name, model), color in zip(trained_models.items(), colors_roc):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.2f})')
    ax.plot([0, 1], [0, 1], 'w--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', color='#94a3b8')
    ax.set_ylabel('True Positive Rate', color='#94a3b8')
    ax.set_title('ROC Curves — All Models', color='white', fontsize=13)
    ax.tick_params(colors='#94a3b8')
    ax.legend(facecolor='#1e293b', labelcolor='white', fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/scatter")
def chart_scatter():
    feat_x = request.args.get("x", "Glucose")
    feat_y = request.args.get("y", "BMI")
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    for outcome, color, label in [(0, '#22d3ee', 'Non-Diabetic'),
                                   (1, '#f43f5e', 'Diabetic')]:
        subset = df[df['Outcome'] == outcome]
        ax.scatter(subset[feat_x], subset[feat_y],
                   alpha=0.5, s=20, c=color, label=label)
    ax.set_xlabel(feat_x, color='#94a3b8')
    ax.set_ylabel(feat_y, color='#94a3b8')
    ax.set_title(f'{feat_x} vs {feat_y}', color='white', fontsize=13)
    ax.tick_params(colors='#94a3b8')
    ax.legend(facecolor='#1e293b', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    return jsonify({"img": fig_to_b64(fig)})


@app.route("/api/chart/confusion/<model_name>")
def chart_confusion(model_name):
    model = trained_models.get(model_name, best_model)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'],
                annot_kws={'size': 14, 'color': 'white'},
                linewidths=1, linecolor='#0f172a')
    ax.set_title(f'Confusion Matrix — {model_name}', color='white', fontsize=12)
    ax.tick_params(colors='#94a3b8')
    return jsonify({"img": fig_to_b64(fig)})


# ─── API: Predict ──────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    model_name = data.get("model", best_model_name)
    model = trained_models.get(model_name, best_model)

    try:
        values = [
            float(data.get("Pregnancies", 0)),
            float(data.get("Glucose", 120)),
            float(data.get("BloodPressure", 70)),
            float(data.get("SkinThickness", 20)),
            float(data.get("Insulin", 80)),
            float(data.get("BMI", 25)),
            float(data.get("DiabetesPedigreeFunction", 0.5)),
            float(data.get("Age", 30))
        ]
    except ValueError:
        return jsonify({"error": "Invalid input values"}), 400

    scaled = scaler.transform(imputer.transform([values]))
    prob = model.predict_proba(scaled)[0][1]
    prediction = int(model.predict(scaled)[0])

    if prob < 0.35:
        risk = "Low"
        risk_color = "#22d3ee"
        advice = "Your indicators look healthy. Maintain a balanced diet, regular exercise, and annual check-ups."
    elif prob < 0.60:
        risk = "Moderate"
        risk_color = "#fbbf24"
        advice = "Some indicators warrant attention. Consider consulting a healthcare professional and monitoring glucose levels."
    else:
        risk = "High"
        risk_color = "#f43f5e"
        advice = "Several indicators suggest elevated risk. Please consult a doctor for proper medical evaluation as soon as possible."

    # Feature contributions (SHAP-lite: multiply scaled input by feature importance)
    contributions = {FEATURES[i]: round(abs(scaled[0][i]) * rf_importances[i], 4)
                     for i in range(len(FEATURES))}

    return jsonify({
        "prediction": prediction,
        "probability": round(prob * 100, 1),
        "risk": risk,
        "risk_color": risk_color,
        "advice": advice,
        "model_used": model_name,
        "contributions": contributions,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


# ─── API: Dataset preview ──────────────────────────────────────────────────
@app.route("/api/dataset")
def api_dataset():
    page = int(request.args.get("page", 1))
    per_page = 15
    start = (page - 1) * per_page
    end = start + per_page
    total = len(df)
    rows = df.iloc[start:end].to_dict(orient="records")
    return jsonify({
        "rows": rows,
        "total": total,
        "page": page,
        "pages": (total + per_page - 1) // per_page
    })


@app.route("/api/dataset/summary")
def api_summary():
    summary = df.describe().round(2).to_dict()
    return jsonify(summary)


# ─── API: AI Chat ──────────────────────────────────────────────────────────
@app.route("/api/ask", methods=["POST"])
def api_ask():
    messages = request.json.get("messages", [])
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are DiabetesAI, an expert educational assistant specialized in diabetes awareness and prevention.\n"
                    "You have deep knowledge of Type 1, Type 2, and gestational diabetes.\n\n"
                    "Your behavior:\n"
                    "- Ask ONE clear question at a time, like a knowledgeable doctor\n"
                    "- Collect: age, gender, BMI/weight, family history, glucose levels (if known), symptoms, lifestyle\n"
                    "- Use medical knowledge to give insightful, educational explanations\n"
                    "- At assessment end, provide: Risk Level (Low/Medium/High), key risk factors, lifestyle recommendations\n"
                    "- ALWAYS end with: '⚠️ This is educational only, not a medical diagnosis. Consult a doctor.'\n\n"
                    "Style:\n"
                    "- Warm, professional, empathetic tone\n"
                    "- Short replies (2-5 lines max)\n"
                    "- Use simple language, avoid excessive jargon\n"
                    "- Use bold for key terms\n"
                    "- Never diagnose. Never scare. Always educate."
                )
            }
        ] + messages
    }
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=payload, timeout=60)
        return jsonify(r.json()["choices"][0]["message"])
    except Exception as e:
        return jsonify({"role": "assistant",
                        "content": "Sorry, the AI service is temporarily unavailable. Please try again."})


# ─── API: Model metrics ────────────────────────────────────────────────────
@app.route("/api/model_metrics")
def api_model_metrics():
    return jsonify({"metrics": model_metrics, "best": best_model_name})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

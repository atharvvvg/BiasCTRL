# BiasCTRL: An Interactive Workbench for AI Development

**BiasCTRL is a comprehensive, web-based platform designed to empower developers and data scientists in building fairer and more responsible AI systems. It provides an interactive environment to systematically discover, analyze, visualize, and mitigate biases within machine learning models and the datasets they are trained on.**



The project aims to bridge the gap between theoretical fairness concepts and practical implementation by offering a user-friendly workflow that integrates data analysis, model training, state-of-the-art bias mitigation techniques, robust fairness evaluation, and insightful model explanations.

**Current Status: Fully Functional Backend API & Foundational Frontend**

The core backend, built with Python and FastAPI, is feature-complete, offering a rich set of tools for AI development. A foundational React frontend (using Vite) is in place, enabling interaction with all backend functionalities and providing initial visualizations and structured display of results. The next major phase is to significantly enhance the frontend's interactivity, visualization capabilities, and user experience.

## The Challenge of AI Bias

Machine learning models, despite their power, can inadvertently learn and perpetuate societal biases present in historical data. These biases can lead to unfair or discriminatory outcomes when models are deployed in critical domains like loan applications, hiring, criminal justice, and healthcare. Addressing AI bias is not just a technical challenge but an ethical imperative to ensure AI systems are equitable and just.

BiasCTRL addresses this by providing tools to:

1.  **Identify Bias:** Uncover statistical disparities in how a model performs or treats different demographic groups (defined by sensitive attributes like race, gender, age, etc.).
2.  **Understand Bias:** Explore _why_ a model might be biased through explainability techniques that reveal which features drive its decisions for different groups.
3.  **Mitigate Bias:** Experiment with various pre-processing, in-processing (planned), and post-processing (planned) techniques to reduce identified biases.
4.  **Evaluate Trade-offs:** Assess the impact of mitigation strategies on both model fairness and predictive performance, as these often involve a delicate balance.

## Core Concepts & Features

BiasCTRL is built around a workflow that guides users through the lifecycle of responsible AI development:

### 1. Data Ingestion & Analysis (`/upload`, `/analyze`)

- **Dataset Upload:** Users can upload their datasets in CSV format.
- **Automated Data Profiling:** Upon upload and configuration (specifying target and sensitive attributes), the system provides:
  - **Dataset Overview:** Number of rows, columns, and a list of all feature names.
  - **Target Variable Analysis:** Distribution of the outcome variable (e.g., class balance in classification).
  - **Sensitive Attribute Analysis:**
    - Distribution of individuals across different groups for each specified sensitive attribute (e.g., count of individuals per race, per gender).
    - Target variable distribution _within each group_ of a sensitive attribute. This helps identify initial representation or outcome disparities even before model training.
  - This initial analysis is crucial for understanding potential sources of bias originating from the data itself.

### 2. Baseline Model Training (`/train_baseline`)

- **Automated Training:** Users can train a baseline classification model (currently RandomForestClassifier with standard scikit-learn preprocessing like imputation, scaling, and one-hot encoding).
- **Performance Evaluation:** The baseline model's predictive performance is evaluated using standard metrics (Accuracy, Precision, Recall, F1-score) on a held-out test set.
- **Disaggregated Performance:** Critically, these performance metrics are also **disaggregated** and reported for each group within every specified sensitive attribute. This reveals if the model performs significantly better or worse for certain demographic subgroups.
- **Artifact Storage:** The trained scikit-learn pipeline (preprocessor + model) and detailed metadata (features used, sensitive attributes defined, performance metrics, paths to artifacts) are saved to disk (`models_cache/`).

### 3. Bias Mitigation Strategies

BiasCTRL allows users to apply and evaluate different techniques to reduce unwanted bias. Mitigated models are trained and saved separately, allowing for direct comparison with the baseline.

- **Reweighing (`/mitigate_reweigh`):**
  - **Concept (Pre-processing):** This technique assigns different weights to samples in the training dataset. The goal is to adjust the influence of different groups during model training to achieve a fairer outcome, often targeting **Demographic Parity** (equality of selection rates).
  - **Implementation:** Weights are calculated such that samples belonging to groups that are under-predicted for the positive class (or over-represented overall for a given outcome) are given higher importance, and vice-versa. The model is then trained using these sample weights. The specific formula used aims to make the weighted joint distribution of sensitive attributes and outcomes closer to the product of their marginal distributions.
  - **User Input:** The user specifies which single sensitive attribute to base the reweighing on.
- **Random Oversampling (`/mitigate_oversample`):**
  - **Concept (Pre-processing):** Addresses class imbalance in the target variable, which can sometimes be correlated with fairness issues across sensitive groups.
  - **Implementation:** Uses `RandomOverSampler` from the `imblearn` library. This technique randomly duplicates instances from the minority class(es) in the training set until the target classes are balanced. This is applied _after_ initial feature preprocessing but _before_ the final model training step, typically within an `imblearn.pipeline.Pipeline`.
  - While primarily for target imbalance, its impact on fairness metrics across sensitive groups is evaluated.

### 4. Comprehensive Fairness Evaluation (`/calculate_fairness`)

- **Flexibility:** This endpoint can calculate fairness metrics for _any_ trained model (baseline or mitigated) whose pipeline has been saved. The user provides the path to the model's `.joblib` pipeline file.
- **Multi-Attribute Evaluation:** Fairness metrics are calculated and reported _for each sensitive attribute_ specified by the user (e.g., results for "race" and results for "gender" separately).
- **Key Fairness Metrics Provided (per sensitive attribute):**
  - **Overall Model Performance on Test Set:** (Accuracy, Precision, Recall, F1, True Positive Rate, False Positive Rate, Selection Rate, Count). This is the model's performance on the entire test set, calculated during this fairness evaluation step.
  - **Metrics by Group:** Each of the above performance metrics is broken down and reported for every individual group within the sensitive attribute (e.g., accuracy for "White", accuracy for "Black", etc.).
  - **Fairness Disparities (between groups):**
    - **Difference:** The maximum difference in a metric between any two groups (e.g., `accuracy_difference`).
    - **Ratio:** The ratio of the minimum group metric to the maximum group metric (e.g., `selection_rate_ratio`, also known as Disparate Impact if applied to selection rates).
  - **Standard Fairness Definitions:**
    - **Demographic Parity (Statistical Parity):** Aims for equal selection rates across groups. Reported as difference and ratio.
    - **Equal Opportunity:** Aims for equal True Positive Rates (recall) across groups for the positive outcome. Reported as difference.
    - **Equalized Odds:** Aims for equal True Positive Rates _and_ equal False Positive Rates across groups. Reported as the maximum of TPR and FPR differences.

### 5. Model Explainability (`/explain_model`)

- **Understanding "Why":** To truly address bias, it's important to understand _why_ a model makes certain predictions and if it relies on sensitive features (or their proxies) inappropriately.
- **SHAP (SHapley Additive exPlanations):**
  - **Concept:** A game theory-based approach to explain the output of any machine learning model. SHAP values quantify the contribution of each feature to a specific prediction.
  - **Implementation:** Uses the `shap` library, specifically `KernelExplainer` for model-agnostic explanations of the scikit-learn pipelines.
  - **Output:** Provides **global feature importance** – the average absolute SHAP value for each feature across a sample of the data. This indicates which features have the most impact on the model's output overall.
  - The endpoint takes a pipeline path and returns these importance scores.

### 6. Model Comparison (`/compare_models`)

- **Evaluating Mitigation Impact:** The true test of a mitigation technique is to see its effect relative to the baseline or other interventions.
- **Side-by-Side Reporting:** This endpoint takes paths to two model pipelines (e.g., a baseline and a mitigated model) and the original data evaluation parameters.
- **Output:** Returns a structured JSON comparing key fairness metrics (overall accuracy from the fairness eval, disparities, and standard definitions) for both models, broken down _for each specified sensitive attribute_. This allows users to directly see:
  - Did the mitigation improve fairness for the targeted attribute?
  - Did it inadvertently worsen fairness for other attributes (fairness gerrymandering)?
  - What was the impact on overall accuracy?

## Tech Stack

- **Backend:**
  - **Language:** Python 3.9+
  - **API Framework:** FastAPI (for its speed, ease of use, and automatic docs)
  - **Web Server:** Uvicorn (ASGI server)
  - **Data Handling:** Pandas
  - **Machine Learning:** Scikit-learn (for models, preprocessing, metrics)
  - **Fairness:** Fairlearn (for `MetricFrame` and fairness-specific metrics)
  - **Resampling:** Imbalanced-learn (for `RandomOverSampler`)
  - **Explainability:** SHAP (for `KernelExplainer`)
  - **Serialization:** Joblib (for scikit-learn pipelines), JSON (for metadata)
- **Frontend:**
  - **Framework/Library:** React
  - **Build Tool:** Vite (for fast development and optimized builds)
  - **Language:** JavaScript (JSX)
  - **API Client:** Axios
  - **Charting:** Recharts
  - **Styling:** CSS (currently leveraging Vite's defaults with custom additions)

## Project Directory Structure

```
biasctrl/
├── backend/ # All Python backend code
│ ├── .gitignore
│ ├── requirements.txt
│ ├── main.py # FastAPI application, API endpoints
│ ├── core/ # Core business logic modules
│ │ ├── analysis.py # Data loading and initial analysis
│ │ ├── models.py # Model training (baseline, mitigated), saving
│ │ ├── fairness.py # Fairness metric calculations
│ │ └── explainability.py # SHAP explanations
│ ├── uploads/ # Temporary storage for user-uploaded CSVs (gitignored)
│ ├── models_cache/ # Storage for trained .joblib pipelines and .meta.json files (gitignored)
│ └── data/ # Sample datasets like adult.csv
│
└── frontend/ # All React frontend code
├── .gitignore
├── package.json
├── vite.config.js # Vite build/dev configuration
├── index.html # Main HTML
├── public/ # Static assets served directly
└── src/ # React application source
│ └── App.jsx # Main application component, state management, layout
│ └── main.jsx # React DOM rendering entry point
│ └── services/
│ │ └── api.js # Centralized API call functions
│ └── components/ # Reusable UI components (e.g., for displaying metrics, charts)
│ │ └── AnalysisDisplay.jsx
│ │ └── MetricsTable.jsx
│ │ └── FairnessDisplay.jsx
│ │ └── ShapChart.jsx
```

## Getting Started

### Prerequisites

- Python (version 3.9 or newer recommended)
- `pip` (Python package installer)
- Node.js (version 16.x or newer recommended for Vite)
- `npm` (Node package manager, comes with Node.js) or `yarn`

### Backend Setup & Execution

1.  **Clone the Repository:**
    ```bash
    # git clone <repository-url>
    # cd biasctrl/backend
    ```
    (Or navigate to your existing `biasctrl/backend` directory)
2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows:
    # venv\Scripts\activate
    # macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional) Add Sample Data:**
    If you wish to test with the Adult Census Income dataset, download it and place `adult.csv` into the `biasctrl/backend/data/` directory.
5.  **Run Backend Server:**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be accessible at `http://127.0.0.1:8000`.
    Interactive API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

### Frontend Setup & Execution

1.  **Navigate to Frontend Directory:**
    (From the `biasctrl` root)
    ```bash
    cd frontend
    ```
2.  **Install Dependencies:**
    ```bash
    npm install
    # or: yarn install
    ```
3.  **Run Frontend Development Server:**
    ```bash
    npm run dev
    # or: yarn dev
    ```
    The frontend application will typically open in your browser at `http://localhost:5173`.

**Note:** The backend server must be running for the frontend to function correctly as it relies on API calls to the backend. CORS is pre-configured in `backend/main.py` to allow requests from the default Vite development server port.

## Workflow Example using the API (via `/docs`)

1.  **Upload Data:** Use `POST /upload` to upload your `adult.csv` (or other dataset). Note the `filename` in the response.
2.  **Analyze Data:** Use `POST /analyze` with the `filename`, `target_column` (e.g., "income"), and `sensitive_attribute_columns` (e.g., "race,gender"). Review the distributions.
3.  **Train Baseline:** Use `POST /train_baseline` with the same parameters. Note the `pipeline_path` from the `model_info` in the response (e.g., `models_cache/adult_pipeline.joblib`).
4.  **Evaluate Baseline Fairness:** Use `POST /calculate_fairness`. Provide `filename`, `target_column`, `sensitive_attribute_columns`. You can omit `pipeline_path` to use the default baseline path, or provide the specific path from step 3. The response will show fairness metrics for each sensitive attribute.
5.  **Apply Mitigation:**
    - Use `POST /mitigate_reweigh` with `filename`, `target_column`, `sensitive_attribute_columns`, and `reweigh_attribute` (e.g., "race"). Note the new `pipeline_path` for the reweighed model (e.g., `models_cache/adult_reweighed_race_pipeline.joblib`).
    - Or, use `POST /mitigate_oversample`.
6.  **Compare Models:** Use `POST /compare_models`. Provide:
    - `baseline_pipeline_path`: (e.g., `models_cache/adult_pipeline.joblib`)
    - `mitigated_pipeline_path`: (e.g., `models_cache/adult_reweighed_race_pipeline.joblib`)
    - `filename`, `target_column`, `sensitive_attribute_columns`.
      Review the side-by-side fairness metrics for each sensitive attribute.
7.  **Explain a Model:** Use `POST /explain_model`. Provide `pipeline_path` (for either baseline or a mitigated model) and `n_samples`. Review the global feature importance.

## Future Roadmap

While the backend is robust, the project has significant potential for growth:

- **Simple script to start both backend and frontend**
- **Advanced Frontend UI/UX:**
  - Rich interactive visualizations for all metrics and explanations (beyond current tables/JSON).
  - Intuitive step-by-step workflow guidance for users.
  - Dynamic selection of models for comparison from a list of trained artifacts.
  - Visual SHAP summary plots, dependence plots, per-group explanations.
- **Expanded Mitigation Toolkit:**
  - **Pre-processing:** Add `DisparateImpactRemover` (Fairlearn), m ore advanced resampling (SMOTE).
  - **In-processing:** Integrate fairness-constrained algorithms (e.g., `ExponentiatedGradient`, `GridSearch` with fairness constraints from Fairlearn).
  - **Post-processing:** Implement `ThresholdOptimizer` (Fairlearn) to adjust prediction thresholds.
- **Deeper Explainability:**
  - Local SHAP explanations (for individual predictions).
  - Integration of other XAI techniques like LIME.
- **Intersectional Fairness:** Extend analysis and mitigation to consider combinations of multiple sensitive attributes (e.g., bias towards "Black females").
- **Model & Data Flexibility:**
  - Support for uploading/connecting pre-trained models.
  - More robust automated data type detection and cleaning options.
- **Persistence & Scalability:**
  - Utilize a database (e.g., PostgreSQL, MongoDB) for storing model metadata, experiment results, and user data, instead of relying solely on file system storage.
- **User Management & Collaboration (Long-term):**
  - User accounts, project spaces, and collaborative features.
- **Deployment:** Strategies for deploying BiasCTRL as a web service.

## Contributing

[Contribution guidelines for this project](CONTRIBUTING.md)

## License

[MIT License](LICENSE)

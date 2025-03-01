#####This code classifies wastewater treatment plant flow data into three categories—normal, high, and very high—based on historical flow distributions. 
####It applies machine learning models (Logistic Regression, Support Vector Machine, and Random Forest) to predict future flow classifications using engineered features like rolling statistics, rainfall lags, and seasonal interactions. 
####To address class imbalance, the code implements synthetic data generation techniques, including Temporal ADASYN and Time-Series SMOTE, enhancing the representation of extreme flow events. 
#######Finally, it evaluates model performance through classification metrics, confusion matrices, and visualizations to assess predictive accuracy.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, matthews_corrcoef
    )
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import holidays
import warnings
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

data = pd.read_csv('plant_flow_data.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
flows = data['flow'].sort_values(ascending=False).reset_index(drop=True)
Q3 = np.percentile(flows, 97)  # Very High Flow threshold (Top 3%)
Q15 = np.percentile(flows, 85)  # High Flow threshold (Next 15%)

def classify_flow(flow):
    if flow > Q3:
        return "very high"
    elif Q15 < flow <= Q3:
        return "high"
    else:
        return "normal"

data['flow_class'] = data['flow'].apply(classify_flow)
def flow_class(data, title="Different flow classifications"):
    plt.figure(figsize=(12, 6))
    unique_classes = data['flow_class'].unique()
    for flow_class in unique_classes:
        class_data = data[data['flow_class'] == flow_class]
        plt.scatter(
            class_data.index,
            class_data['flow'],
            label=flow_class,
            alpha=0.7,
            s=50,  # Marker size
        )
    plt.xlabel('date')
    plt.ylabel('flow (million gallons per day)')
    plt.title(title)
    plt.legend(loc='best', title="Flow Categories")
    plt.grid(False)
    plt.show()

#flow_class(data, title="different flow classifications")
if 'season' in data.columns:
    data = pd.get_dummies(data, columns=['season'], drop_first=True)

if 'rainfall' in data.columns:
    data['rainfall_lag1'] = data['rainfall'].shift(1)
    data['rainfall_lag2'] = data['rainfall'].shift(2)

us_holidays = holidays.US(years=range(data.index.year.min(), data.index.year.max() + 1))
data['holiday'] = data.index.to_series().apply(lambda x: 1 if x in us_holidays else 0)

if 'ADD' in data.columns:
    for col in [col for col in data.columns if col.startswith('season_')]:
        data[f'ADD_{col}'] = data['ADD'] * data[col]

def calculate_rolling_features(df):
    df = df.copy()
    for n in range(2, 7):
        df[f'flow_rolling_mean_{n}'] = df['flow'].shift(1).rolling(window=n).mean()
        df[f'flow_rolling_std_{n}'] = df['flow'].shift(1).rolling(window=n).std()
        df[f'flow_lag_{n}'] = df['flow'].shift(n)
    return df

label_encoder = LabelEncoder()
label_encoder.fit(data['flow_class'])

train_size = int(0.7 * len(data))
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]
train_data = calculate_rolling_features(train_data)
test_data = calculate_rolling_features(test_data)

train_data['flow_class_tplus1'] = train_data['flow_class'].shift(-1)
train_data['flow_tplus1'] = train_data['flow'].shift(-1)

test_data['flow_class_tplus1'] = test_data['flow_class'].shift(-1)
test_data['flow_tplus1'] = test_data['flow'].shift(-1)

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
y_train = label_encoder.transform(train_data['flow_class_tplus1'])
y_test = label_encoder.transform(test_data['flow_class_tplus1'])

X_train = train_data.drop(columns=['flow', 'flow_class', 'flow_class_tplus1', 'flow_tplus1'])
X_test = test_data.drop(columns=['flow', 'flow_class', 'flow_class_tplus1', 'flow_tplus1'])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_best = LogisticRegression(
    C=0.5,
    class_weight='balanced',
    penalty='l2'    ,solver='newton-cg',  # Solver for logistic regression
    max_iter=5000  # higher convergence is needed for "some" complex data
)

svm_best = SVC(
    C=10,
    gamma='scale',
    kernel='linear',
    class_weight='balanced',
    decision_function_shape='ovo',
    probability=True,
    random_state=42
)

rf_best = RandomForestClassifier(
    bootstrap=True,
    class_weight='balanced',
    criterion='gini',
    max_depth=10,
    max_features=None,
    min_samples_leaf=10,
    min_samples_split=10,
    n_estimators=100,
    ccp_alpha=0.01,
    max_samples=0.8,
    random_state=42
)
logistic_best.fit(X_train_scaled, y_train)
svm_best.fit(X_train_scaled, y_train)
rf_best.fit(X_train_scaled, y_train)

def calculate_metrics(models, model_names, X_train_scaled, y_train, X_test_scaled, y_test, label_encoder):
    metrics = []
    class_metrics = []

    for model, model_name in zip(models, model_names):
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        train_mcc = matthews_corrcoef(y_train, y_train_pred)

        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_mcc = matthews_corrcoef(y_test, y_test_pred)

        metrics.append({
            "Model": model_name,
            "Train Precision": train_precision,
            "Train Recall": train_recall,
            "Train F1 Score": train_f1,
            "Train MCC": train_mcc,
            "Test Precision": test_precision,
            "Test Recall": test_recall,
            "Test F1 Score": test_f1,
            "Test MCC": test_mcc
        })

        report_train = classification_report(
            y_train, y_train_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0
        )
        report_test = classification_report(
            y_test, y_test_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0
        )

        for cls in label_encoder.classes_:
            cls_idx = label_encoder.transform([cls])[0]
            y_train_binary = (y_train == cls_idx).astype(int)
            y_train_pred_binary = (y_train_pred == cls_idx).astype(int)
            mcc_train_class = matthews_corrcoef(y_train_binary, y_train_pred_binary)
            y_test_binary = (y_test == cls_idx).astype(int)
            y_test_pred_binary = (y_test_pred == cls_idx).astype(int)
            mcc_test_class = matthews_corrcoef(y_test_binary, y_test_pred_binary)
            class_metrics.append({
                "Model": model_name,
                "Class": cls,
                "Dataset": "Train",
                "Precision": report_train[cls]["precision"],
                "Recall": report_train[cls]["recall"],
                "F1-Score": report_train[cls]["f1-score"],
                "MCC": mcc_train_class,  # Add MCC for Train
                "Support": report_train[cls]["support"]
            })
            class_metrics.append({
                "Model": model_name,
                "Class": cls,
                "Dataset": "Test",
                "Precision": report_test[cls]["precision"],
                "Recall": report_test[cls]["recall"],
                "F1-Score": report_test[cls]["f1-score"],
                "MCC": mcc_test_class,
                "Support": report_test[cls]["support"]
            })

    class_metrics_df = pd.DataFrame(class_metrics)
    return class_metrics_df
def plot_confusion_matrices(models, model_names, X_train_scaled, y_train, X_test_scaled, y_test, label_encoder):
    class_labels = ["High", "Normal", "Very High"]
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 6 * n_models), constrained_layout=False)

    for i, (model, model_name) in enumerate(zip(models, model_names)):
        y_train_pred = model.predict(X_train_scaled)
        cm_train = confusion_matrix(y_train, y_train_pred)
        disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_labels)
        disp_train.plot(cmap=plt.cm.Blues, ax=axes[i, 0], colorbar=False)
        axes[i, 0].set_title(f"{model_name} - Training Performance", fontsize=14)
        if i != 0:
            axes[0, 0].set_ylabel("Observed", fontsize=12) and axes[i, 0].set_ylabel("Observed", fontsize=12)
        else:
            axes[i, 0].set_ylabel("")
        y_test_pred = model.predict(X_test_scaled)
        cm_test = confusion_matrix(y_test, y_test_pred)
        disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_labels)
        disp_test.plot(cmap=plt.cm.Blues, ax=axes[i, 1], colorbar=False)
        axes[i, 1].set_title(f"{model_name} - Testing Performance", fontsize=14)
        axes[i, 1].set_ylabel("")
        if i == n_models - 1:
            axes[i, 0].set_xlabel("Predicted", fontsize=12)
            axes[i, 1].set_xlabel("Predicted", fontsize=12)
        else:
            axes[i, 0].set_xlabel("")
            axes[i, 1].set_xlabel("")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()

def plot_class_performance_horizontal(class_metrics_df):
    class_order = ["normal", "high", "very high"]
    model_order = ["logistic regression", "support vector machine", "random forest"]
    class_metrics_df["Class"] = pd.Categorical(class_metrics_df["Class"], categories=class_order, ordered=True)
    class_metrics_df["Model"] = pd.Categorical(class_metrics_df["Model"], categories=model_order, ordered=True)
    class_metrics_df = class_metrics_df.sort_values(["Class", "Model"])
    classes = class_order
    models = model_order
    fig, axes = plt.subplots(len(classes), len(models), figsize=(20, 5 * len(classes)), sharey=True)

    def plot_bars(ax, data, x, bar_width, offsets, colors, labels):
        if not data.empty:
            for i, metric in enumerate(["Precision", "Recall", "F1-Score", "MCC"]):
                bar = ax.bar(x + offsets[i], data[metric], width=bar_width, color=colors[i], label=labels[i])
                for b in bar:
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.2f}",
                            ha='center', va='bottom', fontsize=10)

    for row, cls in enumerate(classes):
        for col, model in enumerate(models):
            data = class_metrics_df[(class_metrics_df["Class"] == cls) & (class_metrics_df["Model"] == model)]
            x = np.array([0])
            bar_width = 0.2
            train_offsets = [-bar_width * 1.5, -bar_width / 2, bar_width / 2, bar_width * 1.5]
            test_offsets = [bar_width * 3, bar_width * 4, bar_width * 5, bar_width * 6]
            train_data = data[data["Dataset"] == "Train"]
            test_data = data[data["Dataset"] == "Test"]
            plot_bars(axes[row, col], train_data, x, bar_width, train_offsets,
                      ["blue", "green", "orange", "cyan"], ["Train Precision", "Train Recall", "Train F1-Score", "Train MCC"])
            plot_bars(axes[row, col], test_data, x, bar_width, test_offsets,
                      ["red", "purple", "brown", "pink"], ["Test Precision", "Test Recall", "Test F1-Score", "Test MCC"])

            axes[row, col].set_xticks([])
            if row == len(classes) - 1:
                axes[row, col].set_xlabel(model, fontsize=12)
            if col == 0:
                axes[row, col].set_ylabel(cls, fontsize=12)
            if row == 1 and col == 0:
                axes[row, col].legend(loc="upper right", fontsize="small", frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
def plot_classification_report(class_metrics_df):
    class_order = ["normal", "high", "very high"]
    model_order = ["logistic regression", "support vector machine", "random forest"]

    fig, axes = plt.subplots(1, len(model_order), figsize=(18, 6))
    greens = sns.color_palette("Greens", as_cmap=True)
    colors = [(0.85, 1, 0.85), greens(0.5), greens(1.0)]  # Light green at 0 instead of white
    positions = [0, 0.5, 1]  # Transition points: 0 (light green), 0.5 (medium green), 1 (dark green)

    custom_greens = mcolors.LinearSegmentedColormap.from_list("custom_greens", list(zip(positions, colors)))
    cbar_ax = fig.add_axes([0.92, 0.4, 0.015, 0.3])  # Adjusted position and size for a shorter bar

    for i, model_name in enumerate(model_order):
        train_metrics = class_metrics_df[(class_metrics_df['Model'] == model_name) & (class_metrics_df['Dataset'] == 'Train')]
        test_metrics = class_metrics_df[(class_metrics_df['Model'] == model_name) & (class_metrics_df['Dataset'] == 'Test')]

        train_report_df = train_metrics[['Class', 'Precision', 'Recall', 'F1-Score', 'MCC']].set_index('Class')
        test_report_df = test_metrics[['Class', 'Precision', 'Recall', 'F1-Score', 'MCC']].set_index('Class')

        train_report_df = train_report_df.reindex(class_order)
        test_report_df = test_report_df.reindex(class_order)

        train_report_df.columns = ['precision', 'recall', 'F1-Score', 'MCC']
        test_report_df.columns = ['precision', 'recall', 'F1-Score', 'MCC']

        combined_report_df = pd.concat([train_report_df, test_report_df], axis=1)
        heatmap = sns.heatmap(combined_report_df, annot=True, cmap=custom_greens, fmt='.2f', ax=axes[i],
                              cbar=(i == len(model_order) - 1), cbar_ax=(cbar_ax if i == len(model_order) - 1 else None))

        axes[i].set_title(f"{model_name} performance", fontsize=14)
        axes[i].set_xlabel("Metrics", fontsize=10)
        axes[i].set_xticklabels(combined_report_df.columns, ha="right", fontsize=10)

        if i == 0:
            axes[i].set_ylabel("flow classes", fontsize=12)
        else:
            axes[i].set_ylabel("")
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0", "0.5", "1"])
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0,0, 0.9, 0.96])  # Adjust layout to fit the colorbar
    plt.show()


models = [logistic_best, svm_best, rf_best]
model_names = ["logistic regression", "support vector machine", "random forest"]

##ADASYN################

def temporal_adasyn(X, y, target_class, class_high, temporal_window, K):
    minority_count = np.sum(y == target_class)
    majority_count = np.sum(y == class_high)
    class_label_mapping = {0: "high", 1: "normal", 2: "very high"}
    sorted_classes = ["normal", "high", "very high"]
    unique_classes, class_counts = np.unique(y, return_counts=True)
    original_class_distribution = pd.Series(
        {class_label_mapping[cls]: count for cls, count in zip(unique_classes, class_counts)},
        name="count",
    )
    print("\nClass distribution in original training set:")
    print(original_class_distribution.reindex(sorted_classes))
    imbalance_ratio = minority_count / max(majority_count, 1)
    beta = max(0.2, min(2.0, 2.0 * (1 - imbalance_ratio)))
    G = int((majority_count - minority_count) * beta)
    max_synthetic_samples = min(int(majority_count * 0.5), minority_count * 2)
    G = min(G, max_synthetic_samples)

    print("\nDynamic Inspection:")
    print(f"  - Minority count (class 'very high'): {minority_count}")
    print(f"  - Majority count (class 'high'): {majority_count}")
    print(f"  - Ratio (Minority/Majority): {imbalance_ratio:.4f}")
    print(f"  - Dynamically selected beta: {beta}")
    print(f"  - Synthetic samples to generate: {G}")

    if minority_count == 0 or majority_count == 0 or G <= 0:
        print("No augmentation needed (minority or majority class is zero).")
        return X, y
    minority_indices = np.where(y == target_class)[0]
    nn = NearestNeighbors(n_neighbors=K + 1)  # K + 1 to include the point itself
    nn.fit(X)
    weights = np.zeros(len(minority_indices))
    for idx, target_idx in enumerate(minority_indices):
        neighbors = nn.kneighbors(X[target_idx].reshape(1, -1), return_distance=False).flatten()
        neighbors = neighbors[1:]  # Exclude the sample itself
        valid_neighbors = [
            neighbor_idx for neighbor_idx in neighbors
            if abs(neighbor_idx - target_idx) <= temporal_window
        ]
        if not valid_neighbors:
            continue
        neighbor_labels = y[valid_neighbors]
        majority_neighbor_count = np.sum(neighbor_labels == class_high)
        weights[idx] = majority_neighbor_count / len(valid_neighbors) if valid_neighbors else 0
    if weights.sum() > 0:
        weights /= weights.sum()
    synthetic_samples = []
    for idx, target_idx in enumerate(minority_indices):
        num_samples_to_generate = int(weights[idx] * G)
        if num_samples_to_generate == 0:
            continue

        neighbors = nn.kneighbors(X[target_idx].reshape(1, -1), return_distance=False).flatten()
        neighbors = neighbors[1:]  # Exclude the sample itself
        valid_neighbors = [
            neighbor_idx for neighbor_idx in neighbors
            if abs(neighbor_idx - target_idx) <= temporal_window
        ]
        if not valid_neighbors:
            continue
        for _ in range(num_samples_to_generate):
            neighbor_idx = np.random.choice(valid_neighbors)
            lambda_ = np.random.random()
            synthetic_sample = X[target_idx] + lambda_ * (X[neighbor_idx] - X[target_idx])
            synthetic_samples.append(synthetic_sample)

    if len(synthetic_samples) == 0:
        print("No synthetic samples generated. Returning original data.")
        return X, y

    synthetic_samples = np.array(synthetic_samples)
    synthetic_labels = np.full(len(synthetic_samples), target_class)

    print(f"  - Generated {len(synthetic_samples)} synthetic samples for class 'very high'.")
    augmented_data = np.vstack([X, synthetic_samples])
    augmented_labels = np.hstack([y, synthetic_labels])
    unique_classes_aug, class_counts_aug = np.unique(augmented_labels, return_counts=True)
    augmented_class_distribution = pd.Series(
        {class_label_mapping[cls]: count for cls, count in zip(unique_classes_aug, class_counts_aug)},
        name="count",
    )
    print("\nClass distribution in augmented training set:")
    print(augmented_class_distribution.reindex(sorted_classes))

    return augmented_data, augmented_labels
label_encoder = LabelEncoder()
train_data_copy = train_data.copy()
train_data_copy['flow_class_encoded'] = label_encoder.fit_transform(train_data_copy['flow_class_tplus1'])
X_original = train_data_copy.drop(columns=['flow_class','flow_tplus1', 'flow_class_tplus1', 'flow_class_encoded']).values
y_original = train_data_copy['flow_class_encoded'].values
X_augmented, y_augmented = temporal_adasyn(
    X=X_original,
    y=y_original,
    target_class=2,       # "very high" class
    class_high=0,         # "high" class (majority)
    temporal_window=10,   # Adjust temporal window as needed
    K=5                   # Number of nearest neighbors
)

test_data_copy = test_data.copy()
X_test = test_data_copy.drop(columns=['flow_class', 'flow_tplus1','flow_class_tplus1'], errors='ignore').values
y_test = label_encoder.transform(test_data_copy['flow_class_tplus1'])
scaler = StandardScaler()
X_train_augmented_scaled = scaler.fit_transform(X_augmented)
X_test_scaled = scaler.transform(X_test)

##smote#########
def ts_smote_with_temporal_order(data, target_column, k_neighbors, multiplier, temporal_window, max_samples=None):
    augmented_data = data.copy()

    target_class = 'very high'
    class_high = 'high'

    class_counts = augmented_data[target_column].value_counts()
    minority_count = class_counts.get(target_class, 0)
    majority_count = class_counts.get(class_high, 0)

    if minority_count == 0 or majority_count == 0:
        raise ValueError(f"Classes '{class_high}' and '{target_class}' must be present in the dataset.")
    minority_data = augmented_data[augmented_data[target_column] == target_class]
    majority_data = augmented_data[augmented_data[target_column] == class_high]
    minority_features = minority_data.drop(columns=[target_column, 'date'])
    n_synthetic_samples = int(len(majority_data) * multiplier - len(minority_data))
    if max_samples is not None:
        n_synthetic_samples = min(n_synthetic_samples, max_samples)
    if n_synthetic_samples <= 0:
        return augmented_data
    X = minority_features.values
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(minority_data)))
    nn.fit(X)
    neighbors = nn.kneighbors(X, return_distance=False)

    synthetic_samples = []
    for _ in range(n_synthetic_samples):
        i = np.random.choice(len(minority_data))
        valid_neighbors = [
            idx for idx in neighbors[i]
            if abs(idx - i) <= temporal_window
        ]
        if not valid_neighbors:
            continue
        neighbor_idx = np.random.choice(valid_neighbors)
        lambda_ = np.random.random()
        synthetic_sample = X[i] + lambda_ * (X[neighbor_idx] - X[i])
        synthetic_sample = np.clip(synthetic_sample, minority_features.min().values, minority_features.max().values)
        synthetic_row = dict(zip(minority_features.columns, synthetic_sample))
        synthetic_row[target_column] = target_class
        synthetic_row['date'] = pd.NaT  # Set synthetic date as NaT

        synthetic_samples.append(synthetic_row)
    synthetic_df = pd.DataFrame(synthetic_samples)
    augmented_data = pd.concat([augmented_data, synthetic_df], ignore_index=True)

    return augmented_data

train_data_copy = train_data.reset_index()
train_data_copy = train_data_copy.drop(columns=['flow_class'])
train_data_augmented = ts_smote_with_temporal_order(
    train_data_copy,
    target_column='flow_class_tplus1',
    k_neighbors=5,
    multiplier=1,
    temporal_window=10
)

train_data_augmented.set_index('date', inplace=True)
test_data_copy = test_data.copy()

label_encoder = LabelEncoder()
train_data_augmented['flow_class_encoded'] = label_encoder.fit_transform(train_data_augmented['flow_class_tplus1'])
test_data_copy['flow_class_encoded'] = label_encoder.transform(test_data_copy['flow_class_tplus1'])

X_train = train_data_augmented.drop(columns=['flow', 'flow_tplus1', 'flow_class_tplus1', 'flow_class_encoded']).copy()
y_train = train_data_augmented['flow_class_encoded'].copy()

X_test_smote = test_data_copy.drop(columns=['flow_class', 'flow_tplus1', 'flow_class_tplus1','flow_class_encoded'], errors='ignore').copy().values
y_test = test_data_copy['flow_class_encoded'].copy().values

train_columns = X_train.columns
test_columns = test_data_copy.drop(columns=['flow_class', 'flow_tplus1', 'flow_class_tplus1', 'flow_class_encoded'], errors='ignore').columns
X_test_smote_aligned = test_data_copy[train_columns].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_data_copy[train_columns])


print("Class distribution in original training set:")
print(train_data['flow_class_tplus1'].value_counts())
print("\nClass distribution in augmented training set:")
print(train_data_augmented['flow_class_tplus1'].value_counts())


train_data_augmented = train_data_augmented.drop(columns=['flow_class_encoded'], errors='ignore')
y_train_augmented = label_encoder.fit_transform(train_data_augmented['flow_class_tplus1'])
X_train_augmented = train_data_augmented.drop(columns=['flow','flow_tplus1','flow_class_tplus1'])
X_train_augmented_scaled = scaler.fit_transform(X_train_augmented)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

# Data Cleaning
# Fitur yang menggunakan imputasi median
median_cols = [
    "num_adults",
    "num_children",
    "days_before_arrival",
    "nightly_rate_idr",
    "special_requests"
]

# Fitur yang memerlukan indikator missing values
indicator_cols = [
    "num_children",
    "special_requests",
    "needs_parking"
]

# Fitur kategorikal yang menggunakan kategori Unknown
categorical_cols = [
    "meal_package",
    "room_category"
]

def fit_cleaning_params(X_train):
    # Menghitung nilai pengganti dari data training
    params = {
        "medians": X_train[median_cols].median(),
        "parking_mode": X_train["needs_parking"].mode().iloc[0]
    }
    return params

def clean_data(df, params):
    df = df.copy()

    # Mempertahankan informasi missing values asli
    for col in indicator_cols:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)

    # Imputasi fitur numerik
    df[median_cols] = df[median_cols].fillna(params["medians"])

    # Imputasi fitur biner
    df["needs_parking"] = df["needs_parking"].fillna(
        params["parking_mode"]
    )

    # Imputasi fitur kategorikal
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    return df


# Data Transformation
# Fitur numerik
numerical_cols = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "days_before_arrival",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
    "arrival_year",
    "arrival_month",
    "arrival_day"
]

# Fitur kategorikal
categorical_cols = [
    "meal_package",
    "room_category",
    "booking_channel"
]

# Fitur biner dan missing indicators
binary_cols = [
    "needs_parking",
    "is_returning_guest",
    "num_children_was_missing",
    "special_requests_was_missing",
    "needs_parking_was_missing"
]

def fit_transform_params(X_train_clean):
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ), categorical_cols),
            ("bin", "passthrough", binary_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    transformer.set_output(transform="pandas")
    transformer.fit(X_train_clean)

    return transformer

def transform_data(df, transformer):
    transformed = transformer.transform(df)
    return transformed


# Feature Selection
# Fitur temporal
temporal_cols = [
    "arrival_year",
    "arrival_month",
    "arrival_day"
]

def select_features(df, include_temporal=True):
    df = df.copy()

    # Mengeluarkan kolom yang bukan fitur prediktor
    df = df.drop(
        columns=["id", "is_canceled"],
        errors="ignore"
    )

    # Mengeluarkan fitur temporal (Pilihan)
    if not include_temporal:
        df = df.drop(columns=temporal_cols)

    return df


## Dimensinality Reduction
def fit_pca(X_train, n_components=0.95):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    return pca

def reduce_dimensions(df, pca):
    return pca.transform(df)
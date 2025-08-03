from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.tensorflow
import tempfile
import joblib


# Base path definition
BASE_PATH = Path(__file__).parent if '__file__' in globals() else Path().resolve()
MODEL_DIR = "mlruns"
OUTPUT_PATH = BASE_PATH / "solution.csv"

def load_datasets(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three core datasets: orders, marketing points, and prices.

    Parameters:
        data_dir (str): Name of the folder containing the CSV files. Defaults to "data".

    Returns:
        Tuple containing:
            - orders_df (pd.DataFrame): historical orders with columns ['product','client','date','quantity'].
            - points_df (pd.DataFrame): marketing points with columns ['product','client','date','points'].
            - price_df (pd.DataFrame): product prices with columns ['product','date','price'].
    """
    data_path = BASE_PATH / data_dir

    orders_df = pd.read_csv(data_path / "historical_orders.csv")
    points_df = pd.read_csv(data_path / "marketing_points.csv")
    price_df  = pd.read_csv(data_path / "price.csv")

    return orders_df, points_df, price_df

def plot_dual_axis(series_left: pd.Series, series_right: pd.Series,
                   left_label: str, right_label: str, title: str) -> None:
    """
    Plot two time series on a shared x-axis with separate y-axes.

    Parameters:
        series_left (pd.Series): First series (plotted on left y-axis).
        series_right (pd.Series): Second series (plotted on right y-axis).
        left_label (str): Label for the left y-axis.
        right_label (str): Label for the right y-axis.
        title (str): Chart title.

    Returns:
        None: Displays the plot inline.
    """
    fig, ax_left = plt.subplots(figsize=(14, 6))

    ax_left.plot(series_left.index, series_left, color='tab:blue', label=left_label)
    ax_left.set_ylabel(left_label, color='tab:blue')
    ax_left.tick_params(axis='y', labelcolor='tab:blue')

    ax_right = ax_left.twinx()
    ax_right.plot(series_right.index, series_right, color='tab:orange', linestyle='--', label=right_label)
    ax_right.set_ylabel(right_label, color='tab:orange')
    ax_right.tick_params(axis='y', labelcolor='tab:orange')

    plt.title(title)
    fig.tight_layout()
    plt.grid(True)
    plt.show()

def print_average_daily_points(points_df: pd.DataFrame) -> float:
    """
    Calculate the overall average daily marketing points.

    Parameters:
        points_df (pd.DataFrame): DataFrame with at least columns ['date','points'].

    Returns:
        float: The computed average daily points.
    """
    df = points_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    daily_totals = df.groupby(df['date'].dt.date)['points'].sum()
    total_days   = daily_totals.size
    overall_sum  = daily_totals.sum()
    average_daily = overall_sum / total_days

    return average_daily

def prepare_daily_series(
    orders_df: pd.DataFrame,
    points_df: pd.DataFrame,
    date_col: str = 'date',
    quantity_col: str = 'quantity',
    points_col: str = 'points'
) -> tuple[pd.Series, pd.Series]:
    """
    Aggregate and align daily sales and marketing points series.

    Parameters:
        orders_df (pd.DataFrame): DataFrame with order data, must contain date_col and quantity_col.
        points_df (pd.DataFrame): DataFrame with marketing points data, must contain date_col and points_col.
        date_col (str): Name of the date column in both DataFrames.
        quantity_col (str): Name of the sales quantity column in orders_df.
        points_col (str): Name of the marketing points column in points_df.

    Returns:
        daily_quantity (pd.Series): Daily summed quantity, reindexed to full date range.
        daily_points (pd.Series): Daily summed points, reindexed to full date range.
    """
    # Ensure datetime
    orders = orders_df.copy()
    points = points_df.copy()
    orders[date_col] = pd.to_datetime(orders[date_col])
    points[date_col] = pd.to_datetime(points[date_col])

    # Aggregate
    daily_quantity = (
        orders.groupby(date_col)[quantity_col]
        .sum()
    )
    daily_points = (
        points.groupby(date_col)[points_col]
        .sum()
    )

    # Full date range
    start = min(daily_quantity.index.min(), daily_points.index.min())
    end   = max(daily_quantity.index.max(), daily_points.index.max())
    full_range = pd.date_range(start=start, end=end, freq='D')

    daily_quantity = daily_quantity.reindex(full_range, fill_value=0)
    daily_points   = daily_points.reindex(full_range, fill_value=0)

    return daily_quantity, daily_points

def analyze_seasonality(
    daily_quantity: pd.Series,
    daily_points: pd.Series,
    window: int = 180,
    period: int = 365
) -> None:
    """
    Perform seasonal analysis with two side-by-side plots.

    Parameters:
        daily_quantity (pd.Series): Daily sales series indexed by date.
        daily_points (pd.Series): Daily marketing points series indexed by date.
        window (int): Rolling window size for smoothing (in days).
        period (int): Seasonal decomposition period (in days).

    Returns:
        None: Displays two plots side by side:
          1. Rolling mean of sales vs points before deseasonalization.
          2. Rolling mean after deseasonalization vs points.
    """
    # Compute rolling means
    rolling_qty = daily_quantity.rolling(window).mean()
    rolling_pts = daily_points.rolling(window).mean()

    # Seasonal decomposition
    decomposition = seasonal_decompose(daily_quantity, model='additive', period=period)
    deseasoned = daily_quantity - decomposition.seasonal
    rolling_deseasoned = deseasoned.rolling(window).mean()

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

    # Plot before deseasonalization
    ax1.plot(rolling_qty.index, rolling_qty, label='Sales (rolling mean)', color='tab:blue')
    ax1.set_ylabel('Sales (rolling mean)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax1_2 = ax1.twinx()
    ax1_2.plot(rolling_pts.index, rolling_pts, label='Marketing Points (rolling mean)', color='tab:orange', linestyle='--')
    ax1_2.set_ylabel('Marketing Points (rolling mean)', color='tab:orange')
    ax1_2.tick_params(axis='y', labelcolor='tab:orange')

    ax1.set_title('Before Deseasonalization')
    ax1.grid(True)

    # Plot after deseasonalization
    ax2.plot(rolling_deseasoned.index, rolling_deseasoned, label='Deseasonalized Sales (rolling mean)', color='tab:green')
    ax2.set_ylabel('Deseasonalized Sales (rolling mean)', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    ax2_2 = ax2.twinx()
    ax2_2.plot(rolling_pts.index, rolling_pts, label='Marketing Points (rolling mean)', color='tab:orange', linestyle='--')
    ax2_2.set_ylabel('Marketing Points (rolling mean)', color='tab:orange')
    ax2_2.tick_params(axis='y', labelcolor='tab:orange')

    ax2.set_title('After Deseasonalization')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_raw_price_by_product(price_df: pd.DataFrame):
    """
    Plot the raw price variation over time for each product.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame with at least 'date', 'product', and 'price' columns.
    """
    # Ensure datetime format
    price_df['date'] = pd.to_datetime(price_df['date'])

    # Pivot table: dates as index, products as columns
    pivot_df = price_df.pivot(index='date', columns='product', values='price')

    # Plot
    plt.figure(figsize=(6, 3))
    for product in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[product], label=f'Product {product}')
    
    plt.title('Raw Price Variation by Product Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(title='Product', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_marketing_sales_correlation(
    historical_orders: pd.DataFrame,
    marketing_points: pd.DataFrame,
    window: int,
    seasonality_period: int = 365
) -> pd.DataFrame:
    """
    Compute the correlation between deseasonalized sales and marketing points 
    for each product using rolling average time series.

    Parameters
    ----------
    historical_orders : pd.DataFrame
        DataFrame with columns ['date', 'client', 'product', 'quantity'].
    marketing_points : pd.DataFrame
        DataFrame with columns ['date', 'client', 'product', 'points'].
    window : int
        Rolling window size for smoothing (in days).
    seasonality_period : int, optional
        Period for seasonal decomposition (default is 365 for yearly seasonality).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'product' and 'correlation' columns.
    """
    # Ensure datetime format
    historical_orders['date'] = pd.to_datetime(historical_orders['date'])
    marketing_points['date'] = pd.to_datetime(marketing_points['date'])

    # Aggregate daily sales and points by product
    sales_daily = historical_orders.groupby(['date', 'product'])['quantity'].sum().unstack(fill_value=0)
    points_daily = marketing_points.groupby(['date', 'product'])['points'].sum().unstack(fill_value=0)

    # Reindex both to the full date range
    full_range = pd.date_range(
        start=min(sales_daily.index.min(), points_daily.index.min()),
        end=max(sales_daily.index.max(), points_daily.index.max()),
        freq='D'
    )

    sales_daily = sales_daily.reindex(full_range, fill_value=0)
    points_daily = points_daily.reindex(full_range, fill_value=0)

    # Deseasonalize sales for each product
    deseasonalized_sales = pd.DataFrame(index=sales_daily.index)
    for product in sales_daily.columns:
        series = sales_daily[product]
        if series.sum() == 0:
            deseasonalized_sales[product] = 0
        else:
            try:
                decomposition = seasonal_decompose(series, model='additive', period=seasonality_period, extrapolate_trend='freq')
                deseasonalized_sales[product] = series - decomposition.seasonal
            except:
                deseasonalized_sales[product] = series  # fallback if decomposition fails

    # Rolling averages
    sales_smoothed = deseasonalized_sales.rolling(window=window, min_periods=1).mean()
    points_smoothed = points_daily.rolling(window=window, min_periods=1).mean()

    # Compute correlation for each product
    correlations = []
    for product in sales_smoothed.columns:
        if product in points_smoothed.columns:
            corr = sales_smoothed[product].corr(points_smoothed[product])
            correlations.append({'product': product, 'correlation': corr})

    return pd.DataFrame(correlations).sort_values(by='correlation', ascending=False)

def plot_window_correlation_by_product(
    historical_orders: pd.DataFrame,
    marketing_points: pd.DataFrame,
    seasonality_period: int = 365,
    min_window: int = 1,
    max_window: int = 180,
    figsize=(5, 3)
):
    """
    Plots side-by-side the correlation between marketing points and deseasonalized sales
    over a range of moving average windows for each product.

    Parameters:
        historical_orders (pd.DataFrame): DataFrame with 'date', 'product', and 'quantity' columns.
        marketing_points (pd.DataFrame): DataFrame with 'date', 'product', and 'points' columns.
        seasonality_period (int): Period to use for seasonality decomposition.
        min_window (int): Minimum moving average window size.
        max_window (int): Maximum moving average window size.
        figsize (tuple): Individual subplot size (width, height) per product.
    """
    # Dictionary to store all correlations per product
    all_correlations = {}

    # Loop over window sizes
    for window in range(min_window, max_window + 1):
        correlations = compute_marketing_sales_correlation(
            historical_orders=historical_orders,
            marketing_points=marketing_points,
            window=window,
            seasonality_period=seasonality_period
        )
        
        for _, row in correlations.iterrows():
            product = row['product']
            corr = row['correlation']
            
            if product not in all_correlations:
                all_correlations[product] = {'window': [], 'correlation': []}
            
            all_correlations[product]['window'].append(window)
            all_correlations[product]['correlation'].append(corr)

    # Number of products
    products = list(all_correlations.keys())
    n_products = len(products)

    # Create subplots in a single row
    fig, axes = plt.subplots(1, n_products, figsize=(figsize[0]*n_products, figsize[1]), sharey=True)

    if n_products == 1:
        axes = [axes]  # Ensure axes is iterable

    # Plot each product's correlation trend
    for ax, product in zip(axes, products):
        data = all_correlations[product]
        ax.plot(data['window'], data['correlation'], marker='o')
        ax.set_title(f'Product {product}')
        ax.set_xlabel('Window (days)')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(True)

    axes[0].set_ylabel('Correlation')
    plt.tight_layout()
    plt.show()

def deseasonalize_spent(
    df: pd.DataFrame,
    product_col: str = "product",
    date_col: str = "date",
    value_col: str = "spend",
    period: int = 365,
    model: str = "additive"
) -> pd.DataFrame:
    """
    Dessazonaliza a coluna `value_col` (spend) de `df` para cada nível em `product_col`,
    usando seasonal_decompose de statsmodels com período `period`.
    Retorna um DataFrame com `value_col` substituído pelos valores dessazonalizados.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    deseasoned_list = []
    for prod, grp in df.groupby(product_col):
        ts = grp.set_index(date_col)[value_col].resample('D').sum()
        decomp = seasonal_decompose(ts, model=model, period=period, extrapolate_trend='freq')
        deseason = ts - decomp.seasonal

        deseasoned_df = (
            deseason
            .reset_index()
            .assign(**{product_col: prod, f"{value_col}_deseason": deseason.values})
        )
        deseasoned_list.append(deseasoned_df)

    deseasoned_all = pd.concat(deseasoned_list, ignore_index=True)

    df = (
        df
        .merge(
            deseasoned_all[[product_col, date_col, f"{value_col}_deseason"]],
            on=[product_col, date_col], how="left"
        )
        .assign(**{value_col: lambda d: d[f"{value_col}_deseason"]})
        .drop(columns=[f"{value_col}_deseason"])
    )
    return df

def build_group_map(
    orders: pd.DataFrame,
    points: pd.DataFrame,
    n_groups: int = 4,
    group_size: int | None = None
) -> pd.DataFrame:
    """
    Agrupa clientes por produto com base na eficiência (pontos gastos por R$ investido).

    Parâmetros
    ----------
    orders : pd.DataFrame
        Dados de pedidos, deve conter as colunas ['product', 'client', 'spend']
    points : pd.DataFrame
        Dados de pontos de marketing por cliente×produto
    n_groups : int
        Número de grupos a dividir os clientes
    group_size : int | None
        Tamanho dos grupos (opcional, calculado automaticamente se None)

    Retorno
    -------
    pd.DataFrame com colunas: ['product', 'client', 'group_id']
    """

    spend_agg = (
        orders
        .groupby(['product', 'client'], as_index=False)['spend']
        .sum()
        .rename(columns={'spend': 'total_spent'})
    )

    points_agg = (
        points
        .groupby(['product', 'client'], as_index=False)['points']
        .sum()
        .rename(columns={'points': 'total_points'})
    )

    report = (
        spend_agg
        .merge(points_agg, on=['product', 'client'], how='outer')
        .fillna({'total_spent': 0, 'total_points': 0})
    )

    report['points_per_spent'] = report.apply(
        lambda r: r.total_points / r.total_spent
        if r.total_spent > 0 else
        (float('inf') if r.total_points > 0 else 0),
        axis=1
    )

    # Tratar infs
    finite = report['points_per_spent'].replace([np.inf, -np.inf], np.nan)
    report['points_per_spent'] = report['points_per_spent'].fillna(finite.max())

    # Determinar tamanho do grupo
    if group_size is None:
        total_clients = report['client'].nunique()
        group_size = max(1, total_clients // n_groups)

    # Agrupamento por produto
    product_groups = {}
    for prod, grp in report.groupby('product'):
        grp = grp.sort_values('points_per_spent').reset_index(drop=True)
        clients = grp['client'].tolist()
        splits = [clients[i * group_size:(i + 1) * group_size] for i in range(n_groups)]
        product_groups[prod] = splits

    # Construir dataframe final
    rows = []
    for prod, splits in product_groups.items():
        for gid, clients in enumerate(splits, start=1):
            for client in clients:
                rows.append({'product': prod, 'client': client, 'group_id': f'G{gid}'})

    return pd.DataFrame(rows)

def aggregate_group_daily(
    orders: pd.DataFrame,
    points: pd.DataFrame,
    group_map: pd.DataFrame
) -> pd.DataFrame:
    """
    Join orders (with spend) and points to group_map, aggregate sum(spend) and sum(points)
    per product, group_id, date.
    """
    df_o = orders.merge(group_map, on=['product','client'], how='inner')
    df_p = points.merge(group_map, on=['product','client'], how='inner')
    df_o['date'] = pd.to_datetime(df_o['date'])
    df_p['date'] = pd.to_datetime(df_p['date'])

    sales = (
        df_o.groupby(['product','group_id','date'])['spend']
            .sum().reset_index()
            .rename(columns={'spend':'total_spend'})
    )
    pts = (
        df_p.groupby(['product','group_id','date'])['points']
            .sum().reset_index()
    )
    df_gp = (
        sales.merge(pts, on=['product','group_id','date'], how='outer')
             .fillna(0)
             .sort_values(['product','group_id','date'])
             .reset_index(drop=True)
    )
    return df_gp

def expand_to_full_calendar(
    df_gp: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> pd.DataFrame:
    """
    Ensure every (product,group_id) has a row for each date in [start..end].
    """
    dates = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})
    prods = df_gp[['product','group_id']].drop_duplicates()
    prods['key'] = 1
    dates['key'] = 1

    full = prods.merge(dates, on='key').drop(columns='key')
    return (
        full
        .merge(df_gp, on=['product','group_id','date'], how='left')
        .fillna(0)
        .sort_values(['product','group_id','date'])
        .reset_index(drop=True)
    )

def compute_base_features(
    df: pd.DataFrame,
    windows: list[int] = [180,90,60,34,28,21,14,7]
) -> pd.DataFrame:
    """
    Adds:
      - spend_lag1
      - points_ma_<w>, spend_ma_<w> (lagged) for each w in windows
    """
    out = []
    for (prod, grp), g in df.groupby(['product','group_id']):
        ts = g.sort_values('date').set_index('date').asfreq('D', fill_value=0)
        ts['spend_lag1'] = ts['total_spend'].shift(1).fillna(0)
        for w in windows:
            ts[f'points_ma_{w}'] = ts['points'].rolling(w, min_periods=1).mean()
            ts[f'spend_ma_{w}']  = ts['spend_lag1'].rolling(w, min_periods=1).mean()
        ts = ts.reset_index()
        ts['product'], ts['group_id'] = prod, grp
        out.append(ts)
    return pd.concat(out, ignore_index=True)

def compute_rolling_targets(
    df: pd.DataFrame,
    horizon: int = 14,
    window: int = 180
) -> pd.DataFrame:
    """
    For each (product,group_id), compute
      target_spend_ma{window}_tplus<h> = rolling(window) of future spend from t to t+h
    """
    out = []
    for (prod, grp), g in df.groupby(['product','group_id']):
        ts = g.sort_values('date').set_index('date').asfreq('D', fill_value=0)
        for h in range(horizon+1):
            ts[f'target_spend_ma{window}_tplus{h}'] = (
                ts['total_spend'].shift(-h)
                  .rolling(window, min_periods=window)
                  .mean()
            )
        ts = ts.reset_index()
        ts['product'], ts['group_id'] = prod, grp
        out.append(ts)
    return pd.concat(out, ignore_index=True)

def preprocess_price(price_df: pd.DataFrame, min_date: str, max_date: str) -> pd.DataFrame:
    # Remove duplicatas: mantém apenas o primeiro preço de cada produto por dia
    price_df = price_df.drop_duplicates(subset=["product", "date"]).copy()

    # Converte para datetime
    price_df["date"] = pd.to_datetime(price_df["date"])

    # Cria intervalo de datas completo
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

    # Expande para todas as datas e preenche com o último preço conhecido
    product_prices = []

    for product, group in price_df.groupby("product"):
        group = group.set_index("date").sort_index()
        group = group.reindex(all_dates).ffill()
        group["product"] = product
        group["date"] = group.index
        product_prices.append(group)

    full_price_df = pd.concat(product_prices).reset_index(drop=True)

    # Reordena as colunas
    return full_price_df[["product", "date", "price"]]

def build_mlp(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(16, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def plot_training_history(history, product, group_id, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(4, 2))
    ax[0].plot(history.history['loss'], label='Treino')
    ax[0].plot(history.history['val_loss'], label='Validação')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Época')
    ax[0].set_ylabel('MSE')
    ax[0].legend()

    ax[1].plot(history.history['mae'], label='Treino')
    ax[1].plot(history.history['val_mae'], label='Validação')
    ax[1].set_title('MAE')
    ax[1].set_xlabel('Época')
    ax[1].set_ylabel('MAE')
    ax[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def train_mlp_model(X_train, y_train, X_val, y_val, product, group_id):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    model = build_mlp(X_train.shape[1], y_train.shape[1])
    
    checkpoint_path = os.path.join(MODEL_DIR, f"mlp_{product}_{group_id}.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    ]

    with mlflow.start_run(run_name=f"{product}_{group_id}"):
        mlflow.log_params({
            "product": product,
            "group_id": group_id,
            "model": "MLP",
            "input_dim": X_train.shape[1],
            "output_dim": y_train.shape[1]
        })

        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=120,
            batch_size=64,
            callbacks=callbacks,
            verbose=0,
            shuffle=True
        )

        # Log métricas finais
        final_val_loss = history.history['val_loss'][-1]
        final_val_mae = history.history['val_mae'][-1]
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("final_val_mae", final_val_mae)

        # Avaliar R2 por saída
        y_pred = model.predict(X_val_scaled)
        y_val_inv = scaler_y.inverse_transform(y_val_scaled)
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        for i, col in enumerate(y_train.shape[1] if hasattr(y_train, 'shape') else range(len(y_train))):
            r2 = r2_score(y_val_inv[:, i], y_pred_inv[:, i])
            mlflow.log_metric(f"r2_output_{i}", r2)

        # Salvar e logar gráfico de histórico
        fig_path = os.path.join(tempfile.gettempdir(), f"history_{product}_{group_id}.png")
        plot_training_history(history, product, group_id, save_path=fig_path)
        mlflow.log_artifact(fig_path)

        # Logar modelo e scalers
        model.save(checkpoint_path)
        mlflow.log_artifact(checkpoint_path)

        scaler_x_path = os.path.join(tempfile.gettempdir(), f"scaler_x_{product}_{group_id}.pkl")
        scaler_y_path = os.path.join(tempfile.gettempdir(), f"scaler_y_{product}_{group_id}.pkl")
        joblib.dump(scaler_x, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        mlflow.log_artifact(scaler_x_path)
        mlflow.log_artifact(scaler_y_path)

    return model, scaler_x, scaler_y, checkpoint_path


def run_training_pipeline(df_feat, x_cols, y_cols, save_dir, build_model_fn = build_mlp):
    """
    Executa o treinamento para cada combinação de produto e grupo.
    """
    df_feat = df_feat.dropna().reset_index(drop=True)
    train_df = df_feat[df_feat['date'] < '2024-06-01']
    val_df = df_feat[(df_feat['date'] >= '2024-06-01') & (df_feat['date'] < '2025-01-01')]

    mlp_models = {}
    history_dict = {}

    for product in df_feat['product'].unique():
        for group_id in df_feat['group_id'].unique():
            train_grp = train_df[(train_df['product'] == product) & (train_df['group_id'] == group_id)]
            val_grp = val_df[(val_df['product'] == product) & (val_df['group_id'] == group_id)]

            if len(train_grp) < 30 or len(val_grp) < 10:
                print(f"   Dados insuficientes para {product} | {group_id}. Pulando...")
                continue

            X_train = train_grp[x_cols].values
            y_train = train_grp[y_cols].values
            X_val = val_grp[x_cols].values
            y_val = val_grp[y_cols].values

            model, scaler_x, scaler_y, model_path, history = train_mlp_model(
                X_train, y_train, X_val, y_val, product, group_id)

            mlp_models[(product, group_id)] = {
                'model': model,
                'scaler_x': scaler_x,
                'scaler_y': scaler_y,
                'model_path': model_path
            }

            history_dict[(product, group_id)] = history

    return mlp_models

def evaluate_model(product, group_id, models, test_df, x_cols, y_cols):
    """
    Avalia o modelo MLP para um dado (product, group_id).
    Retorna DataFrame com predições e verdadeiros valores, além de métricas MAE por horizonte.
    """
    test_grp = test_df[(test_df['product'] == product) & (test_df['group_id'] == group_id)].copy()

    if len(test_grp) == 0:
        print(f"[AVISO] Nenhum dado de teste para produto={product}, grupo={group_id}")
        return None

    if (product, group_id) not in models:
        print(f"[AVISO] Modelo não treinado para produto={product}, grupo={group_id}")
        return None

    model_info = models[(product, group_id)]
    model = model_info['model']
    scaler_x = model_info['scaler_x']
    scaler_y = model_info['scaler_y']

    # Previsão
    X_test_scaled = scaler_x.transform(test_grp[x_cols].values)
    y_true = test_grp[y_cols].values
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Resultados
    df_result = test_grp[['date', 'product', 'group_id']].copy()
    for i, col in enumerate(y_cols):
        df_result[f'{col}_true'] = y_true[:, i]
        df_result[f'{col}_pred'] = y_pred[:, i]

    # Métricas
    maes = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(len(y_cols))]
    mae_mean = np.mean(maes)

    print(f"[AVALIAÇÃO] Produto: {product} | Grupo: {group_id}")
    for i, mae in enumerate(maes):
        print(f"  MAE ({y_cols[i]}): {mae:.2f}")
    print(f"MAE médio ({len(y_cols)} horizontes): {mae_mean:.2f}")
    print("-" * 40)

    return df_result


def predict_mlp(product, group_id, X_test, models):
    """
    Gera predições para um modelo MLP normalizado.
    """
    model_info = models.get((product, group_id))
    if model_info is None:
        print(f"Modelo para produto {product}, grupo {group_id} não encontrado.")
        return None

    model = model_info['model']
    scaler_x = model_info['scaler_x']
    scaler_y = model_info['scaler_y']

    X_test_scaled = scaler_x.transform(X_test)
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return y_pred

def avaliar_todos_modelos(test_df, mlp_models, x_cols, y_cols):
    """
    Avalia todos os modelos treinados e retorna um DataFrame com R² por target (prefixado com 'r2_').
    """
    results = []

    for (product, group_id), model_info in mlp_models.items():
        test_grp = test_df[(test_df['product'] == product) & (test_df['group_id'] == group_id)]
        if len(test_grp) == 0:
            print(f"[INFO] Sem dados de teste para Produto: {product}, Grupo: {group_id}. Pulando...")
            continue

        X_test = test_grp[x_cols].values
        y_test = test_grp[y_cols].values

        y_pred = predict_mlp(product, group_id, X_test, mlp_models)
        if y_pred is None:
            continue

        # R² por target, com prefixo 'r2_'
        r2_scores_dict = {
            f"r2_{target_name}": r2_score(y_test[:, i], y_pred[:, i])
            for i, target_name in enumerate(y_cols)
        }

        r2_mean = np.mean(list(r2_scores_dict.values()))

        results.append({
            'product': product,
            'group_id': group_id,
            'r2_mean': r2_mean,
            **r2_scores_dict
        })

    results_df = pd.DataFrame(results).sort_values(['product', 'group_id'])
    return results_df

def get_history_for_pair(df_gp, product, group_id, lookback=180):
    """
    Retorna histórico dos últimos `lookback` dias com colunas
    necessárias para x_cols, principalmente 'points', 'spend_lag1' e medias móveis.
    """
    grp = df_gp[(df_gp['product']==product) & (df_gp['group_id']==group_id)]
    grp = grp.sort_values('date').set_index('date')
    last = grp.index.max()
    # As colunas devem conter as features usadas no modelo
    cols_needed = ['points', 'total_spend']  # points e total_spend básicos para gerar lag e médias
    # Pode ter outras colunas extras se quiser, mas vamos manter só essenciais
    return grp.loc[last - pd.Timedelta(days=lookback-1): last, cols_needed].copy(), last


WINDOWS = [180, 90, 60, 34, 28, 21, 14, 7]

def enrich_history_with_features(h):
    """
    Cria colunas:
    - spend_lag1 (lag1 de total_spend)
    - points_ma_*, spend_ma_* (médias móveis)
    """
    h = h.copy()
    h['spend_lag1'] = h['total_spend'].shift(1).fillna(0)

    for w in WINDOWS:
        h[f'points_ma_{w}'] = h['points'].rolling(w, min_periods=1).mean()
        h[f'spend_ma_{w}'] = h['spend_lag1'].rolling(w, min_periods=1).mean()
    return h


def compute_response_curve(history, model_info, x_cols, y_cols, B=5000, step=1):
    model, scaler_x, scaler_y = model_info['model'], model_info['scaler_x'], model_info['scaler_y']
    pts = np.arange(0, B+1, step)
    sp = np.zeros_like(pts, dtype=float)

    for i, x in enumerate(pts):
        h = history.copy()
        today = h.index.max() + pd.Timedelta(days=1)
        # Em 'points' insere o orçamento x e em 'total_spend' insere 0 para o dia novo
        h.loc[today] = {'points': x, 'total_spend': 0}
        h = enrich_history_with_features(h)
        feats = h.loc[today, x_cols].values.reshape(1, -1)
        yhat_s = model.predict(scaler_x.transform(feats), verbose=0)
        yhat = scaler_y.inverse_transform(yhat_s)[0]
        sp[i] = yhat.mean()
    return pts, sp


def predict_todays_spend(df_gp, alloc, x_cols, y_cols, mlp_models, lookback=180):
    next_s = {}
    for (p, g), x in alloc.items():
        history, last = get_history_for_pair(df_gp, p, g, lookback)
        today = last + pd.Timedelta(days=1)
        # No histórico para o dia novo, aloca 'points' e zera 'total_spend' para prever
        history.loc[today] = {'points': x, 'total_spend': 0}
        h = enrich_history_with_features(history)
        feats = h.loc[today, x_cols].values.reshape(1, -1)
        info = mlp_models[(p, g)]
        yhat_s = info['model'].predict(info['scaler_x'].transform(feats), verbose=0)
        yhat = info['scaler_y'].inverse_transform(yhat_s)[0]
        next_s[(p, g)] = yhat[0]
    return next_s

def recursive_allocation_forecast(
    df_gp, mlp_models, x_cols, y_cols,
    steps: int, B: int=5000, lookback: int=180,
    step: int = 100
):
    history = df_gp.copy()
    records = []

    for _ in range(steps):
        response_curves = {}
        for key, info in mlp_models.items():
            h, _ = get_history_for_pair(history, *key, lookback)
            pts, sp = compute_response_curve(h, info, x_cols, y_cols, B=B, step=step)
            response_curves[key] = (pts, sp)

        alloc = allocate_budget_dp(response_curves, B=B, step=step)

        next_spend = predict_todays_spend(history, alloc, x_cols, y_cols, mlp_models, lookback)
        new_date = history['date'].max() + pd.Timedelta(days=1)
        new_rows = []

        for (product, group_id), x in alloc.items():
            s_hat = next_spend[(product, group_id)]
            new_rows.append({
                'product': product,
                'group_id': group_id,
                'date': new_date,
                'points': x,           # orçamento aplicado (feature)
                'total_spend': s_hat   # gasto previsto (target)
            })
            records.append(new_rows[-1])

        history = pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True)

    return history, pd.DataFrame(records)

def allocate_budget_dp(response_curves, B=5000, step=1):
    """
    Aloca até B unidades de pontos entre os pares (product, group_id)
    para maximizar a soma da receita prevista com base nas curvas de resposta.

    response_curves: dict (chave=(product, group_id), valor=(points_array, spend_array))
    B: orçamento total de pontos
    step: granularidade de alocação (por exemplo, 100 => alocações em 0, 100, 200, ...)
    
    Retorna:
        dicionário { (product, group_id): points_allocated }
    """
    keys = list(response_curves.keys())
    n = len(keys)
    m = B // step  # número de unidades discretas

    # DP[i][b] = melhor valor possível usando os i primeiros pares e b unidades
    DP = np.zeros((n+1, m+1))
    choice = np.zeros((n+1, m+1), dtype=int)

    for i in range(1, n+1):
        pts, sp = response_curves[keys[i-1]]
        sp_interp = np.interp(np.arange(0, B+1, step), pts, sp)

        for b in range(m+1):
            best_val = DP[i-1][b]
            best_k = 0

            for k in range(0, b+1):
                val = DP[i-1][b-k] + sp_interp[k]
                if val > best_val:
                    best_val = val
                    best_k = k
            DP[i][b] = best_val
            choice[i][b] = best_k

    # Reconstruir solução
    alloc = {}
    b = m
    for i in range(n, 0, -1):
        k = choice[i][b]
        alloc[keys[i-1]] = k * step
        b -= k

    return alloc

import random

def simple_distribute_points(clients, P):
    """
    Distribui P pontos entre os clientes de forma round-robin:
      - Enquanto P >= 10:
          * Escolhe um cliente em sequência (round-robin)
          * Filtra chunks = [500,300,200,100,50,10] que sejam <= P
          * Sorteia aleatoriamente um desses chunks
          * Atribui ao cliente e subtrai de P
      - Se sobrar 1 <= P < 10, atribui todo esse remanescente ao primeiro cliente
    Retorna dict client->pontos.
    """
    alloc = {c: 0 for c in clients}
    chunks = [500, 300, 200, 100, 50, 10]
    idx = 0

    # Enquanto der para usar ao menos o menor chunk (10)
    while P >= 10:
        client = clients[idx % len(clients)]
        # Só consideramos pedaços que cabem no P restante
        possible = [c for c in chunks if c <= P]
        # Sorteamos um tamanho dentre os possíveis
        chosen = random.choice(possible)
        alloc[client] += chosen
        P -= chosen
        idx += 1

    # Remanescente < 10: joga tudo no primeiro cliente da lista
    if P > 0:
        alloc[clients[0]] += P

    return alloc

def expand_group_alloc_to_clients(alloc_schedule: pd.DataFrame, group_map: pd.DataFrame):
    """
    Recebe:
      - alloc_schedule: DataFrame com colunas ['date','product','group_id','points']
      - group_map:       DataFrame com colunas ['product','client','group_id']
    Retorna:
      DataFrame com ['date','product','client','points'] para cada client,
      usando simple_distribute_points para repartir os pontos do grupo.
      Linhas com points==0 são descartadas.
    """
    records = []
    # itera cada linha de alocação de grupo
    for _, row in alloc_schedule.iterrows():
        date      = row['date']
        product   = row['product']
        group_id  = row['group_id']
        P         = int(row['points'])
        # lista de clients do grupo
        clients = (
            group_map
            .loc[
                (group_map['product']==product) &
                (group_map['group_id']==group_id),
                'client'
            ]
            .tolist()
        )
        if not clients or P<=0:
            continue
        # distribui pelos clients
        client_allocs = simple_distribute_points(clients, P)
        # transforma em registros
        for client, pts in client_allocs.items():
            if pts > 0:
                records.append({
                    'date':    date,
                    'product': product,
                    'client':  client,
                    'points':  pts
                })

    return pd.DataFrame(records)
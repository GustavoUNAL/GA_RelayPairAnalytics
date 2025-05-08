import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_statistical_metrics(data):
    """
    Calcula métricas estadísticas detalladas para los datos de coordinación.
    
    Args:
        data (list): Lista de diccionarios con datos de pares de relés
        
    Returns:
        dict: Diccionario con métricas estadísticas
    """
    # Extraer datos relevantes
    delta_t_values = [pair.get('delta_t', 0) for pair in data]
    main_tds = [pair.get('main_relay', {}).get('TDS', 0) for pair in data]
    backup_tds = [pair.get('backup_relay', {}).get('TDS', 0) for pair in data]
    main_pickup = [pair.get('main_relay', {}).get('pick_up', 0) for pair in data]
    backup_pickup = [pair.get('backup_relay', {}).get('pick_up', 0) for pair in data]
    main_ishc = [pair.get('main_relay', {}).get('Ishc', 0) for pair in data]
    backup_ishc = [pair.get('backup_relay', {}).get('Ishc', 0) for pair in data]
    
    # Calcular métricas básicas
    metrics = {
        'delta_t': {
            'mean': np.mean(delta_t_values),
            'median': np.median(delta_t_values),
            'std': np.std(delta_t_values),
            'skew': stats.skew(delta_t_values),
            'kurtosis': stats.kurtosis(delta_t_values),
            'min': np.min(delta_t_values),
            'max': np.max(delta_t_values),
            'q1': np.percentile(delta_t_values, 25),
            'q3': np.percentile(delta_t_values, 75)
        },
        'tds_ratio': {
            'mean': np.mean([b/m if m > 0 else 0 for b, m in zip(backup_tds, main_tds)]),
            'std': np.std([b/m if m > 0 else 0 for b, m in zip(backup_tds, main_tds)]),
            'min': np.min([b/m if m > 0 else 0 for b, m in zip(backup_tds, main_tds)]),
            'max': np.max([b/m if m > 0 else 0 for b, m in zip(backup_tds, main_tds)])
        },
        'pickup_ratio': {
            'mean': np.mean([b/m if m > 0 else 0 for b, m in zip(backup_pickup, main_pickup)]),
            'std': np.std([b/m if m > 0 else 0 for b, m in zip(backup_pickup, main_pickup)]),
            'min': np.min([b/m if m > 0 else 0 for b, m in zip(backup_pickup, main_pickup)]),
            'max': np.max([b/m if m > 0 else 0 for b, m in zip(backup_pickup, main_pickup)])
        },
        'ishc_ratio': {
            'mean': np.mean([b/m if m > 0 else 0 for b, m in zip(backup_ishc, main_ishc)]),
            'std': np.std([b/m if m > 0 else 0 for b, m in zip(backup_ishc, main_ishc)]),
            'min': np.min([b/m if m > 0 else 0 for b, m in zip(backup_ishc, main_ishc)]),
            'max': np.max([b/m if m > 0 else 0 for b, m in zip(backup_ishc, main_ishc)])
        }
    }
    
    return metrics

def perform_correlation_analysis(data):
    """
    Realiza un análisis de correlación detallado entre las variables.
    
    Args:
        data (list): Lista de diccionarios con datos de pares de relés
        
    Returns:
        tuple: (DataFrame de correlaciones, lista de correlaciones significativas)
    """
    # Crear DataFrame para análisis
    df = pd.DataFrame({
        'delta_t': [pair.get('delta_t', 0) for pair in data],
        'main_tds': [pair.get('main_relay', {}).get('TDS', 0) for pair in data],
        'backup_tds': [pair.get('backup_relay', {}).get('TDS', 0) for pair in data],
        'main_pickup': [pair.get('main_relay', {}).get('pick_up', 0) for pair in data],
        'backup_pickup': [pair.get('backup_relay', {}).get('pick_up', 0) for pair in data],
        'main_ishc': [pair.get('main_relay', {}).get('Ishc', 0) for pair in data],
        'backup_ishc': [pair.get('backup_relay', {}).get('Ishc', 0) for pair in data],
        'main_time': [pair.get('main_relay', {}).get('Time_out', 0) for pair in data],
        'backup_time': [pair.get('backup_relay', {}).get('Time_out', 0) for pair in data]
    })
    
    # Calcular correlaciones
    corr_matrix = df.corr()
    
    # Identificar correlaciones significativas (|r| > 0.5)
    significant_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > 0.5:
                significant_correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i,j]
                })
    
    return corr_matrix, significant_correlations

def perform_regression_analysis(data):
    """
    Realiza análisis de regresión para predecir delta_t.
    
    Args:
        data (list): Lista de diccionarios con datos de pares de relés
        
    Returns:
        dict: Resultados del análisis de regresión
    """
    # Preparar datos
    X = pd.DataFrame({
        'tds_ratio': [b/m if m > 0 else 0 for b, m in zip(
            [pair.get('backup_relay', {}).get('TDS', 0) for pair in data],
            [pair.get('main_relay', {}).get('TDS', 0) for pair in data]
        )],
        'pickup_ratio': [b/m if m > 0 else 0 for b, m in zip(
            [pair.get('backup_relay', {}).get('pick_up', 0) for pair in data],
            [pair.get('main_relay', {}).get('pick_up', 0) for pair in data]
        )],
        'ishc_ratio': [b/m if m > 0 else 0 for b, m in zip(
            [pair.get('backup_relay', {}).get('Ishc', 0) for pair in data],
            [pair.get('main_relay', {}).get('Ishc', 0) for pair in data]
        )]
    })
    y = np.array([pair.get('delta_t', 0) for pair in data])
    
    # Estandarizar variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ajustar modelo
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Calcular métricas
    y_pred = model.predict(X_scaled)
    r2 = model.score(X_scaled, y)
    mse = np.mean((y - y_pred) ** 2)
    
    return {
        'coefficients': dict(zip(X.columns, model.coef_)),
        'intercept': model.intercept_,
        'r2': r2,
        'mse': mse,
        'feature_importance': dict(zip(X.columns, np.abs(model.coef_)))
    }

def generate_statistical_report(data):
    """
    Genera un reporte estadístico completo.
    
    Args:
        data (list): Lista de diccionarios con datos de pares de relés
        
    Returns:
        dict: Reporte estadístico completo
    """
    # Calcular métricas estadísticas
    metrics = calculate_statistical_metrics(data)
    
    # Realizar análisis de correlación
    corr_matrix, significant_correlations = perform_correlation_analysis(data)
    
    # Realizar análisis de regresión
    regression_results = perform_regression_analysis(data)
    
    # Generar reporte
    report = {
        'basic_metrics': metrics,
        'correlation_analysis': {
            'matrix': corr_matrix.to_dict(),
            'significant_correlations': significant_correlations
        },
        'regression_analysis': regression_results,
        'summary': {
            'total_pairs': len(data),
            'coordinated_pairs': len([p for p in data if p.get('delta_t', 0) >= 0]),
            'uncoordinated_pairs': len([p for p in data if p.get('delta_t', 0) < 0]),
            'coordination_rate': len([p for p in data if p.get('delta_t', 0) >= 0]) / len(data) * 100,
            'avg_tmt': np.mean([abs(p.get('mt', 0)) for p in data if p.get('delta_t', 0) < 0])
        }
    }
    
    return report

def create_statistical_visualizations(data):
    """
    Crea visualizaciones estadísticas avanzadas.
    
    Args:
        data (list): Lista de diccionarios con datos de pares de relés
        
    Returns:
        dict: Diccionario con figuras de Plotly
    """
    # Preparar datos
    df = pd.DataFrame({
        'delta_t': [pair.get('delta_t', 0) for pair in data],
        'tds_ratio': [b/m if m > 0 else 0 for b, m in zip(
            [pair.get('backup_relay', {}).get('TDS', 0) for pair in data],
            [pair.get('main_relay', {}).get('TDS', 0) for pair in data]
        )],
        'pickup_ratio': [b/m if m > 0 else 0 for b, m in zip(
            [pair.get('backup_relay', {}).get('pick_up', 0) for pair in data],
            [pair.get('main_relay', {}).get('pick_up', 0) for pair in data]
        )],
        'ishc_ratio': [b/m if m > 0 else 0 for b, m in zip(
            [pair.get('backup_relay', {}).get('Ishc', 0) for pair in data],
            [pair.get('main_relay', {}).get('Ishc', 0) for pair in data]
        )]
    })
    
    # Crear visualizaciones
    figures = {}
    
    # 1. Distribución de Δt con curva normal
    fig_delta_t = go.Figure()
    fig_delta_t.add_trace(go.Histogram(
        x=df['delta_t'],
        name="Δt",
        marker_color='#2c3e50',
        nbinsx=30
    ))
    
    # Añadir curva normal
    x = np.linspace(df['delta_t'].min(), df['delta_t'].max(), 100)
    mu = df['delta_t'].mean()
    sigma = df['delta_t'].std()
    y = stats.norm.pdf(x, mu, sigma) * len(df) * (df['delta_t'].max() - df['delta_t'].min()) / 30
    
    fig_delta_t.add_trace(go.Scatter(
        x=x, y=y,
        name="Distribución Normal",
        line=dict(color='red', width=2)
    ))
    
    fig_delta_t.update_layout(
        title="Distribución de Δt con Ajuste Normal",
        xaxis_title="Δt (s)",
        yaxis_title="Frecuencia",
        showlegend=True
    )
    figures['delta_t_distribution'] = fig_delta_t
    
    # 2. Matriz de dispersión
    fig_scatter = make_subplots(rows=2, cols=2,
                              subplot_titles=("TDS Ratio vs Δt", "Pickup Ratio vs Δt",
                                            "I_shc Ratio vs Δt", "TDS vs Pickup Ratio"))
    
    fig_scatter.add_trace(
        go.Scatter(x=df['tds_ratio'], y=df['delta_t'],
                  mode='markers',
                  marker=dict(color=['#27ae60' if dt >= 0 else '#e74c3c' for dt in df['delta_t']]),
                  name="TDS Ratio"),
        row=1, col=1
    )
    
    fig_scatter.add_trace(
        go.Scatter(x=df['pickup_ratio'], y=df['delta_t'],
                  mode='markers',
                  marker=dict(color=['#27ae60' if dt >= 0 else '#e74c3c' for dt in df['delta_t']]),
                  name="Pickup Ratio"),
        row=1, col=2
    )
    
    fig_scatter.add_trace(
        go.Scatter(x=df['ishc_ratio'], y=df['delta_t'],
                  mode='markers',
                  marker=dict(color=['#27ae60' if dt >= 0 else '#e74c3c' for dt in df['delta_t']]),
                  name="I_shc Ratio"),
        row=2, col=1
    )
    
    fig_scatter.add_trace(
        go.Scatter(x=df['tds_ratio'], y=df['pickup_ratio'],
                  mode='markers',
                  marker=dict(color=['#27ae60' if dt >= 0 else '#e74c3c' for dt in df['delta_t']]),
                  name="TDS vs Pickup"),
        row=2, col=2
    )
    
    fig_scatter.update_layout(height=800, showlegend=False)
    figures['scatter_matrix'] = fig_scatter
    
    # 3. Box plots de ratios
    fig_box = make_subplots(rows=1, cols=3,
                           subplot_titles=("TDS Ratio", "Pickup Ratio", "I_shc Ratio"))
    
    fig_box.add_trace(
        go.Box(y=df['tds_ratio'], name="TDS Ratio", marker_color='#3498db'),
        row=1, col=1
    )
    
    fig_box.add_trace(
        go.Box(y=df['pickup_ratio'], name="Pickup Ratio", marker_color='#e74c3c'),
        row=1, col=2
    )
    
    fig_box.add_trace(
        go.Box(y=df['ishc_ratio'], name="I_shc Ratio", marker_color='#2ecc71'),
        row=1, col=3
    )
    
    fig_box.update_layout(height=500, showlegend=False)
    figures['box_plots'] = fig_box
    
    return figures 
import json
import os
import math
import copy
import traceback
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
from plotly.subplots import make_subplots

# --- Constantes ---
CTI = 0.2  # Intervalo de tiempo de coordinación (segundos)
# !IMPORTANTE: Define aquí el escenario que quieres analizar!
TARGET_SCENARIO_ID = "scenario_1"
# !IMPORTANTE: Usa la ruta correcta a TU archivo JSON!
INPUT_FILE = "data/processed/independent_relay_pairs_scenario_base_optimized.json"

# --- Constantes para la Curva de Tiempo Inverso (EJEMPLO: IEC Standard Inverse) ---
# AJUSTA ESTOS VALORES SI USAS OTRA CURVA (Very Inverse, Extremely Inverse, IEEE MI, etc.)
CURVE_A = 0.14
CURVE_B = 0.02
MIN_CURRENT_MULTIPLIER = 1.05  # Para empezar a graficar la curva un poco por encima del pickup
MAX_CURRENT_MULTIPLIER = 50    # Factor máximo de corriente sobre pickup para graficar (ajustable)
MIN_TIME_PLOT = 0.01           # Tiempo mínimo para el eje Y del gráfico (log scale)
MAX_TIME_PLOT = 20             # Tiempo máximo para el eje Y del gráfico (log scale)

print(f"--- Iniciando Análisis y Visualización ---")
print(f"Archivo de entrada: {INPUT_FILE}")
print(f"Analizando SOLAMENTE para: '{TARGET_SCENARIO_ID}'")
print(f"Usando curva IEC SI (A={CURVE_A}, B={CURVE_B}) para graficar.")

# --- Función para calcular la curva de tiempo inverso ---
def calculate_inverse_time_curve(tds, pickup, i_range):
    """Calcula los tiempos de operación para un rango de corrientes."""
    times = []
    # Evitar división por cero o errores si pickup es muy bajo o cero
    if pickup <= 1e-6:
        # Devolver un array de NaNs o un valor alto si el pickup no es válido
        return np.full_like(i_range, np.nan)

    for i in i_range:
        multiple = i / pickup
        if multiple <= 1.0:  # No opera por debajo del pickup
            time = np.inf  # O un valor muy alto para escala log, o np.nan
        else:
            try:
                # Formula IEC / IEEE (simplificada)
                denominator = (multiple ** CURVE_B) - 1
                if denominator <= 1e-9:  # Evitar división por número muy cercano a cero
                    time = np.inf  # O un valor muy alto
                else:
                    time = tds * (CURVE_A / denominator)
                # Asegurarse que el tiempo no sea negativo (puede pasar por errores numéricos)
                if time < 0:
                    time = np.inf
            except (OverflowError, ValueError):
                time = np.inf  # Manejar errores matemáticos
        times.append(time)
    # Reemplazar infinitos con NaN o un valor máximo para graficar en escala log
    # np.nan es mejor porque plotly puede ignorarlos
    return np.nan_to_num(np.array(times), nan=np.nan, posinf=np.nan, neginf=np.nan)

# --- Fase 1: Análisis de Datos ---
coordinated_pairs = []
uncoordinated_pairs = []
tmt_total_scenario = 0.0
total_valid_pairs_scenario = 0
scenario_pairs_found = 0
skipped_pairs_count = 0
total_pairs_read = 0

try:
    print("Cargando datos...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"El archivo especificado no existe: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        relay_pairs_data = json.load(f)
    print("Datos cargados correctamente.")

    if not isinstance(relay_pairs_data, list):
        raise TypeError(f"Error: El archivo {INPUT_FILE} no contiene una lista JSON.")

    print(f"Procesando pares para '{TARGET_SCENARIO_ID}'...")
    for pair_entry in relay_pairs_data:
        total_pairs_read += 1
        if not isinstance(pair_entry, dict):
            continue

        current_scenario_id = pair_entry.get("scenario_id")
        if current_scenario_id != TARGET_SCENARIO_ID:
            continue

        scenario_pairs_found += 1

        main_relay_info = pair_entry.get('main_relay')
        backup_relay_info = pair_entry.get('backup_relay')

        # Validar que tenemos diccionarios y tiempos de operación
        if not isinstance(main_relay_info, dict) or not isinstance(backup_relay_info, dict):
            print(f"Advertencia ({TARGET_SCENARIO_ID}): Falta información de relé principal o de respaldo en par (Fault: {pair_entry.get('fault', 'N/A')}). Par omitido.")
            skipped_pairs_count += 1
            continue

        main_time = main_relay_info.get('Time_out')
        backup_time = backup_relay_info.get('Time_out')

        if not isinstance(main_time, (int, float)) or not isinstance(backup_time, (int, float)):
            print(f"Advertencia ({TARGET_SCENARIO_ID}): Tiempo(s) de operación no numéricos o faltantes en par (Main: {main_relay_info.get('relay', 'N/A')}, Backup: {backup_relay_info.get('relay', 'N/A')}, Fault: {pair_entry.get('fault', 'N/A')}). Main Time: {main_time}, Backup Time: {backup_time}. Par omitido.")
            skipped_pairs_count += 1
            continue

        # --- Realizar Cálculos ---
        delta_t = backup_time - main_time - CTI
        mt = (delta_t - abs(delta_t)) / 2  # Penalización solo si delta_t < 0

        # Crear una copia y añadir resultados
        pair_info = copy.deepcopy(pair_entry)
        pair_info['delta_t'] = delta_t
        pair_info['mt'] = mt

        # --- Clasificar ---
        if delta_t >= 0:
            coordinated_pairs.append(pair_info)
        else:
            uncoordinated_pairs.append(pair_info)

    print("Procesamiento de pares completado.")

    # --- Calcular Métricas Finales ---
    if scenario_pairs_found == 0:
        print(f"ADVERTENCIA: No se encontraron pares para '{TARGET_SCENARIO_ID}'.")
        total_valid_pairs_scenario = 0
        miscoordination_count_scenario = 0
        tmt_total_scenario = 0.0
    else:
        total_valid_pairs_scenario = len(coordinated_pairs) + len(uncoordinated_pairs)
        miscoordination_count_scenario = len(uncoordinated_pairs)
        # Sumar 'mt' solo de los pares descoordinados
        tmt_total_scenario = sum(pair.get("mt", 0) for pair in uncoordinated_pairs if isinstance(pair.get("mt"), (int, float)))

    # --- Imprimir Resultados del Análisis ---
    print(f"\n--- Resultados del Análisis para '{TARGET_SCENARIO_ID}' ---")
    print(f"Total de pares leídos: {total_pairs_read}")
    print(f"Pares encontrados para '{TARGET_SCENARIO_ID}': {scenario_pairs_found}")
    if skipped_pairs_count > 0:
        print(f"Pares omitidos dentro de '{TARGET_SCENARIO_ID}': {skipped_pairs_count}")
    print(f"Pares válidos analizados para '{TARGET_SCENARIO_ID}': {total_valid_pairs_scenario}")
    print(f"Coordinados (delta_t >= 0): {len(coordinated_pairs)}")
    print(f"Descoordinados (delta_t < 0): {miscoordination_count_scenario}")
    # TMT es la suma de los valores absolutos de mt negativos
    print(f"Suma Penalización Descoordinación (TMT = Sum |mt|): {tmt_total_scenario:.5f} s")

except FileNotFoundError as e:
    print(f"Error CRÍTICO: {e}")
    exit()
except (TypeError, json.JSONDecodeError) as e:
    print(f"Error CRÍTICO al leer o procesar JSON ({INPUT_FILE}): {e}")
    exit()
except Exception as e:
    print(f"Error inesperado durante análisis: {e}")
    traceback.print_exc()
    print("ADVERTENCIA: Intentando continuar con la visualización...")

# --- Fase 2: Preparación de Datos para Dash ---
print("\nPreparando datos para Dash...")

# Preparar datos para el análisis
all_pairs = coordinated_pairs + uncoordinated_pairs

# Opciones para dropdowns usando la estructura correcta
def create_dropdown_options(pair_list):
    options = []
    for idx, pair in enumerate(pair_list):
        main_relay_info = pair.get('main_relay', {})
        backup_relay_info = pair.get('backup_relay', {})
        label = (f"L:{pair.get('fault', 'N/A')}% - "
                 f"M:{main_relay_info.get('relay', 'N/A')} ({main_relay_info.get('line', 'N/A')}) / "
                 f"B:{backup_relay_info.get('relay', 'N/A')} ({backup_relay_info.get('line', 'N/A')})")
        options.append({"label": label, "value": idx})
    return options

coordinated_options = create_dropdown_options(coordinated_pairs)
uncoordinated_options = create_dropdown_options(uncoordinated_pairs)

# Resumen para tablas (usando claves correctas)
summary_columns_map = {
    "Falla (%)": "fault",
    "Línea Principal": "main_relay.line",
    "Relé Principal": "main_relay.relay",
    "TDS (Main)": "main_relay.TDS",
    "Pickup (Main)": "main_relay.pick_up",
    "I_shc (Main)": "main_relay.Ishc",
    "t_m (Main)": "main_relay.Time_out",
    "Línea Backup": "backup_relay.line",
    "Relé Backup": "backup_relay.relay",
    "TDS (Backup)": "backup_relay.TDS",
    "Pickup (Backup)": "backup_relay.pick_up",
    "I_shc (Backup)": "backup_relay.Ishc",
    "t_b (Backup)": "backup_relay.Time_out",
    "Δt": "delta_t",
    "MT": "mt"
}

def get_nested_value(d, key_path, default='N/A'):
    keys = key_path.split('.')
    val = d
    try:
        for key in keys:
            val = val[key]
        # Formatear números
        if isinstance(val, (int, float)):
            if 'TDS' in key_path or 'Pickup' in key_path:
                return f"{val:.5f}"
            elif 'I_shc' in key_path or 't_m' in key_path or 't_b' in key_path or 'Δt' in key_path or 'MT' in key_path:
                return f"{val:.3f}"
            else:
                return val  # Caso Falla (%)
        return val
    except (KeyError, TypeError):
        return default

def format_summary(pair_list, column_map):
    summary_data = []
    for pair in pair_list:
        row = {display_name: get_nested_value(pair, json_key)
               for display_name, json_key in column_map.items()}
        summary_data.append(row)
    return summary_data

coordinated_summary = format_summary(coordinated_pairs, summary_columns_map)
uncoordinated_summary = format_summary(uncoordinated_pairs, summary_columns_map)

print("Datos preparados.")

# --- Fase 3: Creación de la Aplicación Dash ---
print("Configurando aplicación Dash...")

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1(f"Análisis de Coordinación de Protecciones", style={'textAlign': 'center'}),
    html.H1(f" Escenario Base Automatizado", style={'textAlign': 'center'}),
    html.H3(f"TMT: {tmt_total_scenario:.3f}, Pares de Relés: {total_valid_pairs_scenario}", style={'textAlign': 'center', 'marginTop': '-10px', 'marginBottom': '20px'}),
    dcc.Tabs([
        dcc.Tab(label=f"Coordinados ({len(coordinated_pairs)})", children=[
            html.Div([
                dcc.Dropdown(
                    id='coordinated-dropdown',
                    options=coordinated_options,
                    value=coordinated_options[0]['value'] if coordinated_options else None,
                    placeholder="Selecciona un par coordinado...",
                    style={'width': '70%', 'margin': '20px auto'}
                ),
                dcc.Graph(id='coordinated-graph', style={'height': '600px', 'width': '90%', 'margin': '0 auto'}),
                html.H4("Detalles del Par Seleccionado", style={'textAlign': 'center', 'marginTop': '20px'}),
                dash_table.DataTable(
                    id='coordinated-pair-table',
                    columns=[{"name": "Parámetro", "id": "parameter"}, {"name": "Valor", "id": "value"}],
                    style_table={'width': '60%', 'margin': '20px auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal', 'height': 'auto'},
                    style_header={'fontWeight': 'bold'}
                ),
                html.H3("Resumen de Pares Coordinados", style={'textAlign': 'center', 'marginTop': '40px'}),
                dash_table.DataTable(
                    id='coordinated-summary-table',
                    columns=[{"name": i, "id": i} for i in summary_columns_map.keys()],
                    data=coordinated_summary,
                    style_table={'overflowX': 'auto', 'width': '95%', 'margin': '20px auto'},
                    style_cell={'textAlign': 'center', 'minWidth': '80px', 'maxWidth': '150px', 'padding': '5px', 'whiteSpace': 'normal'},
                    style_header={'fontWeight': 'bold', 'textAlign': 'center'},
                    page_size=10, sort_action="native", filter_action="native",
                    tooltip_data=[{column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()} for row in coordinated_summary] if coordinated_summary else None,
                    tooltip_duration=None
                )
            ])
        ]),
        dcc.Tab(label=f"Descoordinados ({len(uncoordinated_pairs)})", children=[
            html.Div([
                dcc.Dropdown(
                    id='uncoordinated-dropdown',
                    options=uncoordinated_options,
                    value=uncoordinated_options[0]['value'] if uncoordinated_options else None,
                    placeholder="Selecciona un par descoordinado...",
                    style={'width': '70%', 'margin': '20px auto'}
                ),
                dcc.Graph(id='uncoordinated-graph', style={'height': '600px', 'width': '90%', 'margin': '0 auto'}),
                html.H4("Detalles del Par Seleccionado", style={'textAlign': 'center', 'marginTop': '20px'}),
                dash_table.DataTable(
                    id='uncoordinated-pair-table',
                    columns=[{"name": "Parámetro", "id": "parameter"}, {"name": "Valor", "id": "value"}],
                    style_table={'width': '60%', 'margin': '20px auto'},
                    style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal', 'height': 'auto'},
                    style_header={'fontWeight': 'bold'}
                ),
                html.H3("Magnitud de Penalización |mt| (Descoordinados)", style={'textAlign': 'center', 'marginTop': '40px'}),
                dcc.Graph(id='mt-graph', style={'height': '400px', 'width': '90%', 'margin': '0 auto'}),
                html.H3("Resumen de Pares Descoordinados", style={'textAlign': 'center', 'marginTop': '40px'}),
                dash_table.DataTable(
                    id='uncoordinated-summary-table',
                    columns=[{"name": i, "id": i} for i in summary_columns_map.keys()],
                    data=uncoordinated_summary,
                    style_table={'overflowX': 'auto', 'width': '95%', 'margin': '20px auto'},
                    style_cell={'textAlign': 'center', 'minWidth': '80px', 'maxWidth': '150px', 'padding': '5px', 'whiteSpace': 'normal'},
                    style_header={'fontWeight': 'bold', 'textAlign': 'center'},
                    page_size=10, sort_action="native", filter_action="native",
                    tooltip_data=[{column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()} for row in uncoordinated_summary] if uncoordinated_summary else None,
                    tooltip_duration=None
                )
            ]),
        ]),
        dcc.Tab(label="Analítica de Datos", children=[
            html.Div([
                # Primera fila: Métricas generales y estadísticas
                html.Div([
                    html.Div([
                        html.H4("Métricas y Estadísticas", style={'textAlign': 'center'}),
                        html.Div([
                            html.Div([
                                html.H5("Total de Pares"),
                                html.H3(f"{total_valid_pairs_scenario}", style={'color': '#2c3e50'}),
                                html.P(f"Coordinados: {len(coordinated_pairs)} ({len(coordinated_pairs)/total_valid_pairs_scenario*100:.1f}%)", style={'color': '#27ae60'}),
                                html.P(f"Descoordinados: {len(uncoordinated_pairs)} ({len(uncoordinated_pairs)/total_valid_pairs_scenario*100:.1f}%)", style={'color': '#e74c3c'})
                            ], className="metric-box"),
                            html.Div([
                                html.H5("TMT Total"),
                                html.H3(f"{tmt_total_scenario:.3f}s", style={'color': '#e67e22'}),
                                html.P(f"Promedio por par: {tmt_total_scenario/len(uncoordinated_pairs):.3f}s", style={'color': '#e67e22'}),
                                html.P(f"Desviación estándar: {np.std([abs(p.get('mt', 0)) for p in uncoordinated_pairs]):.3f}s", style={'color': '#e67e22'})
                            ], className="metric-box"),
                            html.Div([
                                html.H5("Estadísticas Δt"),
                                html.P(f"Media: {np.mean([p.get('delta_t', 0) for p in all_pairs]):.3f}s"),
                                html.P(f"Mediana: {np.median([p.get('delta_t', 0) for p in all_pairs]):.3f}s"),
                                html.P(f"Desv. Est.: {np.std([p.get('delta_t', 0) for p in all_pairs]):.3f}s")
                            ], className="metric-box"),
                            html.Div([
                                html.H5("Corrientes"),
                                html.P(f"Media I_shc Main: {np.mean([p.get('main_relay', {}).get('Ishc', 0) for p in all_pairs]):.3f}A"),
                                html.P(f"Media I_shc Backup: {np.mean([p.get('backup_relay', {}).get('Ishc', 0) for p in all_pairs]):.3f}A"),
                                html.P(f"Ratio Medio: {np.mean([p.get('backup_relay', {}).get('Ishc', 0)/p.get('main_relay', {}).get('Ishc', 1) for p in all_pairs]):.3f}")
                            ], className="metric-box")
                        ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px 0'})
                    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
                ]),

                # Segunda fila: Análisis de Coordinación
                html.Div([
                    html.Div([
                        html.H4("Análisis de Coordinación", style={'textAlign': 'center'}),
                        dcc.Graph(id='coordination-analysis')
                    ], style={'width': '100%'})
                ]),

                # Tercera fila: Distribuciones y Correlaciones
                html.Div([
                    html.Div([
                        html.H4("Distribuciones y Correlaciones", style={'textAlign': 'center'}),
                        dcc.Graph(id='correlation-matrix')
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([
                        html.H4("Análisis de Sensibilidad", style={'textAlign': 'center'}),
                        dcc.Graph(id='sensitivity-analysis')
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),

                # Cuarta fila: Análisis por Línea y Relé
                html.Div([
                    html.Div([
                        html.H4("Análisis por Línea", style={'textAlign': 'center'}),
                        dcc.Graph(id='line-analysis')
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    html.Div([
                        html.H4("Análisis por Relé", style={'textAlign': 'center'}),
                        dcc.Graph(id='relay-analysis')
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),

                # Quinta fila: Análisis de Fallas
                html.Div([
                    html.Div([
                        html.H4("Análisis de Fallas", style={'textAlign': 'center'}),
                        dcc.Graph(id='fault-analysis')
                    ], style={'width': '100%'})
                ]),

                # Sexta fila: Conclusiones y Recomendaciones
                html.Div([
                    html.H4("Conclusiones y Recomendaciones", style={'textAlign': 'center'}),
                    html.Div(id='conclusions-content', style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
                ]),

                # Séptima fila: Tabla de resumen detallado
                html.Div([
                    html.H4("Resumen Detallado de Pares", style={'textAlign': 'center'}),
                    dash_table.DataTable(
                        id='detailed-summary-table',
                        columns=[
                            {"name": "Falla (%)", "id": "fault"},
                            {"name": "Línea Principal", "id": "main_line"},
                            {"name": "Relé Principal", "id": "main_relay"},
                            {"name": "TDS (Main)", "id": "main_tds"},
                            {"name": "Pickup (Main)", "id": "main_pickup"},
                            {"name": "I_shc (Main)", "id": "main_ishc"},
                            {"name": "t_m (Main)", "id": "main_time"},
                            {"name": "Línea Backup", "id": "backup_line"},
                            {"name": "Relé Backup", "id": "backup_relay"},
                            {"name": "TDS (Backup)", "id": "backup_tds"},
                            {"name": "Pickup (Backup)", "id": "backup_pickup"},
                            {"name": "I_shc (Backup)", "id": "backup_ishc"},
                            {"name": "t_b (Backup)", "id": "backup_time"},
                            {"name": "Δt", "id": "delta_t"},
                            {"name": "MT", "id": "mt"},
                            {"name": "Estado", "id": "status"}
                        ],
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'center',
                            'minWidth': '80px',
                            'maxWidth': '150px',
                            'padding': '5px',
                            'whiteSpace': 'normal'
                        },
                        style_header={
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                        page_size=10,
                        sort_action="native",
                        filter_action="native"
                    )
                ])
            ]),
            html.Div([
                html.H4("Distribución de TDS y Pickup por Relé", style={'textAlign': 'center'}),
                dcc.Graph(id='relay-settings-graph')
            ])
        ])
    ])
])

# --- Fase 4: Callback para Actualizar Contenido ---
print("Definiendo callbacks...")

@app.callback(
    [Output('coordinated-graph', 'figure'),
     Output('coordinated-pair-table', 'data'),
     Output('uncoordinated-graph', 'figure'),
     Output('uncoordinated-pair-table', 'data'),
     Output('mt-graph', 'figure'),
     Output('coordination-analysis', 'figure'),
     Output('correlation-matrix', 'figure'),
     Output('sensitivity-analysis', 'figure'),
     Output('line-analysis', 'figure'),
     Output('relay-analysis', 'figure'),
     Output('fault-analysis', 'figure'),
     Output('conclusions-content', 'children'),
     Output('detailed-summary-table', 'data'),
     Output('relay-settings-graph', 'figure')],
    [Input('coordinated-dropdown', 'value'),
     Input('uncoordinated-dropdown', 'value')]
)
def update_analytics(coordinated_idx, uncoordinated_idx):
    # Valores por defecto
    default_fig_layout = {
        'title': {'text': "Selecciona un par para ver la gráfica", 'x': 0.5},
        'xaxis': {'visible': False}, 'yaxis': {'visible': False},
        'plot_bgcolor': '#f9f9f9', 'paper_bgcolor': '#f9f9f9'
    }
    
    # Inicializar todas las figuras con valores por defecto
    coordinated_fig = go.Figure(layout=default_fig_layout)
    coordinated_table_data = [{"parameter": "Info", "value": "Selecciona un par coordinado"}]
    uncoordinated_fig = go.Figure(layout=default_fig_layout)
    uncoordinated_table_data = [{"parameter": "Info", "value": "Selecciona un par descoordinado"}]
    mt_fig = go.Figure(layout={'title': {'text': "Gráfico |mt| (solo para descoordinados)", 'x': 0.5}})
    coordination_fig = go.Figure(layout=default_fig_layout)
    correlation_fig = go.Figure(layout=default_fig_layout)
    sensitivity_fig = go.Figure(layout=default_fig_layout)
    line_fig = go.Figure(layout=default_fig_layout)
    relay_fig = go.Figure(layout=default_fig_layout)
    fault_fig = go.Figure(layout=default_fig_layout)
    relay_settings_fig = go.Figure(layout=default_fig_layout)
    conclusions = html.Div("Selecciona un par para ver las conclusiones")
    detailed_data = []

    # Verificar si hay datos para procesar
    if not all_pairs:
        return (coordinated_fig, coordinated_table_data, uncoordinated_fig, 
                uncoordinated_table_data, mt_fig, coordination_fig, correlation_fig,
                sensitivity_fig, line_fig, relay_fig, fault_fig, conclusions, detailed_data,
                relay_settings_fig)

    try:
        # Preparar datos para el análisis
        df = pd.DataFrame({
            'delta_t': [pair.get('delta_t', 0) for pair in all_pairs],
            'main_tds': [pair.get('main_relay', {}).get('TDS', 0) for pair in all_pairs],
            'backup_tds': [pair.get('backup_relay', {}).get('TDS', 0) for pair in all_pairs],
            'main_pickup': [pair.get('main_relay', {}).get('pick_up', 0) for pair in all_pairs],
            'backup_pickup': [pair.get('backup_relay', {}).get('pick_up', 0) for pair in all_pairs],
            'main_ishc': [pair.get('main_relay', {}).get('Ishc', 0) for pair in all_pairs],
            'backup_ishc': [pair.get('backup_relay', {}).get('Ishc', 0) for pair in all_pairs],
            'main_time': [pair.get('main_relay', {}).get('Time_out', 0) for pair in all_pairs],
            'backup_time': [pair.get('backup_relay', {}).get('Time_out', 0) for pair in all_pairs]
        })

        # 1. Análisis de Coordinación
        coordination_fig = make_subplots(rows=2, cols=2,
                                       subplot_titles=("Distribución de Δt", "Relación Δt vs I_shc",
                                                      "Distribución de Tiempos", "Relación TDS vs Pickup"))
        
        # Distribución de Δt
        coordination_fig.add_trace(
            go.Histogram(x=df['delta_t'], name="Δt", marker_color='#2c3e50'),
            row=1, col=1
        )
        
        # Relación Δt vs I_shc
        coordination_fig.add_trace(
            go.Scatter(x=df['main_ishc'], y=df['delta_t'], mode='markers',
                      marker=dict(color=['#27ae60' if dt >= 0 else '#e74c3c' for dt in df['delta_t']]),
                      name="Δt vs I_shc"),
            row=1, col=2
        )
        
        # Distribución de Tiempos
        coordination_fig.add_trace(
            go.Box(y=df['main_time'], name="t_m", marker_color='#3498db'),
            row=2, col=1
        )
        coordination_fig.add_trace(
            go.Box(y=df['backup_time'], name="t_b", marker_color='#e74c3c'),
            row=2, col=1
        )
        
        # Relación TDS vs Pickup
        coordination_fig.add_trace(
            go.Scatter(x=df['main_tds'], y=df['main_pickup'], mode='markers',
                      marker=dict(color=['#27ae60' if dt >= 0 else '#e74c3c' for dt in df['delta_t']]),
                      name="TDS vs Pickup"),
            row=2, col=2
        )
        
        coordination_fig.update_layout(height=800, showlegend=False)

        # 2. Matriz de Correlación
        corr_matrix = df.corr()
        correlation_fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        correlation_fig.update_layout(
            title="Matriz de Correlación",
            height=600
        )

        # 3. Análisis de Sensibilidad
        sensitivity_fig = make_subplots(rows=1, cols=2,
                                      subplot_titles=("Sensibilidad a TDS", "Sensibilidad a Pickup"))
        
        # Sensibilidad a TDS
        tds_ratio = df['backup_tds'] / df['main_tds'].replace(0, np.nan)
        sensitivity_fig.add_trace(
            go.Scatter(x=tds_ratio, y=df['delta_t'],
                      mode='markers',
                      marker=dict(color=['#27ae60' if dt >= 0 else '#e74c3c' for dt in df['delta_t']]),
                      name="TDS Ratio"),
            row=1, col=1
        )
        
        # Sensibilidad a Pickup
        pickup_ratio = df['backup_pickup'] / df['main_pickup'].replace(0, np.nan)
        sensitivity_fig.add_trace(
            go.Scatter(x=pickup_ratio, y=df['delta_t'],
                      mode='markers',
                      marker=dict(color=['#27ae60' if dt >= 0 else '#e74c3c' for dt in df['delta_t']]),
                      name="Pickup Ratio"),
            row=1, col=2
        )
        
        sensitivity_fig.update_layout(height=500, showlegend=False)

        # 4. Análisis por Línea
        line_data = {}
        for pair in all_pairs:
            main_line = pair.get('main_relay', {}).get('line', 'N/A')
            backup_line = pair.get('backup_relay', {}).get('line', 'N/A')
            status = "Coordinado" if pair.get('delta_t', 0) >= 0 else "Descoordinado"
            
            if main_line not in line_data:
                line_data[main_line] = {'coordinated': 0, 'uncoordinated': 0, 'tmt': 0}
            if backup_line not in line_data:
                line_data[backup_line] = {'coordinated': 0, 'uncoordinated': 0, 'tmt': 0}
                
            if status == "Coordinado":
                line_data[main_line]['coordinated'] += 1
                line_data[backup_line]['coordinated'] += 1
            else:
                line_data[main_line]['uncoordinated'] += 1
                line_data[backup_line]['uncoordinated'] += 1
                line_data[main_line]['tmt'] += abs(pair.get('mt', 0))
                line_data[backup_line]['tmt'] += abs(pair.get('mt', 0))

        lines = list(line_data.keys())
        coordinated = [line_data[line]['coordinated'] for line in lines]
        uncoordinated = [line_data[line]['uncoordinated'] for line in lines]
        tmt = [line_data[line]['tmt'] for line in lines]

        line_fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Coordinación por Línea", "TMT por Línea"))
        
        line_fig.add_trace(
            go.Bar(x=lines, y=coordinated, name='Coordinados', marker_color='#27ae60'),
            row=1, col=1
        )
        line_fig.add_trace(
            go.Bar(x=lines, y=uncoordinated, name='Descoordinados', marker_color='#e74c3c'),
            row=1, col=1
        )
        
        line_fig.add_trace(
            go.Bar(x=lines, y=tmt, name='TMT', marker_color='#e67e22'),
            row=1, col=2
        )
        
        line_fig.update_layout(height=500, barmode='stack', showlegend=True)

        # 5. Análisis por Relé
        relay_data = {}
        for pair in all_pairs:
            main_relay = pair.get('main_relay', {}).get('relay', 'N/A')
            backup_relay = pair.get('backup_relay', {}).get('relay', 'N/A')
            status = "Coordinado" if pair.get('delta_t', 0) >= 0 else "Descoordinado"
            
            if main_relay not in relay_data:
                relay_data[main_relay] = {'coordinated': 0, 'uncoordinated': 0, 'tmt': 0}
            if backup_relay not in relay_data:
                relay_data[backup_relay] = {'coordinated': 0, 'uncoordinated': 0, 'tmt': 0}
                
            if status == "Coordinado":
                relay_data[main_relay]['coordinated'] += 1
                relay_data[backup_relay]['coordinated'] += 1
            else:
                relay_data[main_relay]['uncoordinated'] += 1
                relay_data[backup_relay]['uncoordinated'] += 1
                relay_data[main_relay]['tmt'] += abs(pair.get('mt', 0))
                relay_data[backup_relay]['tmt'] += abs(pair.get('mt', 0))

        relays = list(relay_data.keys())
        relay_coordinated = [relay_data[relay]['coordinated'] for relay in relays]
        relay_uncoordinated = [relay_data[relay]['uncoordinated'] for relay in relays]
        relay_tmt = [relay_data[relay]['tmt'] for relay in relays]

        relay_fig = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Coordinación por Relé", "TMT por Relé"))
        
        relay_fig.add_trace(
            go.Bar(x=relays, y=relay_coordinated, name='Coordinados', marker_color='#27ae60'),
            row=1, col=1
        )
        relay_fig.add_trace(
            go.Bar(x=relays, y=relay_uncoordinated, name='Descoordinados', marker_color='#e74c3c'),
            row=1, col=1
        )
        
        relay_fig.add_trace(
            go.Bar(x=relays, y=relay_tmt, name='TMT', marker_color='#e67e22'),
            row=1, col=2
        )
        
        relay_fig.update_layout(height=500, barmode='stack', showlegend=True)

        # 6. Análisis de Fallas
        fault_data = {}
        for pair in all_pairs:
            fault = pair.get('fault', 'N/A')
            status = "Coordinado" if pair.get('delta_t', 0) >= 0 else "Descoordinado"
            
            if fault not in fault_data:
                fault_data[fault] = {'coordinated': 0, 'uncoordinated': 0, 'tmt': 0}
                
            if status == "Coordinado":
                fault_data[fault]['coordinated'] += 1
            else:
                fault_data[fault]['uncoordinated'] += 1
                fault_data[fault]['tmt'] += abs(pair.get('mt', 0))

        faults = list(fault_data.keys())
        fault_coordinated = [fault_data[fault]['coordinated'] for fault in faults]
        fault_uncoordinated = [fault_data[fault]['uncoordinated'] for fault in faults]
        fault_tmt = [fault_data[fault]['tmt'] for fault in faults]

        fault_fig = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Coordinación por Falla", "TMT por Falla"))
        
        fault_fig.add_trace(
            go.Bar(x=faults, y=fault_coordinated, name='Coordinados', marker_color='#27ae60'),
            row=1, col=1
        )
        fault_fig.add_trace(
            go.Bar(x=faults, y=fault_uncoordinated, name='Descoordinados', marker_color='#e74c3c'),
            row=1, col=1
        )
        
        fault_fig.add_trace(
            go.Bar(x=faults, y=fault_tmt, name='TMT', marker_color='#e67e22'),
            row=1, col=2
        )
        
        fault_fig.update_layout(height=500, barmode='stack', showlegend=True)

        # 7. Distribución de TDS y Pickup por Relé
        relay_settings_fig = make_subplots(rows=2, cols=1,
                                         subplot_titles=("Distribución de TDS por Relé", "Distribución de Pickup por Relé"),
                                         vertical_spacing=0.15)

        # Preparar datos para TDS y Pickup
        relay_tds_data = {}
        relay_pickup_data = {}

        for pair in all_pairs:
            main_relay = pair.get('main_relay', {})
            backup_relay = pair.get('backup_relay', {})
            
            # TDS
            main_relay_name = main_relay.get('relay', 'N/A')
            backup_relay_name = backup_relay.get('relay', 'N/A')
            main_tds = main_relay.get('TDS', 0)
            backup_tds = backup_relay.get('TDS', 0)
            
            if main_relay_name not in relay_tds_data:
                relay_tds_data[main_relay_name] = {'main': [], 'backup': []}
            if backup_relay_name not in relay_tds_data:
                relay_tds_data[backup_relay_name] = {'main': [], 'backup': []}
            
            relay_tds_data[main_relay_name]['main'].append(main_tds)
            relay_tds_data[backup_relay_name]['backup'].append(backup_tds)
            
            # Pickup
            main_pickup = main_relay.get('pick_up', 0)
            backup_pickup = backup_relay.get('pick_up', 0)
            
            if main_relay_name not in relay_pickup_data:
                relay_pickup_data[main_relay_name] = {'main': [], 'backup': []}
            if backup_relay_name not in relay_pickup_data:
                relay_pickup_data[backup_relay_name] = {'main': [], 'backup': []}
            
            relay_pickup_data[main_relay_name]['main'].append(main_pickup)
            relay_pickup_data[backup_relay_name]['backup'].append(backup_pickup)

        # Crear box plots para TDS
        relay_names = sorted(list(relay_tds_data.keys()))
        main_tds_values = [relay_tds_data[relay]['main'] for relay in relay_names]
        backup_tds_values = [relay_tds_data[relay]['backup'] for relay in relay_names]

        relay_settings_fig.add_trace(
            go.Box(y=main_tds_values, name='TDS Main', marker_color='#3498db', boxpoints='all'),
            row=1, col=1
        )
        relay_settings_fig.add_trace(
            go.Box(y=backup_tds_values, name='TDS Backup', marker_color='#e74c3c', boxpoints='all'),
            row=1, col=1
        )

        # Crear box plots para Pickup
        main_pickup_values = [relay_pickup_data[relay]['main'] for relay in relay_names]
        backup_pickup_values = [relay_pickup_data[relay]['backup'] for relay in relay_names]

        relay_settings_fig.add_trace(
            go.Box(y=main_pickup_values, name='Pickup Main', marker_color='#2ecc71', boxpoints='all'),
            row=2, col=1
        )
        relay_settings_fig.add_trace(
            go.Box(y=backup_pickup_values, name='Pickup Backup', marker_color='#f1c40f', boxpoints='all'),
            row=2, col=1
        )

        relay_settings_fig.update_layout(
            height=800,
            showlegend=True,
            xaxis=dict(tickangle=45),
            xaxis2=dict(tickangle=45),
            yaxis_title="TDS",
            yaxis2_title="Pickup (A)",
            margin=dict(b=100)
        )

        # 8. Conclusiones y Recomendaciones
        conclusions = html.Div([
            html.H5("Análisis de Coordinación"),
            html.P([
                f"• Total de pares analizados: {len(all_pairs)}",
                html.Br(),
                f"• Tasa de coordinación: {len(coordinated_pairs)/len(all_pairs)*100:.1f}%",
                html.Br(),
                f"• TMT total: {tmt_total_scenario:.3f}s",
                html.Br(),
                f"• TMT promedio por par descoordinado: {tmt_total_scenario/len(uncoordinated_pairs):.3f}s"
            ]),
            html.H5("Patrones Identificados"),
            html.P([
                "• Relación entre TDS y coordinación: ",
                "Se observa una correlación significativa entre la relación TDS backup/main y la coordinación.",
                html.Br(),
                "• Impacto de las corrientes de falla: ",
                "Las fallas con mayor magnitud tienden a mostrar mejor coordinación.",
                html.Br(),
                "• Sensibilidad a parámetros: ",
                "El sistema es más sensible a variaciones en el TDS que en el pickup."
            ]),
            html.H5("Recomendaciones"),
            html.P([
                "• Ajuste de TDS: ",
                "Considerar incrementar el TDS de los relés de respaldo en los pares descoordinados.",
                html.Br(),
                "• Optimización de pickup: ",
                "Revisar los valores de pickup para mejorar la selectividad.",
                html.Br(),
                "• Análisis de líneas críticas: ",
                "Priorizar la revisión de las líneas con mayor TMT acumulado."
            ])
        ])

        # 9. Tabla de resumen detallado
        detailed_data = []
        for pair in all_pairs:
            main_relay = pair.get('main_relay', {})
            backup_relay = pair.get('backup_relay', {})
            detailed_data.append({
                'fault': pair.get('fault', 'N/A'),
                'main_line': main_relay.get('line', 'N/A'),
                'main_relay': main_relay.get('relay', 'N/A'),
                'main_tds': f"{main_relay.get('TDS', 0):.5f}",
                'main_pickup': f"{main_relay.get('pick_up', 0):.5f}",
                'main_ishc': f"{main_relay.get('Ishc', 0):.3f}",
                'main_time': f"{main_relay.get('Time_out', 0):.3f}",
                'backup_line': backup_relay.get('line', 'N/A'),
                'backup_relay': backup_relay.get('relay', 'N/A'),
                'backup_tds': f"{backup_relay.get('TDS', 0):.5f}",
                'backup_pickup': f"{backup_relay.get('pick_up', 0):.5f}",
                'backup_ishc': f"{backup_relay.get('Ishc', 0):.3f}",
                'backup_time': f"{backup_relay.get('Time_out', 0):.3f}",
                'delta_t': f"{pair.get('delta_t', 0):.3f}",
                'mt': f"{pair.get('mt', 0):.3f}",
                'status': "Coordinado" if pair.get('delta_t', 0) >= 0 else "Descoordinado"
            })

    except Exception as e:
        print(f"Error en el procesamiento de datos: {e}")
        traceback.print_exc()

    return (coordinated_fig, coordinated_table_data, uncoordinated_fig, 
            uncoordinated_table_data, mt_fig, coordination_fig, correlation_fig,
            sensitivity_fig, line_fig, relay_fig, fault_fig, conclusions, detailed_data,
            relay_settings_fig)

# --- Fase 5: Ejecutar la Aplicación ---
if __name__ == '__main__':
    print("\nIniciando servidor Dash...")
    print(f"Accede a la aplicación en: http://127.0.0.1:8050/")
    app.run_server(debug=True)  # debug=True para desarrollo 

# Agregar estilos CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Análisis de Coordinación de Protecciones</title>
        {%favicon%}
        {%css%}
        <style>
            .metric-box {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
                margin: 10px;
                flex: 1;
            }
            .metric-box h5 {
                margin: 0;
                color: #7f8c8d;
            }
            .metric-box h3 {
                margin: 10px 0 0 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
''' 
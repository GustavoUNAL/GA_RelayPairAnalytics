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
import datetime

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

# --- Función para validar que las curvas TCC no se crucen ---
def validate_tcc_curves(main_tds, main_pickup, backup_tds, backup_pickup, num_points=100):
    """
    Valida que las curvas TCC no se crucen en ningún punto.
    Retorna True si las curvas no se cruzan, False si hay cruce.
    """
    # Crear rango de corrientes para comparar
    min_pickup = min(main_pickup, backup_pickup)
    max_current = max(main_pickup, backup_pickup) * 20  # Multiplicador para asegurar rango suficiente
    i_range = np.logspace(np.log10(min_pickup * 1.05), np.log10(max_current), num=num_points)
    
    # Calcular tiempos para ambas curvas
    main_times = calculate_inverse_time_curve(main_tds, main_pickup, i_range)
    backup_times = calculate_inverse_time_curve(backup_tds, backup_pickup, i_range)
    
    # Verificar que la curva de respaldo siempre esté por encima de la principal
    # (tiempo de respaldo siempre mayor que tiempo principal)
    for main_time, backup_time in zip(main_times, backup_times):
        if not np.isnan(main_time) and not np.isnan(backup_time):
            if backup_time <= main_time:
                return False
    return True

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
        # Primero validar que las curvas no se crucen
        curves_valid = validate_tcc_curves(
            main_relay_info.get('TDS', 0),
            main_relay_info.get('pick_up', 0),
            backup_relay_info.get('TDS', 0),
            backup_relay_info.get('pick_up', 0)
        )
        
        # Un par es coordinado solo si cumple todas las condiciones
        if delta_t > CTI and mt == 0 and curves_valid:
            coordinated_pairs.append(pair_info)
        else:
            uncoordinated_pairs.append(pair_info)
            # Agregar razón de descoordinación
            if not curves_valid:
                pair_info['razon_descoordinacion'] = 'Curvas TCC se cruzan'
            elif delta_t <= CTI:
                pair_info['razon_descoordinacion'] = f'Δt ({delta_t:.3f}s) <= CTI ({CTI}s)'
            elif mt != 0:
                pair_info['razon_descoordinacion'] = f'MT ({mt:.3f}s) != 0'

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
    print(f"Coordinados (delta_t > {CTI}s, mt = 0): {len(coordinated_pairs)}")
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

# Calcular métricas de TMT para el layout
if len(uncoordinated_pairs) > 0:
    tmt_promedio = f"{tmt_total_scenario/len(uncoordinated_pairs):.3f}s"
    tmt_std = f"{np.std([abs(p.get('mt', 0)) for p in uncoordinated_pairs]):.3f}s"
else:
    tmt_promedio = "N/A"
    tmt_std = "N/A"

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1(f"Análisis de Coordinación de Protecciones", style={'textAlign': 'center'}),
    html.H1(f" Escenario Base Automatizado", style={'textAlign': 'center'}),
    html.H3(f"TMT: {tmt_total_scenario:.3f}, Pares de Relés: {total_valid_pairs_scenario}", style={'textAlign': 'center', 'marginTop': '-10px', 'marginBottom': '20px'}),
    html.H4(f"Criterios de Coordinación: CTI > {CTI}s, MT = 0, Curvas TCC sin cruce", 
            style={'textAlign': 'center', 'marginTop': '-10px', 'marginBottom': '20px'}),
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
                                html.P(f"Promedio por par: {tmt_promedio}", style={'color': '#e67e22'}),
                                html.P(f"Desviación estándar: {tmt_std}", style={'color': '#e67e22'}),
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
                            {"name": "Estado", "id": "status"},
                            {"name": "Razón Descoordinación", "id": "razon_descoordinacion"}
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
        # Procesar par coordinado seleccionado
        if coordinated_idx is not None and coordinated_idx < len(coordinated_pairs):
            pair = coordinated_pairs[coordinated_idx]
            main_relay = pair.get('main_relay', {})
            backup_relay = pair.get('backup_relay', {})
            
            # Crear rango de corrientes para las curvas
            main_pickup = main_relay.get('pick_up', 0)
            backup_pickup = backup_relay.get('pick_up', 0)
            min_pickup = min(main_pickup, backup_pickup)
            max_ishc = max(main_relay.get('Ishc', 0), backup_relay.get('Ishc', 0))
            i_range = np.logspace(np.log10(min_pickup * MIN_CURRENT_MULTIPLIER),
                                np.log10(max_ishc * MAX_CURRENT_MULTIPLIER),
                                num=100)
            
            # Calcular curvas
            main_times = calculate_inverse_time_curve(main_relay.get('TDS', 0),
                                                    main_pickup, i_range)
            backup_times = calculate_inverse_time_curve(backup_relay.get('TDS', 0),
                                                      backup_pickup, i_range)
            
            # Crear gráfica
            coordinated_fig = go.Figure()
            
            # Agregar curvas
            coordinated_fig.add_trace(
                go.Scatter(y=main_times, x=i_range, name='Main',
                          mode='lines', line=dict(color='#3498db'),
                          hovertemplate='t: %{y:.3f}s<br>I: %{x:.1f}A')
            )
            coordinated_fig.add_trace(
                go.Scatter(y=backup_times, x=i_range, name='Backup',
                          mode='lines', line=dict(color='#e74c3c'),
                          hovertemplate='t: %{y:.3f}s<br>I: %{x:.1f}A')
            )
            
            # Agregar puntos de operación
            coordinated_fig.add_trace(
                go.Scatter(y=[main_relay.get('Time_out', 0)],
                          x=[main_relay.get('Ishc', 0)],
                          name='Main Op. Point', mode='markers',
                          marker=dict(size=12, color='#3498db', symbol='star'),
                          hovertemplate='t_m: %{y:.3f}s<br>I_m: %{x:.1f}A')
            )
            coordinated_fig.add_trace(
                go.Scatter(y=[backup_relay.get('Time_out', 0)],
                          x=[backup_relay.get('Ishc', 0)],
                          name='Backup Op. Point', mode='markers',
                          marker=dict(size=12, color='#e74c3c', symbol='star'),
                          hovertemplate='t_b: %{y:.3f}s<br>I_b: %{x:.1f}A')
            )
            
            # Agregar línea vertical para delta_t
            coordinated_fig.add_trace(
                go.Scatter(y=[main_relay.get('Time_out', 0),
                             backup_relay.get('Time_out', 0)],
                          x=[main_relay.get('Ishc', 0),
                             main_relay.get('Ishc', 0)],
                          name='Δt', mode='lines',
                          line=dict(color='#2ecc71', width=2, dash='dash'),
                          hovertemplate='Δt: %{text}',
                          text=[f'Δt = {pair.get("delta_t", 0):.3f}s'])
            )
            
            # Actualizar layout
            coordinated_fig.update_layout(
                title=f'Par Coordinado (Δt = {pair.get("delta_t", 0):.3f}s)',
                yaxis_title='Tiempo de operación (s)',
                xaxis_title='Corriente (A)',
                xaxis_type='log',
                yaxis_type='log',
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    gridcolor='lightgray',
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=1
                ),
                yaxis=dict(
                    gridcolor='lightgray',
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=1
                )
            )
            
            # Actualizar tabla de datos
            coordinated_table_data = [
                {"parameter": "Falla (%)", "value": pair.get('fault', 'N/A')},
                {"parameter": "Línea Principal", "value": main_relay.get('line', 'N/A')},
                {"parameter": "Relé Principal", "value": main_relay.get('relay', 'N/A')},
                {"parameter": "TDS (Main)", "value": f"{main_relay.get('TDS', 0):.5f}"},
                {"parameter": "Pickup (Main)", "value": f"{main_relay.get('pick_up', 0):.5f}"},
                {"parameter": "I_shc (Main)", "value": f"{main_relay.get('Ishc', 0):.3f}"},
                {"parameter": "t_m (Main)", "value": f"{main_relay.get('Time_out', 0):.3f}"},
                {"parameter": "Línea Backup", "value": backup_relay.get('line', 'N/A')},
                {"parameter": "Relé Backup", "value": backup_relay.get('relay', 'N/A')},
                {"parameter": "TDS (Backup)", "value": f"{backup_relay.get('TDS', 0):.5f}"},
                {"parameter": "Pickup (Backup)", "value": f"{backup_relay.get('pick_up', 0):.5f}"},
                {"parameter": "I_shc (Backup)", "value": f"{backup_relay.get('Ishc', 0):.3f}"},
                {"parameter": "t_b (Backup)", "value": f"{backup_relay.get('Time_out', 0):.3f}"},
                {"parameter": "Δt", "value": f"{pair.get('delta_t', 0):.3f}"},
                {"parameter": "MT", "value": f"{pair.get('mt', 0):.3f}"}
            ]

        # Procesar par descoordinado seleccionado
        if uncoordinated_idx is not None and uncoordinated_idx < len(uncoordinated_pairs):
            pair = uncoordinated_pairs[uncoordinated_idx]
            main_relay = pair.get('main_relay', {})
            backup_relay = pair.get('backup_relay', {})
            
            # Crear rango de corrientes para las curvas
            main_pickup = main_relay.get('pick_up', 0)
            backup_pickup = backup_relay.get('pick_up', 0)
            min_pickup = min(main_pickup, backup_pickup)
            max_ishc = max(main_relay.get('Ishc', 0), backup_relay.get('Ishc', 0))
            i_range = np.logspace(np.log10(min_pickup * MIN_CURRENT_MULTIPLIER),
                                np.log10(max_ishc * MAX_CURRENT_MULTIPLIER),
                                num=100)
            
            # Calcular curvas
            main_times = calculate_inverse_time_curve(main_relay.get('TDS', 0),
                                                    main_pickup, i_range)
            backup_times = calculate_inverse_time_curve(backup_relay.get('TDS', 0),
                                                      backup_pickup, i_range)
            
            # Crear gráfica
            uncoordinated_fig = go.Figure()
            
            # Agregar curvas
            uncoordinated_fig.add_trace(
                go.Scatter(y=main_times, x=i_range, name='Main',
                          mode='lines', line=dict(color='#3498db'),
                          hovertemplate='t: %{y:.3f}s<br>I: %{x:.1f}A')
            )
            uncoordinated_fig.add_trace(
                go.Scatter(y=backup_times, x=i_range, name='Backup',
                          mode='lines', line=dict(color='#e74c3c'),
                          hovertemplate='t: %{y:.3f}s<br>I: %{x:.1f}A')
            )
            
            # Agregar puntos de operación
            uncoordinated_fig.add_trace(
                go.Scatter(y=[main_relay.get('Time_out', 0)],
                          x=[main_relay.get('Ishc', 0)],
                          name='Main Op. Point', mode='markers',
                          marker=dict(size=12, color='#3498db', symbol='star'),
                          hovertemplate='t_m: %{y:.3f}s<br>I_m: %{x:.1f}A')
            )
            uncoordinated_fig.add_trace(
                go.Scatter(y=[backup_relay.get('Time_out', 0)],
                          x=[backup_relay.get('Ishc', 0)],
                          name='Backup Op. Point', mode='markers',
                          marker=dict(size=12, color='#e74c3c', symbol='star'),
                          hovertemplate='t_b: %{y:.3f}s<br>I_b: %{x:.1f}A')
            )
            
            # Agregar línea vertical para delta_t
            uncoordinated_fig.add_trace(
                go.Scatter(y=[main_relay.get('Time_out', 0),
                             backup_relay.get('Time_out', 0)],
                          x=[main_relay.get('Ishc', 0),
                             main_relay.get('Ishc', 0)],
                          name='Δt', mode='lines',
                          line=dict(color='#e74c3c', width=2, dash='dash'),
                          hovertemplate='Δt: %{text}',
                          text=[f'Δt = {pair.get("delta_t", 0):.3f}s'])
            )
            
            # Actualizar layout
            uncoordinated_fig.update_layout(
                title=f'Par Descoordinado (Δt = {pair.get("delta_t", 0):.3f}s)',
                yaxis_title='Tiempo de operación (s)',
                xaxis_title='Corriente (A)',
                xaxis_type='log',
                yaxis_type='log',
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    gridcolor='lightgray',
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=1
                ),
                yaxis=dict(
                    gridcolor='lightgray',
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=1
                )
            )
            
            # Actualizar tabla de datos
            uncoordinated_table_data = [
                {"parameter": "Falla (%)", "value": pair.get('fault', 'N/A')},
                {"parameter": "Línea Principal", "value": main_relay.get('line', 'N/A')},
                {"parameter": "Relé Principal", "value": main_relay.get('relay', 'N/A')},
                {"parameter": "TDS (Main)", "value": f"{main_relay.get('TDS', 0):.5f}"},
                {"parameter": "Pickup (Main)", "value": f"{main_relay.get('pick_up', 0):.5f}"},
                {"parameter": "I_shc (Main)", "value": f"{main_relay.get('Ishc', 0):.3f}"},
                {"parameter": "t_m (Main)", "value": f"{main_relay.get('Time_out', 0):.3f}"},
                {"parameter": "Línea Backup", "value": backup_relay.get('line', 'N/A')},
                {"parameter": "Relé Backup", "value": backup_relay.get('relay', 'N/A')},
                {"parameter": "TDS (Backup)", "value": f"{backup_relay.get('TDS', 0):.5f}"},
                {"parameter": "Pickup (Backup)", "value": f"{backup_relay.get('pick_up', 0):.5f}"},
                {"parameter": "I_shc (Backup)", "value": f"{backup_relay.get('Ishc', 0):.3f}"},
                {"parameter": "t_b (Backup)", "value": f"{backup_relay.get('Time_out', 0):.3f}"},
                {"parameter": "Δt", "value": f"{pair.get('delta_t', 0):.3f}"},
                {"parameter": "MT", "value": f"{pair.get('mt', 0):.3f}"}
            ]

            # Actualizar gráfica de MT
            mt_values = [abs(pair.get('mt', 0)) for pair in uncoordinated_pairs]
            mt_fig = go.Figure(data=[
                go.Bar(x=list(range(len(mt_values))), y=mt_values,
                      marker_color='#e74c3c',
                      hovertemplate='|mt|: %{y:.3f}s')
            ])
            mt_fig.update_layout(
                title='Magnitud de Penalización |mt| (Descoordinados)',
                xaxis_title='Índice del Par',
                yaxis_title='|mt| (s)',
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    gridcolor='lightgray',
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=1
                ),
                yaxis=dict(
                    gridcolor='lightgray',
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=1
                )
            )

        # Generar gráficas de analítica automáticamente
        # 1. Gráfica de Análisis de Coordinación
        coordination_fig = go.Figure()
        delta_t_values = [p.get('delta_t', 0) for p in all_pairs]
        coordination_fig.add_trace(go.Histogram(
            y=delta_t_values,
            name='Distribución Δt',
            marker_color='#3498db',
            nbinsx=30
        ))
        coordination_fig.update_layout(
            title='Distribución de Tiempos de Coordinación (Δt)',
            yaxis_title='Δt (segundos)',
            xaxis_title='Frecuencia',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 2. Matriz de Correlación
        correlation_data = pd.DataFrame([
            {
                'TDS_Main': p.get('main_relay', {}).get('TDS', 0),
                'Pickup_Main': p.get('main_relay', {}).get('pick_up', 0),
                'Ishc_Main': p.get('main_relay', {}).get('Ishc', 0),
                'TDS_Backup': p.get('backup_relay', {}).get('TDS', 0),
                'Pickup_Backup': p.get('backup_relay', {}).get('pick_up', 0),
                'Ishc_Backup': p.get('backup_relay', {}).get('Ishc', 0),
                'Delta_t': p.get('delta_t', 0)
            } for p in all_pairs
        ])
        corr_matrix = correlation_data.corr()
        correlation_fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        correlation_fig.update_layout(
            title='Matriz de Correlación',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 3. Análisis de Sensibilidad
        sensitivity_fig = go.Figure()
        # Ishc vs Delta_t
        sensitivity_fig.add_trace(go.Scatter(
            x=[p.get('main_relay', {}).get('Ishc', 0) for p in all_pairs],
            y=[p.get('delta_t', 0) for p in all_pairs],
            mode='markers',
            name='Ishc Main vs Δt',
            marker=dict(color='#3498db')
        ))
        sensitivity_fig.add_trace(go.Scatter(
            x=[p.get('backup_relay', {}).get('Ishc', 0) for p in all_pairs],
            y=[p.get('delta_t', 0) for p in all_pairs],
            mode='markers',
            name='Ishc Backup vs Δt',
            marker=dict(color='#e74c3c')
        ))
        sensitivity_fig.update_layout(
            title='Sensibilidad: Corriente vs Δt',
            xaxis_title='Corriente (A)',
            yaxis_title='Δt (segundos)',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 4. Análisis por Línea
        line_data = {}
        for pair in all_pairs:
            main_line = pair.get('main_relay', {}).get('line', 'N/A')
            backup_line = pair.get('backup_relay', {}).get('line', 'N/A')
            if main_line not in line_data:
                line_data[main_line] = {'coordinated': 0, 'uncoordinated': 0}
            if backup_line not in line_data:
                line_data[backup_line] = {'coordinated': 0, 'uncoordinated': 0}
            
            if pair.get('delta_t', 0) >= 0:
                line_data[main_line]['coordinated'] += 1
                line_data[backup_line]['coordinated'] += 1
            else:
                line_data[main_line]['uncoordinated'] += 1
                line_data[backup_line]['uncoordinated'] += 1

        line_fig = go.Figure(data=[
            go.Bar(name='Coordinados', x=list(line_data.keys()),
                  y=[d['coordinated'] for d in line_data.values()],
                  marker_color='#2ecc71'),
            go.Bar(name='Descoordinados', x=list(line_data.keys()),
                  y=[d['uncoordinated'] for d in line_data.values()],
                  marker_color='#e74c3c')
        ])
        line_fig.update_layout(
            title='Análisis de Coordinación por Línea',
            xaxis_title='Línea',
            yaxis_title='Número de Pares',
            barmode='stack',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 5. Análisis por Relé
        relay_data = {}
        for pair in all_pairs:
            main_relay = pair.get('main_relay', {}).get('relay', 'N/A')
            backup_relay = pair.get('backup_relay', {}).get('relay', 'N/A')
            if main_relay not in relay_data:
                relay_data[main_relay] = {'coordinated': 0, 'uncoordinated': 0}
            if backup_relay not in relay_data:
                relay_data[backup_relay] = {'coordinated': 0, 'uncoordinated': 0}
            
            if pair.get('delta_t', 0) >= 0:
                relay_data[main_relay]['coordinated'] += 1
                relay_data[backup_relay]['coordinated'] += 1
            else:
                relay_data[main_relay]['uncoordinated'] += 1
                relay_data[backup_relay]['uncoordinated'] += 1

        relay_fig = go.Figure(data=[
            go.Bar(name='Coordinados', x=list(relay_data.keys()),
                  y=[d['coordinated'] for d in relay_data.values()],
                  marker_color='#2ecc71'),
            go.Bar(name='Descoordinados', x=list(relay_data.keys()),
                  y=[d['uncoordinated'] for d in relay_data.values()],
                  marker_color='#e74c3c')
        ])
        relay_fig.update_layout(
            title='Análisis de Coordinación por Relé',
            xaxis_title='Relé',
            yaxis_title='Número de Pares',
            barmode='stack',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 6. Análisis de Fallas
        fault_data = {}
        for pair in all_pairs:
            fault = pair.get('fault', 'N/A')
            if fault not in fault_data:
                fault_data[fault] = {'coordinated': 0, 'uncoordinated': 0}
            
            if pair.get('delta_t', 0) >= 0:
                fault_data[fault]['coordinated'] += 1
            else:
                fault_data[fault]['uncoordinated'] += 1

        fault_fig = go.Figure(data=[
            go.Bar(name='Coordinados', x=list(fault_data.keys()),
                  y=[d['coordinated'] for d in fault_data.values()],
                  marker_color='#2ecc71'),
            go.Bar(name='Descoordinados', x=list(fault_data.keys()),
                  y=[d['uncoordinated'] for d in fault_data.values()],
                  marker_color='#e74c3c')
        ])
        fault_fig.update_layout(
            title='Análisis de Coordinación por Porcentaje de Falla',
            xaxis_title='Porcentaje de Falla',
            yaxis_title='Número de Pares',
            barmode='stack',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 7. Gráfica de Ajustes de Relés
        relay_settings_fig = go.Figure()
        # Ishc vs Time para relés principales
        relay_settings_fig.add_trace(go.Scatter(
            x=[p.get('main_relay', {}).get('Ishc', 0) for p in all_pairs],
            y=[p.get('main_relay', {}).get('Time_out', 0) for p in all_pairs],
            mode='markers',
            name='Relés Principales',
            marker=dict(color='#3498db')
        ))
        # Ishc vs Time para relés de respaldo
        relay_settings_fig.add_trace(go.Scatter(
            x=[p.get('backup_relay', {}).get('Ishc', 0) for p in all_pairs],
            y=[p.get('backup_relay', {}).get('Time_out', 0) for p in all_pairs],
            mode='markers',
            name='Relés de Respaldo',
            marker=dict(color='#e74c3c')
        ))
        relay_settings_fig.update_layout(
            title='Distribución de Corrientes vs Tiempos de Operación',
            xaxis_title='Corriente (A)',
            yaxis_title='Tiempo de Operación (s)',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 8. Tabla de resumen detallado
        detailed_data = []
        for pair in all_pairs:
            main_relay = pair.get('main_relay', {})
            backup_relay = pair.get('backup_relay', {})
            detailed_data.append({
                'fault': pair.get('fault', 'N/A'),
                'main_line': main_relay.get('line', 'N/A'),
                'main_relay': main_relay.get('relay', 'N/A'),
                'main_tds': f"{main_relay.get('TDS', 0):.3f}",
                'main_pickup': f"{main_relay.get('pick_up', 0):.3f}",
                'main_ishc': f"{main_relay.get('Ishc', 0):.3f}",
                'main_time': f"{main_relay.get('Time_out', 0):.3f}",
                'backup_line': backup_relay.get('line', 'N/A'),
                'backup_relay': backup_relay.get('relay', 'N/A'),
                'backup_tds': f"{backup_relay.get('TDS', 0):.3f}",
                'backup_pickup': f"{backup_relay.get('pick_up', 0):.3f}",
                'backup_ishc': f"{backup_relay.get('Ishc', 0):.3f}",
                'backup_time': f"{backup_relay.get('Time_out', 0):.3f}",
                'delta_t': f"{pair.get('delta_t', 0):.3f}",
                'mt': f"{pair.get('mt', 0):.3f}",
                'status': 'Coordinado' if pair.get('delta_t', 0) > CTI and pair.get('mt', 0) == 0 else 'Descoordinado',
                'razon_descoordinacion': pair.get('razon_descoordinacion', 'N/A')
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
    # --- Exportar informe estadístico de pares de relés (solo CSV) ---
    import datetime

    # Construir DataFrame con todos los pares
    reporte_data = []
    for pair in all_pairs:
        main = pair.get('main_relay', {})
        backup = pair.get('backup_relay', {})
        reporte_data.append({
            'Falla (%)': pair.get('fault', 'N/A'),
            'Línea Principal': main.get('line', 'N/A'),
            'Relé Principal': main.get('relay', 'N/A'),
            'TDS (Main)': main.get('TDS', 'N/A'),
            'Pickup (Main)': main.get('pick_up', 'N/A'),
            'I_shc (Main)': main.get('Ishc', 'N/A'),
            't_m (Main)': main.get('Time_out', 'N/A'),
            'Línea Backup': backup.get('line', 'N/A'),
            'Relé Backup': backup.get('relay', 'N/A'),
            'TDS (Backup)': backup.get('TDS', 'N/A'),
            'Pickup (Backup)': backup.get('pick_up', 'N/A'),
            'I_shc (Backup)': backup.get('Ishc', 'N/A'),
            't_b (Backup)': backup.get('Time_out', 'N/A'),
            'Δt': pair.get('delta_t', 'N/A'),
            'MT': pair.get('mt', 'N/A'),
            'Estado': 'Coordinado' if (pair.get('delta_t', 0) > CTI and pair.get('mt', 0) == 0 and 
                                     validate_tcc_curves(main.get('TDS', 0), main.get('pick_up', 0),
                                                       backup.get('TDS', 0), backup.get('pick_up', 0))) 
                     else 'Descoordinado',
            'Razón Descoordinación': pair.get('razon_descoordinacion', 'N/A')
        })

    reporte_df = pd.DataFrame(reporte_data)

    # Crear nombre de archivo con fecha y hora para evitar sobrescribir
    fecha = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"data/processed/reporte_estadistico_reles_{fecha}.csv"

    # Guardar como CSV
    reporte_df.to_csv(csv_path, index=False)

    print(f"\nInforme estadístico exportado a: {csv_path}\n")
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
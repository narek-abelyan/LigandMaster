import base64
import io
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html, Input, Output, State
from dash_resizable_panels import PanelGroup, Panel, PanelResizeHandle
import dash_bootstrap_components as dbc
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Draw, Descriptors
from PIL import Image
from dash import no_update
import plotly.io as pio

pio.templates.default = "plotly"
pio.templates.default = None

# ========== Параметры ==========
CSV_PATH = "molecules_scaf_maccs.csv"  # путь к CSV
SCATTER_SAMPLE = 20000


def prepare_df(new_df: pd.DataFrame) -> pd.DataFrame:
    if 'ID' in new_df.columns:
        if pd.api.types.is_numeric_dtype(new_df['ID']):
            new_df['ID'] = new_df['ID'].astype(int).astype(str)
        else:
            new_df['ID'] = new_df['ID'].astype(str).str.strip()

    required = {"ID", "SMILES"}
    missing_basic = required - set(new_df.columns)
    if missing_basic:
        raise ValueError(f"В CSV отсутствуют необходимые столбцы: {missing_basic}")

    if "MolWt" not in new_df.columns or "TPSA" not in new_df.columns:
        print(f"⚠️ Вычисляем отсутствующие колонки: MolWt и/или TPSA из SMILES...")
        mols = [Chem.MolFromSmiles(str(s)) for s in new_df["SMILES"]]
        mw_list = []
        tpsa_list = []

        for mol in mols:
            if mol is None:
                mw_list.append(np.nan)
                tpsa_list.append(np.nan)
            else:
                mw_list.append(Descriptors.MolWt(mol))
                tpsa_list.append(Descriptors.TPSA(mol))

        if "MolWt" not in new_df.columns:
            new_df["MolWt"] = mw_list
        if "TPSA" not in new_df.columns:
            new_df["TPSA"] = tpsa_list

        print("✅ MolWt и TPSA успешно добавлены.")

    required = {"ID", "SMILES", "MolWt", "TPSA"}
    missing = required - set(new_df.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют необходимые столбцы: {missing}")
    return new_df


def parse_uploaded_csv(contents: str) -> pd.DataFrame:
    _, content_string = contents.split(',', 1)
    decoded = base64.b64decode(content_string)

    for sep in [";", ","]:
        try:
            parsed = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=sep)
            if len(parsed.columns) > 1:
                return prepare_df(parsed)
        except Exception:
            continue
    return prepare_df(pd.read_csv(io.StringIO(decoded.decode('utf-8'))))


# ========== Загрузка данных ==========
df = prepare_df(pd.read_csv(CSV_PATH, sep=None, engine="python"))

numeric_cols = df.select_dtypes(include='number').columns.tolist()
dropdown_options = [{'label': c, 'value': c} for c in numeric_cols]
color_options = [
    {'label': c, 'value': v} for c, v in [
        ('Blue', '#1f77b4'),
        ('Orange', '#ff7f0e'),
        ('Green', '#2ca02c'),
        ('Red', '#d62728'),
        ('Purple', '#9467bd'),
        ('NPurple', '#800080'),
        ('Cyan', '#17becf'),
        ('Magenta', '#e377c2'),
        ('Yellow', '#bcbd22'),
        ('Brown', '#8c564b'),
        ('Gray', '#7f7f7f'),
        ('Lime', '#aec7e8'),
        ('Teal', '#98df8a'),
        ('Pink', '#ffb6c1'),
        ('Gold', '#ffd700'),
        ('DBlue', '#003f5c'),
        ('DGreen', '#2f4b2f'),
        ('Coral', '#ff6f61'),
        ('Violet', '#8a2be2'),
        ('Turquoise', '#40e0d0'),
        ('Salmon', '#fa8072')
    ]
]

# ========== ДОПОЛНИТЕЛЬНЫЕ СВОЙСТВА RDKit ==========
extra_properties = {
    # Липофильность / полярность
    "MolLogP": Descriptors.MolLogP,
    "ExactMolWt": Descriptors.ExactMolWt,
    "MolWt": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "MolMR": Descriptors.MolMR,

    # H-bond свойства
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,

    # Размер и состав
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,

    # Кольца / гибкость
    "RingCount": Descriptors.RingCount,
    "NumSaturatedRings": Descriptors.NumSaturatedRings,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,

    # Drug-likeness
    "QED": Descriptors.qed,

    # Топология / сложность
    "BertzCT": Descriptors.BertzCT,
    "BalabanJ": Descriptors.BalabanJ,
    "Chi0v": Descriptors.Chi0v,
    "Chi1v": Descriptors.Chi1v,
    "Chi2v": Descriptors.Chi2v,
    "Chi3v": Descriptors.Chi3v,
    "Chi4v": Descriptors.Chi4v,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
}

extra_property_options = [{'label': name, 'value': name} for name in extra_properties.keys()]
extra_property_groups = {
    "Lipophilicity / polarity": ["MolLogP", "ExactMolWt", "MolWt", "TPSA", "MolMR"],
    "H-bond properties": ["NumHDonors", "NumHAcceptors"],
    "Size and composition": ["NumValenceElectrons", "HeavyAtomMolWt", "HeavyAtomCount"],
    "Rings / flexibility": ["NumRotatableBonds", "RingCount", "NumSaturatedRings", "FractionCSP3", "NumAromaticRings", "NumAliphaticRings"],
    "Drug-likeness": ["QED"],
    "Topology / complexity": ["BertzCT", "BalabanJ", "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v", "Kappa1", "Kappa2", "Kappa3"],
}


# ========== Формат колонок для DataTable ==========
def build_table_columns(local_df):
    n_cols = local_df.select_dtypes(include='number').columns.tolist()
    t_cols = []
    for c in local_df.columns:
        if c in n_cols:
            t_cols.append({
                "name": c,
                "id": c,
                "type": "numeric",
                "format": {"specifier": ".2f"}
            })
        else:
            t_cols.append({"name": c, "id": c})
    add_col = {"name": "S", "id": "__add__"}
    return n_cols, [{'label': c, 'value': c} for c in n_cols], [add_col] + t_cols


def with_add_marker(records):
    out = []
    for row in records:
        r = dict(row)
        r["__add__"] = "⬜"
        out.append(r)
    return out


def apply_selected_markers(records, selected_data):
    selected_data = selected_data or []
    selected_keys = {(str(r.get("ID")), str(r.get("SMILES", ""))) for r in selected_data}
    out = []
    for row in records:
        r = dict(row)
        key = (str(r.get("ID")), str(r.get("SMILES", "")))
        r["__add__"] = "✅" if key in selected_keys else "⬜"
        out.append(r)
    return out


numeric_cols, dropdown_options, table_columns = build_table_columns(df)


# ========== Вспомогательные функции ==========
def pil_to_b64(pil_img, fmt="PNG"):
    buff = io.BytesIO()
    pil_img.save(buff, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buff.getvalue()).decode()}"


def smiles_to_base64(smiles, img_size=(800, 600)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        im = Image.new("RGB", img_size, (255, 255, 255))
        return pil_to_b64(im)
    try:
        img = Draw.MolToImage(mol, size=img_size)
    except Exception:
        img = Image.new("RGB", img_size, (255, 255, 255))
    return pil_to_b64(img)


def smiles_to_thumb_md(smiles, size=(120, 80)):
    return f"![mol]({smiles_to_base64(smiles, img_size=size)})"


def smiles_to_thumb_html(smiles, size=(120, 80)):
    src = smiles_to_base64(smiles, img_size=(220, 160))
    return f"<img src='{src}' style='height:90px; width:130px; object-fit:contain; image-rendering:auto;'/>"


# ========== Инициализация Dash ==========
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# ========== Стили ==========
DROPDOWN_STYLE = {
    "width": "100%",
    "height": "36px",
    "fontSize": "14px",
    "borderRadius": "8px",
    "border": "1px solid #d0d0d0",
    "backgroundColor": "#fafafa",
    "paddingLeft": "8px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
}
DROPDOWN_CONTAINER_STYLE = {
    "display": "flex",
    "gap": "10px",
    "marginBottom": "10px",
}
SLIDER_CONTAINER_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
    "flex": 1
}

# ========== HTML-шаблон ==========
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Molecular Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: "Segoe UI", Roboto, sans-serif;
                background-color: #f8f9fb;
                margin: 0;
                padding: 0;
                overflow: hidden;
            }
            html, body {
                height: 100%;
            }
            .dash-slider {
                width: 100%;
                height: 4px;
                background: transparent;
                border-radius: 2px;
            }
            .dash-slider .rc-slider-handle {
                border: none !important;
                background: #007bff !important;
                box-shadow: 0 0 6px rgba(0,123,255,0.3);
                width: 16px;
                height: 16px;
                margin-top: -6px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# ========== Функция красивой гистограммы ==========
def make_pretty_hist(df, x, color, title, nbinsx=40, compute_kde=False):
    fig = go.Figure()
    mean_val = df[x].mean()
    std_val = df[x].std()

    # Гистограмма
    fig.add_trace(go.Histogram(
        x=df[x],
        nbinsx=nbinsx,
        marker=dict(color=color, line=dict(width=0.5, color='white')),
        opacity=0.8,
        name="Histogram"
    ))

    # KDE линия — только если compute_kde True
    if compute_kde:
        try:
            kde_data = df[x].dropna()
            if len(kde_data) > 2000:
                kde_data = kde_data.sample(2000, random_state=1)
            kde = gaussian_kde(kde_data)
            x_range = np.linspace(df[x].min(), df[x].max(), 100)
            y_kde = kde(x_range)
            y_kde_scaled = y_kde * len(df[x]) * (x_range[1] - x_range[0])
            fig.add_trace(go.Scatter(
                x=x_range, y=y_kde_scaled,
                mode='lines',
                line=dict(color="black", width=1),
                name="Density"
            ))
        except Exception:
            pass

    # Градиентная тень ±σ
    factors = [0.9, 0.7, 0.5, 0.3, 0.1]
    opacities = [0.025, 0.05, 0.075, 0.1, 0.125]
    for factor, op in zip(factors, opacities):
        fig.add_vrect(
            x0=mean_val-factor*std_val, x1=mean_val+factor*std_val,
            fillcolor=color, opacity=op, line_width=0
        )

    # Средняя пунктирная линия
    fig.add_vline(
        x=mean_val,
        line_width=1,
        line_color="black",
        line_dash="dot",
        annotation_text=f"Mean = {mean_val:.2f}",
        annotation_position="top right",
        annotation_font=dict(size=12, color=color)
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        margin=dict(l=45, r=10, t=40, b=60),
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        font=dict(family="Segoe UI", size=12),
        xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
        bargap=0.05,
        showlegend=False
    )
    return fig


# ========== Layout ==========
app.layout = html.Div(
    style={"height": "100vh", "margin": "0", "padding": "0"},
    children=[
        dcc.Store(id="selected-molecules-store", data=[]),
        dcc.Tabs(
            id="page-tabs",
            value="main-page",
            children=[
                dcc.Tab(
                    label="Main Page",
                    value="main-page",
                    style={"height": "42px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                    selected_style={"height": "42px", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": "600"}
                ),
                dcc.Tab(
                    label="Selected Molecules",
                    value="selected-page",
                    style={"height": "42px", "display": "flex", "alignItems": "center", "justifyContent": "center"},
                    selected_style={"height": "42px", "display": "flex", "alignItems": "center", "justifyContent": "center", "fontWeight": "600"}
                ),
            ],
            style={"padding": "0 4px", "backgroundColor": "#ffffff", "height": "44px"}
        ),
        html.Div(
            id="main-page-container",
            style={"display": "block", "height": "calc(100vh - 56px)"},
            children=[PanelGroup(
                direction="horizontal",
                children=[
                Panel(
                    defaultSizePercentage=70,
                    minSizePixels=400,
                    style={"padding": "8px"},
                    children=[
                        PanelGroup(
                            direction="vertical",
                            children=[
                                # ---------- Scatter и TPSA ----------
                                Panel(
                                    defaultSizePercentage=40,
                                    children=[
                                        PanelGroup(
                                            direction="horizontal",
                                            children=[
                                                Panel(
                                                    defaultSizePercentage=50,
                                                    children=[
                                                        html.Div(
                                                            id="scatter-container",
                                                            style={
                                                                "height": "100%",
                                                                "border": "1px solid #e5e7eb",
                                                                "borderRadius": "8px",
                                                                "padding": "10px",
                                                                "display": "flex",
                                                                "flexDirection": "column",
                                                                "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"
                                                            },
                                                            children=[
                                                                html.Div(
                                                                    style=DROPDOWN_CONTAINER_STYLE,
                                                                    children=[
                                                                        html.Div(style={"flex": 1}, children=[
                                                                            dcc.Dropdown(id="scatter-x-dropdown", options=dropdown_options, value="MolWt", style=DROPDOWN_STYLE)
                                                                        ]),
                                                                        html.Div(style={"flex": 1}, children=[
                                                                            dcc.Dropdown(id="scatter-y-dropdown", options=dropdown_options, value="TPSA", style=DROPDOWN_STYLE)
                                                                        ]),
                                                                        html.Div(style={"flex": 1}, children=[
                                                                            dcc.Dropdown(id="scatter-color-dropdown", options=[{'label': 'None', 'value': ''}] + dropdown_options, value='', style=DROPDOWN_STYLE)
                                                                        ]),
                                                                    ]
                                                                ),
                                                                dcc.Graph(id="scatter-graph", config={"displayModeBar": True}, style={"flex": 1})
                                                            ]
                                                        )
                                                    ]
                                                ),
                                                PanelResizeHandle(html.Div(style={"width": "3px", "cursor": "col-resize", "backgroundColor": "#ccc"})),
                                                Panel(
                                                    defaultSizePercentage=50,
                                                    children=[
                                                        html.Div(
                                                            id="tpsa-hist-container",
                                                            style={
                                                                "height": "100%",
                                                                "border": "1px solid #e5e7eb",
                                                                "borderRadius": "8px",
                                                                "padding": "10px",
                                                                "display": "flex",
                                                                "flexDirection": "column",
                                                                "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"
                                                            },
                                                            children=[
                                                                html.Div(
                                                                    style=DROPDOWN_CONTAINER_STYLE,
                                                                    children=[
                                                                        html.Div(style={"flex": 1}, children=[
                                                                            dcc.Dropdown(id="tpsa-hist-col-dropdown", options=dropdown_options, value="TPSA", style=DROPDOWN_STYLE)
                                                                        ]),
                                                                        html.Div(style={"flex": 1}, children=[
                                                                            dcc.Dropdown(id="tpsa-hist-color-dropdown", options=color_options, value="green", style=DROPDOWN_STYLE)
                                                                        ]),
                                                                        html.Div(style={"flex": 1}, children=[
                                                                            dcc.Input(
                                                                                id="hist-nbins-input",
                                                                                type="number",
                                                                                min=1, max=200, step=1,
                                                                                value=40,
                                                                                style={"width": "100%", "height": "36px", "fontSize": "14px", "borderRadius": "8px", "border": "1px solid #d0d0d0"}
                                                                            )
                                                                        ])
                                                                    ]
                                                                ),
                                                                dcc.Graph(id="tpsa-hist-graph", config={"displayModeBar": True}, style={"flex": 1})
                                                            ]
                                                        )
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                # ---------- Таблица ----------
                                PanelResizeHandle(html.Div(style={"height": "3px", "cursor": "row-resize", "backgroundColor": "#ccc"})),
                                Panel(
                                    defaultSizePercentage=60,
                                    children=[
                                        html.Div(
                                            style={"height": "100%", "display": "flex", "flexDirection": "column"},
                                            children=[
                                                html.Div(
                                                    style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "8px", "flexWrap": "wrap"},
                                                    children=[
                                                        dcc.Upload(
                                                            id='upload-data',
                                                            children=dbc.Button("Upload CSV", color="secondary", size="sm"),
                                                            multiple=False
                                                        ),
                                                        html.Div(id="upload-status", children="Using default CSV", style={"fontSize": "12px", "color": "#555"}),
                                                        dbc.DropdownMenu(
                                                            label="Select columns",
                                                            children=[
                                                                dbc.Checklist(
                                                                    id="select-all-toggle",
                                                                    options=[{'label': 'Select / Unselect All', 'value': 'toggle'}],
                                                                    value=["toggle"],
                                                                    inline=False,
                                                                    style={"marginBottom": "4px"}
                                                                ),
                                                                dbc.Checklist(
                                                                    id="column-selector",
                                                                    options=[{'label': col['name'], 'value': col['id']} for col in table_columns],
                                                                    value=[col['id'] for col in table_columns],
                                                                    inline=False,
                                                                    style={"maxHeight": "300px", "overflowY": "auto"}
                                                                )
                                                            ],
                                                            color="dark",
                                                            size="sm"
                                                        ),
                                                        dbc.DropdownMenu(
                                                            label="Calculate properties",
                                                            color="primary",
                                                            size="sm",
                                                            children=[
                                                                html.Div(
                                                                    style={"maxHeight": "340px", "overflowY": "auto"},
                                                                    children=[
                                                                        html.Div("Lipophilicity / polarity", style={"fontWeight": "bold", "padding": "0 8px"}),
                                                                        dbc.Checklist(
                                                                            id="extra-props-lipo",
                                                                            options=[{'label': name, 'value': name} for name in extra_property_groups["Lipophilicity / polarity"]],
                                                                            value=[],
                                                                            inline=False,
                                                                            style={"padding": "0 8px"}
                                                                        ),
                                                                        html.Div("H-bond properties", style={"fontWeight": "bold", "padding": "8px 8px 0 8px"}),
                                                                        dbc.Checklist(
                                                                            id="extra-props-hbond",
                                                                            options=[{'label': name, 'value': name} for name in extra_property_groups["H-bond properties"]],
                                                                            value=[],
                                                                            inline=False,
                                                                            style={"padding": "0 8px"}
                                                                        ),
                                                                        html.Div("Size and composition", style={"fontWeight": "bold", "padding": "8px 8px 0 8px"}),
                                                                        dbc.Checklist(
                                                                            id="extra-props-size",
                                                                            options=[{'label': name, 'value': name} for name in extra_property_groups["Size and composition"]],
                                                                            value=[],
                                                                            inline=False,
                                                                            style={"padding": "0 8px"}
                                                                        ),
                                                                        html.Div("Rings / flexibility", style={"fontWeight": "bold", "padding": "8px 8px 0 8px"}),
                                                                        dbc.Checklist(
                                                                            id="extra-props-rings",
                                                                            options=[{'label': name, 'value': name} for name in extra_property_groups["Rings / flexibility"]],
                                                                            value=[],
                                                                            inline=False,
                                                                            style={"padding": "0 8px"}
                                                                        ),
                                                                        html.Div("Drug-likeness", style={"fontWeight": "bold", "padding": "8px 8px 0 8px"}),
                                                                        dbc.Checklist(
                                                                            id="extra-props-drug",
                                                                            options=[{'label': name, 'value': name} for name in extra_property_groups["Drug-likeness"]],
                                                                            value=[],
                                                                            inline=False,
                                                                            style={"padding": "0 8px"}
                                                                        ),
                                                                        html.Div("Topology / complexity", style={"fontWeight": "bold", "padding": "8px 8px 0 8px"}),
                                                                        dbc.Checklist(
                                                                            id="extra-props-topology",
                                                                            options=[{'label': name, 'value': name} for name in extra_property_groups["Topology / complexity"]],
                                                                            value=[],
                                                                            inline=False,
                                                                            style={"padding": "0 8px 8px 8px"}
                                                                        ),
                                                                    ]
                                                                ),
                                                                html.Hr(style={"margin": "8px 0"}),
                                                                dbc.Button(
                                                                    "Calculate selected",
                                                                    id="calculate-extra-props-btn",
                                                                    color="success",
                                                                    size="sm",
                                                                    style={"width": "100%"}
                                                                )
                                                            ]
                                                        ),
                                                        html.Div(
                                                            style=SLIDER_CONTAINER_STYLE,
                                                            children=[
                                                                dcc.Dropdown(
                                                                    id="slider-mode-dropdown",
                                                                    options=[{"label": "By row count", "value": "rows"}] + dropdown_options,
                                                                    value="rows",
                                                                    style={"width": "160px", "fontSize": "12px"}
                                                                ),
                                                                dcc.Slider(
                                                                    id="table-row-slider",
                                                                    min=0,
                                                                    max=100,
                                                                    step=1,
                                                                    value=10,
                                                                    marks=None,
                                                                    tooltip={"placement": "left", "always_visible": False},
                                                                    updatemode="mouseup",
                                                                    className="dash-slider",
                                                                    vertical=False,
                                                                    verticalHeight=200,
                                                                ),
                                                                html.Div(id="slider-percentage-label", style={"width": "50px", "textAlign": "center", "fontSize": "12px"}),
                                                                html.Div(id="slider-count-label", style={"fontSize": "12px", "marginLeft": "8px"})
                                                            ]
                                                        )
                                                    ]
                                                ),
                                                html.Div(
                                                    style={"padding": "2px 2px", "fontSize": "12px", "color": "#555"},
                                                    children=[
                                                        html.Span("Circle = select row for 2D view, square(S) = copy row to Selected Molecules."),
                                                        html.Span(id="copy-row-status", style={"marginLeft": "8px"})
                                                    ]
                                                ),
                                                dash_table.DataTable(
                                                    id="molecules-table",
                                                    columns=[col for col in table_columns],
                                                    data=with_add_marker(df.head(10).to_dict("records")),
                                                    page_size=10,
                                                    filter_action="native",
                                                    sort_action="native",
                                                    sort_mode="multi",
                                                    row_selectable="single",
                                                    selected_rows=[0],
                                                    style_table={"overflowX": "auto", "overflowY": "hidden"},
                                                    fill_width=False,
                                                    style_header={
                                                        'backgroundColor': '#f8f9fa',
                                                        'fontWeight': 'bold',
                                                        'textAlign': 'left',
                                                        'padding': '8px 16px',
                                                        'fontSize': '16px',
                                                        'border': '1px solid #dee2e6',
                                                        'cursor': 'pointer'
                                                    },
                                                    style_cell={
                                                        'textAlign': 'left',
                                                        'padding': '6px',
                                                        'fontSize': '12px',
                                                        'width': 'auto',
                                                        'minWidth': '40px',
                                                        'maxWidth': '180px',
                                                        'whiteSpace': 'nowrap',
                                                        'overflow': 'hidden',
                                                        'textOverflow': 'ellipsis',
                                                        'border': '1px solid #dee2e6',
                                                        'lineHeight': '20px'
                                                    },
                                                    style_data_conditional=[
                                                        {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'},
                                                        {
                                                            'if': {'column_id': '__add__'},
                                                            'textAlign': 'center',
                                                            'fontSize': '18px',
                                                            'padding': '0px',
                                                        },
                                                    ],
                                                    style_cell_conditional=[
                                                        {
                                                            'if': {'column_id': '__add__'},
                                                            'width': '42px',
                                                            'minWidth': '42px',
                                                            'maxWidth': '42px',
                                                            'textAlign': 'center',
                                                        },
                                                        {'if': {'column_id': 'ID'}, 'minWidth': '55px', 'maxWidth': '80px', 'width': '70px'}
                                                    ],
                                                    virtualization=False
                                                ),
                                                html.Div(
                                                    id="table-status-label",
                                                    style={
                                                        "display": "flex",
                                                        "justifyContent": "flex-start",
                                                        "gap": "20px",
                                                        "padding": "6px 10px",
                                                        "fontSize": "18px",
                                                        "color": "#555",
                                                        "backgroundColor": "#f1f3f5",
                                                        "borderTop": "1px solid #dee2e6",
                                                        "borderRadius": "0 0 8px 8px",
                                                    },
                                                    children=[
                                                        html.Div(id="table-rows-info"),
                                                        html.Div(id="table-selected-info")
                                                    ]
                                                )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                PanelResizeHandle(html.Div(style={"width": "3px", "cursor": "col-resize", "backgroundColor": "#ccc"})),
                Panel(
                    defaultSizePercentage=30,
                    minSizePixels=300,
                    style={"padding": "8px"},
                    children=[
                        PanelGroup(
                            direction="vertical",
                            children=[
                                Panel(defaultSizePercentage=50, children=[html.Div(id="img-container", style={
                                    "height": "100%",
                                    "border": "1px solid #e5e7eb",
                                    "borderRadius": "8px",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"
                                })]),
                                PanelResizeHandle(html.Div(style={"height": "3px", "cursor": "row-resize", "backgroundColor": "#ccc"})),
                                Panel(defaultSizePercentage=50, children=[
                                    html.Div(
                                        id="hist-container",
                                        style={
                                            "height": "100%",
                                            "border": "1px solid #e5e7eb",
                                            "borderRadius": "8px",
                                            "padding": "10px",
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "boxShadow": "0 2px 6px rgba(0,0,0,0.08)"
                                        },
                                        children=[
                                            html.Div(
                                                style=DROPDOWN_CONTAINER_STYLE,
                                                children=[
                                                    html.Div(style={"flex": 1}, children=[
                                                        dcc.Dropdown(id="hist-col-dropdown", options=dropdown_options, value="MolWt", style=DROPDOWN_STYLE)
                                                    ]),
                                                    html.Div(style={"flex": 1}, children=[
                                                        dcc.Dropdown(id="hist-color-dropdown", options=color_options, value="blue", style=DROPDOWN_STYLE)
                                                    ]),
                                                    html.Div(style={"flex": 1}, children=[
                                                        dcc.Input(
                                                            id="hist-nbins-input-2",
                                                            type="number",
                                                            min=1, max=200, step=1,
                                                            value=40,
                                                            style={"width": "100%", "height": "36px", "fontSize": "14px", "borderRadius": "8px", "border": "1px solid #d0d0d0"}
                                                        )
                                                    ]),
                                                    dbc.Button("Compute KDE", id="compute-kde-btn", color="primary", size="sm",
                                                               style={"height": "36px", "alignSelf": "center", "marginLeft": "8px"}),
                                                ]
                                            ),
                                            dcc.Graph(id="hist-graph", config={"displayModeBar": True}, style={"flex": 1})
                                        ]
                                    )
                                ])
                            ]
                        )
                    ]
                )
            ]
            )]
        ),
        html.Div(
            id="selected-page-container",
            style={"display": "none", "height": "calc(100vh - 56px)", "padding": "8px"},
            children=[
                dcc.Download(id="selected-download"),
                html.Div(style={"height": "100%", "display": "flex", "gap": "10px"}, children=[
                    html.Div(style={"flex": "0 0 80%", "maxWidth": "80%", "display": "flex", "flexDirection": "column", "height": "100%"}, children=[
                        html.Div(
                            style={"display": "flex", "gap": "8px", "marginBottom": "8px", "alignItems": "center"},
                            children=[
                                dbc.DropdownMenu(
                                    label="Select columns",
                                    color="primary",
                                    size="sm",
                                    children=[
                                        dbc.Checklist(
                                            id="selected-select-all-toggle",
                                            options=[{'label': 'Select / Unselect All', 'value': 'toggle'}],
                                            value=["toggle"],
                                            inline=False,
                                            style={"padding": "6px"}
                                        ),
                                        dbc.Checklist(
                                            id="selected-column-selector",
                                            options=[],
                                            value=[],
                                            inline=False,
                                            style={"maxHeight": "240px", "overflowY": "auto", "padding": "6px"}
                                        )
                                    ]
                                ),
                                dbc.Button("Delete selected row", id="delete-selected-row-btn", color="danger", size="sm"),
                                dbc.Button("Clear All", id="clear-selected-btn", color="warning", size="sm"),
                                dbc.Button("Remove SMILES Duplicates", id="dedup-selected-btn", color="dark", size="sm"),
                                dbc.Button("Round numeric values", id="round-selected-btn", color="secondary", size="sm"),
                                dcc.Dropdown(
                                    id="round-digits-dropdown",
                                    options=[{"label": str(i), "value": i} for i in range(0, 7)],
                                    value=2,
                                    clearable=False,
                                    style={"width": "100px", "fontSize": "12px"}
                                ),
                                html.Span(id="selected-actions-status", style={"fontSize": "12px", "color": "#555"}),
                                html.Div(style={"marginLeft": "auto", "display": "flex", "alignItems": "center"}, children=[
                                    dcc.Upload(
                                        id='upload-selected-data',
                                        children=dbc.Button("Upload CSV", color="info", size="sm", style={"marginRight": "8px"}),
                                        multiple=False,
                                        style={"display": "inline-block"}
                                    ),
                                    dbc.Button("Download Selected CSV", id="download-selected-btn", color="success", size="sm"),
                                ]),
                            ]
                        ),
                        dash_table.DataTable(
                            id="selected-molecules-table",
                            columns=[col for col in table_columns[:4]],
                            data=[],
                            page_size=15,
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            row_selectable="single",
                            cell_selectable=True,
                            selected_rows=[],
                            style_table={"width": "100%", "height": "100%", "maxHeight": "100%", "overflowY": "auto", "overflowX": "auto"},
                            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                            style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '12px'},
                            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}],
                            virtualization=False,
                            markdown_options={"html": True}
                        )
                    ]),
                    html.Div(style={"flex": "0 0 20%", "maxWidth": "20%", "border": "1px solid #e5e7eb", "borderRadius": "8px", "display": "flex",
                                    "justifyContent": "center", "alignItems": "center", "backgroundColor": "#fff"}, children=[
                        html.Div(id="selected-img-container")
                    ])
                ])
            ]
        )
    ]
)


# ========== Колбэки ==========

@app.callback(
    Output("upload-status", "children"),
    Output("molecules-table", "data", allow_duplicate=True),
    Output("molecules-table", "columns", allow_duplicate=True),
    Output("column-selector", "options", allow_duplicate=True),
    Output("column-selector", "value", allow_duplicate=True),
    Output("scatter-x-dropdown", "options", allow_duplicate=True),
    Output("scatter-y-dropdown", "options", allow_duplicate=True),
    Output("tpsa-hist-col-dropdown", "options", allow_duplicate=True),
    Output("hist-col-dropdown", "options", allow_duplicate=True),
    Output("scatter-color-dropdown", "options", allow_duplicate=True),
    Output("slider-mode-dropdown", "options", allow_duplicate=True),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("table-row-slider", "value"),
    State("slider-mode-dropdown", "value"),
    prevent_initial_call=True
)
def upload_csv(contents, filename, slider_value, slider_mode):
    global df, numeric_cols, dropdown_options, table_columns
    if not contents:
        return no_update

    try:
        df = parse_uploaded_csv(contents)
        numeric_cols, dropdown_options, table_columns = build_table_columns(df)
        column_options = [{'label': col['name'], 'value': col['id']} for col in table_columns]
        selected_columns = [col['id'] for col in table_columns]
        scatter_color_options = [{'label': 'None', 'value': ''}] + dropdown_options
        slider_options = [{"label": "By row count", "value": "rows"}] + dropdown_options

        if slider_mode == "rows":
            n_rows = max(1, int(len(df) * (slider_value or 100) / 100))
            table_data = with_add_marker(df.head(n_rows).to_dict("records"))
        elif slider_mode in numeric_cols:
            tdf = df.sort_values(slider_mode, ascending=False)
            n_rows = max(1, int(len(df) * (slider_value or 100) / 100))
            table_data = with_add_marker(tdf.head(n_rows).to_dict("records"))
        else:
            table_data = with_add_marker(df.to_dict("records"))

        return (
            f"✅ Uploaded file: {filename}",
            table_data,
            [col for col in table_columns],
            column_options,
            selected_columns,
            dropdown_options,
            dropdown_options,
            dropdown_options,
            dropdown_options,
            scatter_color_options,
            slider_options,
        )
    except Exception as e:
        return f"❌ CSV upload error: {e}", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update


# ---------- 1. Toggle Select / Unselect All ----------
@app.callback(
    Output("column-selector", "value"),
    Output("select-all-toggle", "value"),
    Input("select-all-toggle", "value"),
    State("column-selector", "value"),
    State("column-selector", "options")
)
def toggle_select_all(toggle_val, current_val, options):
    all_vals = [opt['value'] for opt in options]

    if 'toggle' in toggle_val:
        return all_vals, ['toggle']
    elif current_val == all_vals:
        return [], []
    else:
        return no_update, toggle_val


# ---------- Новый колбэк: Расчёт дополнительных свойств ----------
@app.callback(
    Output("column-selector", "options"),
    Output("extra-props-lipo", "value"),
    Output("extra-props-hbond", "value"),
    Output("extra-props-size", "value"),
    Output("extra-props-rings", "value"),
    Output("extra-props-drug", "value"),
    Output("extra-props-topology", "value"),
    Output("scatter-x-dropdown", "options"),
    Output("scatter-y-dropdown", "options"),
    Output("tpsa-hist-col-dropdown", "options"),
    Output("hist-col-dropdown", "options"),
    Output("scatter-color-dropdown", "options"),
    Output("slider-mode-dropdown", "options"),
    Input("calculate-extra-props-btn", "n_clicks"),
    State("extra-props-lipo", "value"),
    State("extra-props-hbond", "value"),
    State("extra-props-size", "value"),
    State("extra-props-rings", "value"),
    State("extra-props-drug", "value"),
    State("extra-props-topology", "value"),
    prevent_initial_call=True
)
def calculate_extra_properties(n_clicks, lipo_vals, hbond_vals, size_vals, rings_vals, drug_vals, topology_vals):
    selected_props = (lipo_vals or []) + (hbond_vals or []) + (size_vals or []) + (rings_vals or []) + (drug_vals or []) + (topology_vals or [])
    selected_props = list(dict.fromkeys(selected_props))
    if not n_clicks or not selected_props:
        return [no_update] * 13

    print(f"🔬 Рассчитываем {len(selected_props)} свойств: {selected_props}")

    for prop_name in selected_props:
        if prop_name not in df.columns:
            func = extra_properties[prop_name]
            values = []
            for smiles in df["SMILES"]:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    values.append(np.nan)
                else:
                    try:
                        values.append(func(mol))
                    except Exception:
                        values.append(np.nan)
            df[prop_name] = values
            print(f" ✓ Добавлена колонка: {prop_name}")

    global table_columns, numeric_cols, dropdown_options
    numeric_cols, dropdown_options, table_columns = build_table_columns(df)

    new_column_options = [{'label': col['name'], 'value': col['id']} for col in table_columns]
    all_column_ids = [col['id'] for col in table_columns]

    numeric_dropdown = dropdown_options
    color_dropdown = [{'label': 'None', 'value': ''}] + numeric_dropdown
    slider_dropdown = [{"label": "By row count", "value": "rows"}] + numeric_dropdown

    return (
        new_column_options,
        [],
        [],
        [],
        [],
        [],
        [],
        numeric_dropdown,
        numeric_dropdown,
        numeric_dropdown,
        numeric_dropdown,
        color_dropdown,
        slider_dropdown
    )


# ---------- 2. Синхронизация клика по scatter с таблицей ----------
@app.callback(
    Output("molecules-table", "selected_rows"),
    Output("molecules-table", "page_current"),
    Input("scatter-graph", "clickData"),
    Input("table-row-slider", "value"),
    Input("slider-mode-dropdown", "value"),
    State("molecules-table", "data")
)
def sync_table_selection(scatter_click, slider_value, slider_mode, table_data):
    if slider_mode == "rows":
        n_rows = max(1, int(len(df) * slider_value / 100))
        tdf = df.head(n_rows)
    elif slider_mode in numeric_cols:
        tdf = df.sort_values(slider_mode, ascending=False)
        n_rows = max(1, int(len(df) * slider_value / 100))
        tdf = tdf.head(n_rows)
    else:
        tdf = df.copy()

    if scatter_click and 'points' in scatter_click:
        clicked_id = str(scatter_click['points'][0]['customdata'][0])
        local_data = table_data or tdf.to_dict("records")
        for i, row in enumerate(local_data):
            if str(row.get("ID")) == clicked_id:
                return [i], i // 10
        return no_update, no_update

    return no_update, no_update


@app.callback(
    Output("main-page-container", "style"),
    Output("selected-page-container", "style"),
    Input("page-tabs", "value")
)
def switch_page(active_tab):
    if active_tab == "selected-page":
        return {"display": "block", "height": "0", "overflow": "hidden"}, {"display": "block", "height": "calc(100vh - 56px)", "padding": "8px"}
    return {"display": "block", "height": "calc(100vh - 56px)", "overflow": "visible"}, {"display": "none", "height": "calc(100vh - 56px)", "padding": "8px"}


@app.callback(
    Output("selected-molecules-store", "data"),
    Output("copy-row-status", "children"),
    Output("molecules-table", "data", allow_duplicate=True),
    Output("molecules-table", "active_cell", allow_duplicate=True),
    Input("molecules-table", "active_cell"),
    State("molecules-table", "data"),
    State("molecules-table", "derived_viewport_data"),
    State("selected-molecules-store", "data"),
    prevent_initial_call=True
)
def copy_selected_row(active_cell, table_data, viewport_data, selected_data):
    if not active_cell or active_cell.get("column_id") != "__add__" or not table_data:
        return selected_data, no_update, no_update, no_update
    selected_data = selected_data or []
    idx = active_cell.get("row")
    if idx is None or viewport_data is None or idx >= len(viewport_data):
        return selected_data, "Invalid row.", no_update, None
    row = {k: v for k, v in viewport_data[idx].items() if k != "__add__"}
    key_id = str(row.get("ID"))
    key_smiles = str(row.get("SMILES", ""))
    existing_index = next(
        (i for i, item in enumerate(selected_data)
         if str(item.get("ID")) == key_id and str(item.get("SMILES", "")) == key_smiles),
        None
    )
    updated_table_data = [dict(r) for r in (table_data or [])]
    global_idx = next(
        (i for i, r in enumerate(updated_table_data)
         if str(r.get("ID")) == key_id and str(r.get("SMILES", "")) == key_smiles),
        None
    )
    if existing_index is not None:
        selected_data.pop(existing_index)
        if global_idx is not None and global_idx < len(updated_table_data):
            updated_table_data[global_idx]["__add__"] = "⬜"
        return selected_data, f"Removed molecule ID {row.get('ID')} from Selected table.", updated_table_data, None

    selected_data.append(row)
    if global_idx is not None and global_idx < len(updated_table_data):
        updated_table_data[global_idx]["__add__"] = "✅"
    return selected_data, f"Added molecule ID {row.get('ID')} to Selected table.", updated_table_data, None


@app.callback(
    Output("selected-molecules-table", "data"),
    Output("selected-molecules-table", "columns"),
    Input("selected-molecules-store", "data"),
    Input("selected-column-selector", "value")
)
def refresh_selected_table(selected_data, selected_columns):
    selected_data = selected_data or []
    if not selected_data:
        cols = [col for col in table_columns if col['id'] in (selected_columns or [])]
        if not cols:
            cols = [col for col in table_columns[:4]]
        return [], cols

    enriched = []
    for row in selected_data:
        rr = dict(row)
        rr["2DMol"] = smiles_to_thumb_html(str(row.get("SMILES", "")))
        enriched.append(rr)

    # Сохраняем все колонки, которые когда-либо попали в selected_data (включая из других CSV)
    all_keys = []
    for row in enriched:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    selected_set = set(selected_columns or all_keys)
    ordered_keys = [k for k in all_keys if k in selected_set]
    cols = []
    for k in ordered_keys:
        if k == "2DMol":
            cols.append({"name": k, "id": k, "presentation": "markdown"})
        else:
            cols.append({"name": k, "id": k})
    return enriched, cols


@app.callback(
    Output("selected-column-selector", "options"),
    Output("selected-column-selector", "value"),
    Input("selected-molecules-store", "data"),
    State("selected-column-selector", "value")
)
def sync_selected_column_selector(selected_data, current_value):
    selected_data = selected_data or []
    keys = []
    for row in selected_data:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    if selected_data and "2DMol" not in keys:
        keys.append("2DMol")
    options = [{"label": k, "value": k} for k in keys]
    if not keys:
        return [], []
    if not current_value:
        return options, keys
    valid = [v for v in current_value if v in keys]
    return options, (valid if valid else keys)


@app.callback(
    Output("selected-column-selector", "value", allow_duplicate=True),
    Output("selected-select-all-toggle", "value"),
    Input("selected-select-all-toggle", "value"),
    State("selected-column-selector", "options"),
    State("selected-column-selector", "value"),
    prevent_initial_call=True
)
def toggle_selected_columns(toggle_val, options, current_vals):
    all_vals = [opt['value'] for opt in (options or [])]
    if 'toggle' in (toggle_val or []):
        if set(current_vals or []) == set(all_vals):
            return [], []
        return all_vals, ['toggle']
    return no_update, no_update


@app.callback(
    Output("selected-molecules-store", "data", allow_duplicate=True),
    Output("selected-actions-status", "children", allow_duplicate=True),
    Input("clear-selected-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_all_selected(n_clicks):
    if not n_clicks:
        return no_update, no_update
    return [], "Cleared all selected molecules."


@app.callback(
    Output("selected-molecules-store", "data", allow_duplicate=True),
    Output("selected-actions-status", "children", allow_duplicate=True),
    Input("dedup-selected-btn", "n_clicks"),
    State("selected-molecules-store", "data"),
    prevent_initial_call=True
)
def dedup_selected(n_clicks, selected_data):
    if not n_clicks:
        return no_update, no_update
    selected_data = selected_data or []
    seen = set()
    deduped = []
    for row in selected_data:
        raw_smiles = str(row.get("SMILES", "")).strip().replace(" ", "")
        mol = Chem.MolFromSmiles(raw_smiles) if raw_smiles else None
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) if mol is not None else raw_smiles
        raw_key = raw_smiles.lower()
        key = canonical_smiles.lower() if isinstance(canonical_smiles, str) else raw_key
        if key in seen or raw_key in seen:
            continue
        seen.add(key)
        seen.add(raw_key)
        deduped.append(row)
    removed = len(selected_data) - len(deduped)
    return deduped, f"Removed {removed} SMILES duplicates (canonicalized with RDKit)."


@app.callback(
    Output("selected-molecules-store", "data", allow_duplicate=True),
    Output("selected-actions-status", "children"),
    Input("delete-selected-row-btn", "n_clicks"),
    State("selected-molecules-table", "selected_rows"),
    State("selected-molecules-store", "data"),
    prevent_initial_call=True
)
def delete_selected_row(n_clicks, selected_rows, selected_data):
    selected_data = list(selected_data or [])
    if not n_clicks or not selected_rows:
        return selected_data, "No row selected for deletion."
    idx = selected_rows[0]
    if 0 <= idx < len(selected_data):
        removed = selected_data.pop(idx)
        return selected_data, f"Deleted ID {removed.get('ID')}."
    return selected_data, "Invalid row index."


@app.callback(
    Output("selected-download", "data"),
    Input("download-selected-btn", "n_clicks"),
    State("selected-molecules-store", "data"),
    State("selected-column-selector", "value"),
    prevent_initial_call=True
)
def download_selected_csv(n_clicks, selected_data, selected_columns):
    if not n_clicks:
        return no_update
    selected_data = selected_data or []
    if not selected_data:
        return no_update
    sdf = pd.DataFrame(selected_data)
    if selected_columns:
        visible_cols = [c for c in selected_columns if c in sdf.columns]
        if visible_cols:
            sdf = sdf[visible_cols]
    return dcc.send_data_frame(sdf.to_csv, "selected_molecules.csv", index=False)


@app.callback(
    Output("selected-molecules-store", "data", allow_duplicate=True),
    Output("selected-actions-status", "children", allow_duplicate=True),
    Input("round-selected-btn", "n_clicks"),
    State("round-digits-dropdown", "value"),
    State("selected-molecules-store", "data"),
    prevent_initial_call=True
)
def round_selected_values(n_clicks, digits, selected_data):
    if not n_clicks:
        return no_update, no_update
    selected_data = list(selected_data or [])
    if not selected_data:
        return selected_data, "Selected table is empty."
    digits = int(digits or 0)
    rounded = []
    for row in selected_data:
        nr = {}
        for k, v in row.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                nr[k] = round(float(v), digits)
            else:
                nr[k] = v
        rounded.append(nr)
    return rounded, f"Rounded numeric values to {digits} decimals."


@app.callback(
    Output("selected-molecules-store", "data", allow_duplicate=True),
    Output("selected-actions-status", "children", allow_duplicate=True),
    Input("upload-selected-data", "contents"),
    State("upload-selected-data", "filename"),
    prevent_initial_call=True
)
def upload_selected_csv(contents, filename):
    if not contents:
        return no_update, no_update
    try:
        up_df = parse_uploaded_csv(contents)
        rows = up_df.to_dict("records")
        return rows, f"Loaded selected table from: {filename}"
    except Exception as e:
        return no_update, f"Selected upload error: {e}"


@app.callback(
    Output("selected-img-container", "children"),
    Input("selected-molecules-table", "selected_rows"),
    Input("selected-molecules-table", "data")
)
def update_selected_image(selected_rows, selected_data):
    if not selected_data:
        return html.Div("No molecules selected yet.")
    idx = selected_rows[0] if selected_rows else 0
    idx = min(idx, len(selected_data) - 1)
    smiles = selected_data[idx].get("SMILES", "")
    img_b64 = smiles_to_base64(smiles)
    return html.Img(src=img_b64, style={"maxWidth": "100%", "maxHeight": "100%", "objectFit": "contain"})


# ---------- 3. Обновление всех графиков и изображения ----------
@app.callback(
    Output("img-container", "children"),
    Output("hist-graph", "figure"),
    Output("tpsa-hist-graph", "figure"),
    Output("scatter-graph", "figure"),
    Input("molecules-table", "selected_rows"),
    Input("molecules-table", "derived_viewport_selected_rows"),
    Input("hist-col-dropdown", "value"),
    Input("scatter-x-dropdown", "value"),
    Input("scatter-y-dropdown", "value"),
    Input("tpsa-hist-col-dropdown", "value"),
    Input("hist-color-dropdown", "value"),
    Input("tpsa-hist-color-dropdown", "value"),
    Input("scatter-color-dropdown", "value"),
    Input("table-row-slider", "value"),
    Input("slider-mode-dropdown", "value"),
    Input("hist-nbins-input", "value"),
    Input("hist-nbins-input-2", "value"),
    Input("compute-kde-btn", "n_clicks"),
    Input("molecules-table", "filter_query"),
    Input("molecules-table", "data"),
    State("molecules-table", "derived_virtual_data"),
    State("molecules-table", "derived_viewport_data"),
)
def update_all(selected_rows, viewport_selected_rows, hist_col, scatter_x, scatter_y, tpsa_hist_col,
               hist_color, tpsa_hist_color, scatter_color_col, slider_value, slider_mode,
               hist_nbins, hist_nbins2, compute_kde_clicks, filter_query, table_data, virtual_data, viewport_data):
    try:
        if filter_query and virtual_data is not None:
            tdf = pd.DataFrame(virtual_data).copy()
            if "__add__" in tdf.columns:
                tdf = tdf.drop(columns=["__add__"])
            if tdf.empty:
                tdf = df.head(1).copy()
        elif table_data is not None:
            tdf = pd.DataFrame(table_data).copy()
            if "__add__" in tdf.columns:
                tdf = tdf.drop(columns=["__add__"])
            if tdf.empty:
                tdf = df.head(1).copy()
        elif slider_mode == "rows":
            n_rows = max(1, int(len(df) * slider_value / 100))
            tdf = df.head(n_rows).copy()
        elif slider_mode in numeric_cols:
            tdf = df.sort_values(slider_mode, ascending=False)
            n_rows = max(1, int(len(df) * slider_value / 100))
            tdf = tdf.head(n_rows)
        else:
            tdf = df.copy()

        compute_kde = bool(compute_kde_clicks and compute_kde_clicks > 0)
        if viewport_data is not None:
            vdf = pd.DataFrame(viewport_data).copy()
            if "__add__" in vdf.columns:
                vdf = vdf.drop(columns=["__add__"])
            if vdf.empty:
                vdf = tdf.copy()
        else:
            vdf = tdf.copy()

        selected_idx_list = viewport_selected_rows if viewport_selected_rows else (selected_rows or [])
        img_components = []
        if selected_idx_list:
            for sel_idx in selected_idx_list:
                if sel_idx < len(vdf):
                    row = vdf.iloc[sel_idx]
                    smiles = str(row.get("SMILES", ""))
                    img_b64 = smiles_to_base64(smiles)
                    img_components.append(
                        html.Img(src=img_b64, style={"maxWidth": "100%", "maxHeight": "100%", "objectFit": "contain", "margin": "4px"})
                    )
        if not img_components:
            row = vdf.iloc[0] if len(vdf) > 0 else df.iloc[0]
            img_b64 = smiles_to_base64(row.get("SMILES", ""))
            img_components.append(
                html.Img(src=img_b64, style={"maxWidth": "100%", "maxHeight": "100%", "objectFit": "contain"})
            )

        img_component = html.Div(
            img_components,
            style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center", "overflowY": "auto", "height": "100%"}
        )


        safe_x = scatter_x or "MolWt"
        safe_y = scatter_y or "TPSA"
        s_tdf = tdf.sample(SCATTER_SAMPLE, random_state=1) if len(tdf) > SCATTER_SAMPLE else tdf
        color_arg = scatter_color_col if scatter_color_col in s_tdf.columns else None

        fig_sc = px.scatter(
            s_tdf, x=safe_x, y=safe_y, color=color_arg,
            size_max=12, hover_data=["ID", safe_x, safe_y],
            custom_data=["ID"],
            title=f"{safe_x} vs {safe_y}"
        )
        point_size = max(3, min(10, 5000 / len(s_tdf)))
        fig_sc.update_traces(marker=dict(size=point_size, line=dict(width=0.5, color="DarkSlateGrey"), opacity=0.6))

        for sel_idx in (selected_idx_list or []):
            if sel_idx < len(vdf):
                sel_point = vdf.iloc[sel_idx]
                sel_id = str(sel_point["ID"])
                if sel_id in s_tdf["ID"].astype(str).values:
                    fig_sc.add_trace(go.Scatter(
                        x=[sel_point[safe_x]],
                        y=[sel_point[safe_y]],
                        mode="markers",
                        marker=dict(size=16, symbol="asterisk", color="black", line=dict(width=2, color="black")),
                        customdata=[[sel_id]],
                        hovertemplate=f"<b>Selected ID: {sel_id}</b><br>{safe_x}: %{{x}}<br>{safe_y}: %{{y}}<extra></extra>",
                        showlegend=False
                    ))

        fig_sc.update_layout(
            margin=dict(l=45, r=10, t=30, b=60),
            plot_bgcolor='rgba(250,250,250,1)',
            paper_bgcolor='rgba(250,250,250,1)'
        )

        fig_hist = make_pretty_hist(tdf, hist_col or "MolWt", hist_color or "#1f77b4",
                                    f"Distribution of {hist_col or 'MolWt'}", nbinsx=hist_nbins2 or 40, compute_kde=compute_kde)
        fig_tpsa_hist = make_pretty_hist(tdf, tpsa_hist_col or "TPSA", tpsa_hist_color or "#2ca02c",
                                         f"Distribution of {tpsa_hist_col or 'TPSA'}", nbinsx=hist_nbins or 40, compute_kde=compute_kde)

        return img_component, fig_hist, fig_tpsa_hist, fig_sc

    except Exception:
        blank = go.Figure()
        blank.add_annotation(text="Ошибка в графиках (см. терминал)", showarrow=False, font=dict(size=18, color="red"))
        return html.Div("Ошибка"), blank, blank, blank


# ---------- 4. Обновление колонок таблицы ----------
@app.callback(
    Output("molecules-table", "columns", allow_duplicate=True),
    Input("column-selector", "value"),
    prevent_initial_call=True
)
def update_table_columns(selected_columns):
    if not selected_columns:
        return []
    return [col for col in table_columns if col['id'] in selected_columns]


# ---------- 5. Лейбл ползунка ----------
@app.callback(Output("slider-percentage-label", "children"), Input("table-row-slider", "value"))
def update_slider_label(val):
    return f"{val}%"


# ---------- 6. Данные таблицы ----------
@app.callback(
    Output("molecules-table", "data", allow_duplicate=True),
    Input("table-row-slider", "value"),
    Input("slider-mode-dropdown", "value"),
    Input("calculate-extra-props-btn", "n_clicks"),
    State("selected-molecules-store", "data"),
    prevent_initial_call=True
)
def update_table_rows(slider_value, slider_mode, n_clicks, selected_data):
    if slider_mode == "rows":
        n_rows = max(1, int(len(df) * slider_value / 100))
        return apply_selected_markers(df.head(n_rows).to_dict("records"), selected_data)
    elif slider_mode in numeric_cols:
        tdf = df.sort_values(slider_mode, ascending=False)
        n_rows = max(1, int(len(df) * slider_value / 100))
        return apply_selected_markers(tdf.head(n_rows).to_dict("records"), selected_data)
    else:
        return apply_selected_markers(df.to_dict("records"), selected_data)


@app.callback(
    Output("slider-count-label", "children"),
    Input("table-row-slider", "value"),
    Input("slider-mode-dropdown", "value"),
    Input("upload-status", "children")
)
def update_slider_count(slider_value, slider_mode, _upload_status):
    if slider_mode == "rows":
        n_rows = max(1, int(len(df) * slider_value / 100))
    elif slider_mode in numeric_cols:
        tdf = df.sort_values(slider_mode, ascending=False)
        n_rows = max(1, int(len(df) * slider_value / 100))
    else:
        n_rows = len(df)
    return f"{n_rows} molecules"


@app.callback(
    Output("table-rows-info", "children"),
    Output("table-selected-info", "children"),
    Input("table-row-slider", "value"),
    Input("slider-mode-dropdown", "value"),
    Input("molecules-table", "selected_rows"),
    Input("upload-status", "children")
)
def update_table_status(slider_value, slider_mode, selected_rows, _upload_status):
    if slider_mode == "rows":
        n_rows = max(1, int(len(df) * slider_value / 100))
    elif slider_mode in numeric_cols:
        tdf = df.sort_values(slider_mode, ascending=False)
        n_rows = max(1, int(len(df) * slider_value / 100))
    else:
        n_rows = len(df)

    sel_text = f"Selected row: {selected_rows[0]+1}" if selected_rows else "No selection"

    return f"Showing {n_rows} of {len(df)} molecules", sel_text


# ========== Запуск ==========
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8054)

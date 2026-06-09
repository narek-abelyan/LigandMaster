import base64
import io
from urllib.parse import quote
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dash_table, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# ========== Config ==========
CSV_PATH = "molecules_non_radical.csv"
DEFAULT_COLUMNS = ["ID", "SMILES", "MolWt", "TPSA"]


def prepare_df(new_df: pd.DataFrame) -> pd.DataFrame:
    if "ID" in new_df.columns:
        if pd.api.types.is_numeric_dtype(new_df["ID"]):
            new_df["ID"] = new_df["ID"].astype(int).astype(str)
        else:
            new_df["ID"] = new_df["ID"].astype(str).str.strip()

    required = {"ID", "SMILES"}
    missing_basic = required - set(new_df.columns)
    if missing_basic:
        raise ValueError(f"В CSV отсутствуют необходимые столбцы: {missing_basic}")

    if "MolWt" not in new_df.columns:
        new_df["MolWt"] = [Descriptors.MolWt(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else np.nan for s in new_df["SMILES"]]
    if "TPSA" not in new_df.columns:
        new_df["TPSA"] = [Descriptors.TPSA(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else np.nan for s in new_df["SMILES"]]
    return new_df


def parse_uploaded_csv(contents: str) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    text = decoded.decode("utf-8")
    for sep in [";", ","]:
        try:
            parsed = pd.read_csv(io.StringIO(text), sep=sep)
            if len(parsed.columns) > 1:
                return prepare_df(parsed)
        except Exception:
            continue
    return prepare_df(pd.read_csv(io.StringIO(text)))


def initial_df() -> pd.DataFrame:
    try:
        return prepare_df(pd.read_csv(CSV_PATH, sep=";"))
    except Exception:
        return prepare_df(pd.DataFrame({
            "ID": ["1", "2", "3"],
            "SMILES": ["CCO", "c1ccccc1", "CC(=O)O"],
        }))


def build_table_columns(local_df: pd.DataFrame):
    numeric_cols = local_df.select_dtypes(include="number").columns.tolist()
    table_columns = []
    for c in local_df.columns:
        if c in numeric_cols:
            table_columns.append({"name": c, "id": c, "type": "numeric", "format": {"specifier": ".3f"}})
        else:
            table_columns.append({"name": c, "id": c})
    return numeric_cols, table_columns


@dataclass
class FPConfig:
    method: str
    radius: int
    nbits: int


def make_fp(mol: Chem.Mol, cfg: FPConfig):
    if cfg.method == "Morgan":
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=cfg.radius, nBits=cfg.nbits)
    if cfg.method == "RDKit":
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=cfg.nbits)
        return gen.GetFingerprint(mol)
    if cfg.method == "MACCS":
        return MACCSkeys.GenMACCSKeys(mol)
    if cfg.method == "AtomPair":
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=cfg.nbits)
        return gen.GetFingerprint(mol)
    if cfg.method == "TopologicalTorsion":
        gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=cfg.nbits)
        return gen.GetFingerprint(mol)
    if cfg.method == "Pattern":
        return Chem.PatternFingerprint(mol, fpSize=cfg.nbits)
    if cfg.method == "Layered":
        return Chem.LayeredFingerprint(mol, fpSize=cfg.nbits)
    if cfg.method == "Avalon":
        try:
            from rdkit.Avalon import pyAvalonTools
            return pyAvalonTools.GetAvalonFP(mol, nBits=cfg.nbits)
        except Exception as exc:
            raise ValueError(f"Avalon fingerprint is not available in this RDKit build: {exc}")
    raise ValueError("Unknown fingerprint method")


def fp_numpy(fp):
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def sim_score(fp_a, fp_b, metric: str, alpha: float = 0.5, beta: float = 0.5):
    if metric == "Tanimoto":
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    if metric == "Dice":
        return DataStructs.DiceSimilarity(fp_a, fp_b)
    if metric == "Cosine":
        return DataStructs.CosineSimilarity(fp_a, fp_b)
    if metric == "Sokal":
        return DataStructs.SokalSimilarity(fp_a, fp_b)
    if metric == "Russel":
        return DataStructs.RusselSimilarity(fp_a, fp_b)
    if metric == "Tversky":
        return DataStructs.TverskySimilarity(fp_a, fp_b, alpha, beta)
    raise ValueError("Unknown similarity metric")


def page_main(df: pd.DataFrame):
    numeric_cols, table_columns = build_table_columns(df)
    return html.Div([
        html.H4("Main page"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="main-x", options=numeric_cols, value="MolWt"), md=4),
            dbc.Col(dcc.Dropdown(id="main-y", options=numeric_cols, value="TPSA"), md=4),
            dbc.Col(dcc.Slider(id="main-row-pct", min=5, max=100, step=5, value=40), md=4),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="main-scatter"), md=6),
            dbc.Col(dash_table.DataTable(
                id="main-table",
                columns=[c for c in table_columns if c["id"] in DEFAULT_COLUMNS],
                data=df.head(30).to_dict("records"),
                page_size=15,
                row_selectable="single",
                selected_rows=[0],
                filter_action="native",
                sort_action="native",
                style_table={"overflowX": "auto", "height": "65vh"},
            ), md=6),
        ]),
    ])


def page_similarity_controls(df: pd.DataFrame):
    _, table_columns = build_table_columns(df)
    fp_section_style = {
        "background": "#f3f6fb",
        "border": "1px solid #d8e2f0",
        "borderRadius": "12px",
        "padding": "12px 14px",
        "boxShadow": "0 4px 12px rgba(39, 74, 120, 0.07)",
        "marginBottom": "10px",
    }
    sim_section_style = {
        "background": "#eef3fa",
        "border": "1px solid #d4deec",
        "borderRadius": "12px",
        "padding": "12px 14px",
        "boxShadow": "0 4px 12px rgba(48, 82, 126, 0.07)",
        "marginBottom": "10px",
    }
    emb_section_style = {
        "background": "#edf2f9",
        "border": "1px solid #cfdaea",
        "borderRadius": "12px",
        "padding": "12px 14px",
        "boxShadow": "0 4px 12px rgba(54, 88, 130, 0.08)",
        "marginBottom": "10px",
    }
    cluster_section_style = {
        "background": "#e9eff8",
        "border": "1px solid #cad7e8",
        "borderRadius": "12px",
        "padding": "12px 14px",
        "boxShadow": "0 4px 12px rgba(47, 76, 113, 0.08)",
        "marginBottom": "10px",
    }
    section_title_style = {
        "fontWeight": "700",
        "fontSize": "16px",
        "marginBottom": "6px",
        "color": "#1f3b64",
    }
    sample_card_style = {
        "background": "#eef3fa",
        "border": "1px solid #d4deec",
        "borderRadius": "12px",
        "padding": "12px 14px",
        "boxShadow": "0 4px 12px rgba(48, 82, 126, 0.07)",
        "marginBottom": "10px",
    }
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Выбери метод для генерации fingerprint", style=section_title_style),
                    dcc.Dropdown(
                        id="fp-method",
                        options=["Morgan", "RDKit", "MACCS", "AtomPair", "TopologicalTorsion", "Pattern", "Layered", "Avalon"],
                        value="Morgan",
                        style={"width": "100%"},
                    ),
                    html.Div(id="fp-params-container", className="mt-2", style={"width": "100%"}),
                ], style=fp_section_style),
                html.Div([
                    html.Div("Выбор метода оценки схожости", style=section_title_style),
                    dcc.Dropdown(
                        id="sim-method",
                        options=["Tanimoto", "Dice", "Cosine", "Sokal", "Russel", "Tversky"],
                        value="Tanimoto",
                        style={"width": "100%"},
                    ),
                    html.Div(id="sim-params-container", className="mt-2", style={"width": "100%"}),
                ], style=sim_section_style),
                html.Div([
                    html.Div("Методы снижения размерности", style=section_title_style),
                    dcc.Dropdown(
                        id="embedding-method",
                        options=["Не сжимать данные", "PCA", "T-SNE", "UMAP"],
                        value="Не сжимать данные",
                        style={"width": "100%"},
                    ),
                    html.Div(id="embedding-params-container", className="mt-2", style={"width": "100%"}),
                ], style=emb_section_style),
                html.Div([
                    html.Div("Выбор алгоритма кластеризации", style=section_title_style),
                    dcc.Dropdown(
                        id="cluster-method",
                        options=["Butina", "Hierarchical", "HDBSCAN", "KMeans", "DBSCAN"],
                        value="Butina",
                        style={"width": "100%"},
                    ),
                    html.Div(id="cluster-params-container", className="mt-2", style={"width": "100%"}),
                ], style=cluster_section_style),
            ], md=6),
            dbc.Col([
                html.Div([
                    html.Div("Percent of table for clustering", style=section_title_style),
                    dcc.Slider(
                        id="sample-pct",
                        min=5,
                        max=100,
                        step=5,
                        value=100,
                        marks={i: f"{i}%" for i in range(5, 101, 5)},
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="mt-1",
                    ),
                ], style=sample_card_style),
                dbc.Card([
                    dbc.CardBody([
                        html.Div("Fingerprint method preview", style={"fontWeight": "600", "marginBottom": "8px"}),
                        html.Img(id="fp-method-image", style={
                            "width": "100%",
                            "maxWidth": "50vw",
                            "aspectRatio": "2 / 1",
                            "objectFit": "contain",
                            "border": "1px solid #d9d9d9",
                            "borderRadius": "8px",
                            "backgroundColor": "#fff",
                        }),
                        html.Div(id="fp-method-caption", style={"fontSize": "16px", "color": "#444", "marginTop": "6px", "lineHeight": "1.55", "whiteSpace": "normal"}),
                    ])
                ], style={"height": "100%", "maxWidth": "50vw", "margin": "0 auto"}),
            ], md=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Button("Запустить кластеризацию", id="run-clustering", color="success", className="w-100"), md=4),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="sim-hist"), md=4),
            dbc.Col(dcc.Graph(id="cluster-scatter"), md=4),
            dbc.Col(dcc.Graph(id="cluster-bar"), md=4),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="tsne-graph"), md=6),
            dbc.Col(dcc.Graph(id="hier-dendrogram"), md=6),
        ], className="mt-2"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="hier-cluster-hist"), md=12),
        ]),
        html.Hr(),
        dash_table.DataTable(
            id="sim-table",
            columns=[*table_columns, {"name": "Similarity", "id": "Similarity", "type": "numeric", "format": {"specifier": ".3f"}}, {"name": "Cluster", "id": "Cluster"}],
            data=[],
            page_size=15,
            filter_action="native",
            sort_action="native",
            style_table={"overflowX": "auto", "height": "40vh"},
        ),
    ])


df = initial_df()
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.layout = dbc.Container([
    dcc.Store(id="df-store", data=df.to_dict("records")),
    dcc.Location(id="url"),
    dbc.Row([
        dbc.Col(
            dbc.Nav([
                dbc.NavLink("Main", href="/", active="exact"),
                dbc.NavLink("Схожесть и кластеризация", href="/clustering", active="exact"),
            ], pills=True, className="mb-2"),
            md=8,
        ),
        dbc.Col(
            dcc.Upload(id="upload-data", children=dbc.Button("Upload CSV", color="secondary"), multiple=False),
            md=4,
            className="d-flex justify-content-end align-items-start",
        ),
    ], className="my-2"),
    html.Div(id="upload-status", className="mb-2 text-muted"),
    html.Div(id="page-content"),
], fluid=True)


@app.callback(
    Output("df-store", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def upload_csv(contents, filename):
    if not contents:
        return no_update, no_update
    try:
        new_df = parse_uploaded_csv(contents)
        return new_df.to_dict("records"), f"✅ Loaded: {filename} ({len(new_df)} rows)"
    except Exception as exc:
        return no_update, f"❌ Upload error: {exc}"


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("df-store", "data"),
)
def render_page(pathname, data):
    local_df = prepare_df(pd.DataFrame(data))
    if pathname == "/clustering":
        return page_similarity_controls(local_df)
    return page_main(local_df)


def _fingerprint_visual_content(method: str):
    method = method or "Morgan"
    diagrams = {
        "Morgan": {
            "title": "Morgan / ECFP",
            "accent": "#31a7ff",
            "left_svg": "<g>"
                        "<circle cx='120' cy='120' r='74' fill='#31a7ff14' stroke='#31a7ff66' stroke-dasharray='6,4'/>"
                        "<circle cx='120' cy='120' r='52' fill='#74c5ff1e' stroke='#2f94de88'/>"
                        "<circle cx='120' cy='120' r='30' fill='#b7e3ff33' stroke='#2b82c388'/>"
                        "<circle cx='120' cy='120' r='8' fill='#ffffff' stroke='#1f6ea8' stroke-width='2'/>"
                        "<circle cx='82' cy='92' r='6' fill='#2f94de'/><circle cx='164' cy='106' r='6' fill='#2f94de'/><circle cx='92' cy='152' r='6' fill='#2f94de'/>"
                        "<line x1='120' y1='120' x2='82' y2='92' stroke='#2f94de' stroke-width='2'/>"
                        "<line x1='120' y1='120' x2='164' y2='106' stroke='#2f94de' stroke-width='2'/>"
                        "<line x1='120' y1='120' x2='92' y2='152' stroke='#2f94de' stroke-width='2'/>"
                        "<text x='200' y='84' fill='#1f5f90' font-size='11'>r=3 shell</text>"
                        "<text x='200' y='110' fill='#1f5f90' font-size='11'>r=2 shell</text>"
                        "<text x='200' y='136' fill='#1f5f90' font-size='11'>r=1 shell</text>"
                        "<text x='58' y='206' fill='#1b456a' font-size='12.5'>Circular atom environments + iterative hashing</text>"
                        "</g>",
            "steps": ["Select atom seeds and radius depth.", "Expand neighborhoods shell-by-shell around each atom.", "Canonicalize local environments into integer identifiers.", "Fold hashed identifiers into fixed-length bit space."],
            "caption": "Morgan/ECFP: concentric atom neighborhoods are iteratively encoded and folded into a compact binary signature.",
        },
        "RDKit": {
            "title": "RDKit Path FP",
            "accent": "#2eae66",
            "left_svg": "<g>"
                        "<path d='M44 84 L92 64 L140 88 L190 70 L238 96' stroke='#2eae66' stroke-width='3' fill='none'/>"
                        "<path d='M92 64 L102 118 L68 150' stroke='#2eae66' stroke-width='3' fill='none'/>"
                        "<path d='M140 88 L148 142 L196 162' stroke='#2eae66' stroke-width='3' fill='none'/>"
                        "<rect x='46' y='176' width='206' height='20' rx='4' fill='#e9f9ef' stroke='#8dd3ad'/>"
                        "<text x='56' y='190' fill='#1f5f3a' font-size='11.5'>Path: C-C-N=C-O (len=5)  → hash 1832</text>"
                        "<text x='66' y='210' fill='#1f5f3a' font-size='12.5'>Typed shortest paths + bond order patterns</text>"
                        "</g>",
            "steps": ["Define max topological path length.", "Enumerate linear and branched bond paths.", "Encode path atom/bond types as path tokens.", "Hash tokenized paths into bit indices."],
            "caption": "RDKit Path FP: atom/bond typed graph paths are enumerated and hashed to a fixed-size bit vector.",
        },
        "MACCS": {
            "title": "MACCS Keys",
            "accent": "#d59a2a",
            "left_svg": "<g>"
                        "<rect x='40' y='66' width='224' height='108' rx='10' fill='#fff7e8' stroke='#d59a2a'/>"
                        "<text x='54' y='88' fill='#8a6114' font-size='11.5' font-weight='700'>SMARTS dictionary (fixed 166 keys)</text>"
                        "<text x='56' y='108' fill='#9a6d18' font-size='12'>#12 aromatic ring    ✓</text>"
                        "<text x='56' y='126' fill='#9a6d18' font-size='12'>#48 carboxylate      ✗</text>"
                        "<text x='56' y='144' fill='#9a6d18' font-size='12'>#101 hetero 3-cycle  ✓</text>"
                        "<rect x='58' y='184' width='190' height='14' rx='4' fill='#fff' stroke='#d8be84'/>"
                        "<rect x='58' y='184' width='56' height='14' rx='4' fill='#f0c468'/>"
                        "<text x='66' y='211' fill='#7b5410' font-size='12.5'>Interpretable yes/no structural keys</text>"
                        "</g>",
            "steps": ["Use predefined SMARTS-like structural keys.", "Match each key against the molecular graph.", "Write 1 for present keys and 0 for absent keys.", "Return fixed-length interpretable binary key vector."],
            "caption": "MACCS Keys: a curated fixed dictionary of structural rules is matched directly into an interpretable binary keyset.",
        },
        "AtomPair": {
            "title": "AtomPair FP",
            "accent": "#6c54d8",
            "left_svg": "<g>"
                        "<circle cx='72' cy='94' r='7' fill='#6c54d8'/><text x='60' y='80' fill='#4a36aa' font-size='11'>N(sp2)</text>"
                        "<circle cx='184' cy='138' r='7' fill='#6c54d8'/><text x='170' y='158' fill='#4a36aa' font-size='11'>O(sp2)</text>"
                        "<line x1='72' y1='94' x2='184' y2='138' stroke='#6c54d8' stroke-width='2.5' stroke-dasharray='5,4'/>"
                        "<text x='94' y='108' fill='#4a36aa' font-size='11.5'>graph distance = 5 bonds</text>"
                        "<rect x='58' y='172' width='194' height='22' rx='5' fill='#efeaff' stroke='#b5a5f0'/>"
                        "<text x='66' y='187' fill='#4a36aa' font-size='11.5'>(N.sp2, O.sp2, d=5)  → hash 742</text>"
                        "<text x='72' y='211' fill='#4a36aa' font-size='12.5'>Typed atom pairs + topological distance bins</text>"
                        "</g>",
            "steps": ["Assign atom environment types (element/hybridization/neighbors).", "Compute shortest-path distance for each atom pair.", "Build pair tuples: (typeA, typeB, distance).", "Hash/fold tuples into a fixed fingerprint bitset."],
            "caption": "AtomPair FP: chemically typed atom pairs with graph distance are encoded into hashed binary features.",
        },
        "TopologicalTorsion": {
            "title": "Topological Torsion",
            "accent": "#21a4b5",
            "left_svg": "<g>"
                        "<path d='M52 122 L96 96 L140 120 L186 94' stroke='#21a4b5' stroke-width='3' fill='none'/>"
                        "<circle cx='52' cy='122' r='5' fill='#21a4b5'/><circle cx='96' cy='96' r='5' fill='#21a4b5'/>"
                        "<circle cx='140' cy='120' r='5' fill='#21a4b5'/><circle cx='186' cy='94' r='5' fill='#21a4b5'/>"
                        "<text x='62' y='84' fill='#1d6f7a' font-size='11.5'>ordered 4-atom sequence</text>"
                        "<text x='52' y='146' fill='#1d6f7a' font-size='11'>C(sp2)-N(sp2)-C(sp2)-O(sp2)</text>"
                        "<rect x='52' y='170' width='204' height='22' rx='5' fill='#e7f9fc' stroke='#8bcfd8'/>"
                        "<text x='62' y='185' fill='#1b6b75' font-size='11.5'>torsion token = 11-7-11-9  → hash 2194</text>"
                        "<text x='66' y='211' fill='#1b6b75' font-size='12.5'>Ordered 4-atom motifs preserve sequence context</text>"
                        "</g>",
            "steps": ["Enumerate all connected 4-atom topological paths.", "Preserve order of atom types along each path.", "Create torsion tokens from ordered atom-type sequence.", "Hash token set into fixed-length fingerprint bits."],
            "caption": "Topological Torsion FP: ordered 4-atom motifs retain sequence context and are mapped into hashed bits.",
        },
        "Pattern": {
            "title": "Pattern FP",
            "accent": "#20a67a",
            "left_svg": "<g>"
                        "<rect x='46' y='74' width='208' height='94' rx='10' fill='#eafff5' stroke='#20a67a'/>"
                        "<text x='58' y='96' fill='#14664c' font-size='11.5' font-weight='700'>Pattern matcher</text>"
                        "<text x='58' y='114' fill='#14664c' font-size='11.5'>[#6]-[#7]-[#6]=[#8]     found</text>"
                        "<text x='58' y='132' fill='#14664c' font-size='11.5'>[a]-[a]-[a]-[a]         found</text>"
                        "<text x='58' y='150' fill='#14664c' font-size='11.5'>[#16]-[#16]             absent</text>"
                        "<rect x='62' y='178' width='186' height='18' rx='4' fill='#ffffff' stroke='#90d9bf'/>"
                        "<text x='69' y='191' fill='#14664c' font-size='11'>match events -> hashed bit events</text>"
                        "<text x='74' y='211' fill='#14664c' font-size='12.5'>Substructure dictionary hit profile</text>"
                        "</g>",
            "steps": ["Compile a library of SMARTS-like molecular patterns.", "Search each pattern over the molecular graph.", "Convert pattern hits into feature events.", "Hash events into a compact bit representation."],
            "caption": "Pattern FP: dictionary-style substructure hits are aggregated and hashed into a binary event profile.",
        },
        "Layered": {
            "title": "Layered FP",
            "accent": "#8d63e1",
            "left_svg": "<g>"
                        "<rect x='62' y='74' width='150' height='20' rx='4' fill='#f3ecff' stroke='#8d63e1'/>"
                        "<rect x='52' y='100' width='170' height='20' rx='4' fill='#eadfff' stroke='#8d63e1'/>"
                        "<rect x='42' y='126' width='190' height='20' rx='4' fill='#dfceff' stroke='#8d63e1'/>"
                        "<rect x='32' y='152' width='210' height='20' rx='4' fill='#d2baff' stroke='#8d63e1'/>"
                        "<text x='70' y='88' fill='#5c3aa8' font-size='11.5'>Layer 1: atom types</text>"
                        "<text x='60' y='114' fill='#5c3aa8' font-size='11.5'>Layer 2: bond order paths</text>"
                        "<text x='50' y='140' fill='#5c3aa8' font-size='11.5'>Layer 3: ring/path motifs</text>"
                        "<text x='40' y='166' fill='#5c3aa8' font-size='11.5'>Layer 4: combined topology cues</text>"
                        "<text x='54' y='211' fill='#5c3aa8' font-size='12.5'>Multi-layer graph abstraction before hashing</text>"
                        "</g>",
            "steps": ["Generate features at progressively richer graph layers.", "Combine atom, bond, and path-level feature channels.", "Transform layered features into identifiers.", "Fold accumulated identifiers into fingerprint bits."],
            "caption": "Layered FP: multiple stacked feature layers capture topology at different abstraction levels before hashing.",
        },
        "Avalon": {
            "title": "Avalon FP",
            "accent": "#0f93a8",
            "left_svg": "<g>"
                        "<path d='M64 90 L112 76 L158 98 L204 82 L236 104' stroke='#0f93a8' stroke-width='3' fill='none'/>"
                        "<circle cx='112' cy='76' r='16' fill='#0f93a822' stroke='#0f93a8'/>"
                        "<circle cx='204' cy='82' r='14' fill='#0f93a822' stroke='#0f93a8'/>"
                        "<rect x='52' y='120' width='194' height='54' rx='8' fill='#e7f8fb' stroke='#90cfd9'/>"
                        "<text x='62' y='142' fill='#0c5f6e' font-size='11.5'>feature class A: heteroaromatic motif</text>"
                        "<text x='62' y='158' fill='#0c5f6e' font-size='11.5'>feature class B: donor/acceptor pattern</text>"
                        "<text x='60' y='211' fill='#0c5f6e' font-size='12.5'>Medicinal-chemistry tuned feature hashing</text>"
                        "</g>",
            "steps": ["Extract Avalon-specific medicinal-chemistry features.", "Normalize/canonicalize feature event combinations.", "Map canonical events into fixed numeric identifiers.", "Fold identifiers into dense binary fingerprint space."],
            "caption": "Avalon FP: medicinal-chemistry-oriented feature classes are canonicalized and hashed into dense binary bits.",
        },
    }
    d = diagrams.get(method, diagrams["Morgan"])
    title, accent, steps = d["title"], d["accent"], d["steps"]
    def wrap_lines(text: str, width: int = 38):
        words = text.split()
        lines = []
        cur = ""
        for w in words:
            nxt = f"{cur} {w}".strip()
            if len(nxt) <= width:
                cur = nxt
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    steps_svg = ""
    y = 78
    for i, step in enumerate(steps, start=1):
        wrapped = wrap_lines(f"{i}) {step}", width=40)
        for j, line in enumerate(wrapped):
            steps_svg += f"<text x='352' y='{y}' fill='#2f3b4d' font-size='14'>{line}</text>"
            y += 14
        y += 3
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='640' height='300' viewBox='0 0 640 300'>"
        f"<rect x='4' y='4' width='632' height='292' rx='14' fill='#ffffff' stroke='#cfd8e3' stroke-width='2'/>"
        f"<text x='20' y='28' fill='#1e2b3a' font-size='18' font-weight='700'>{title}</text>"
        f"<text x='20' y='46' fill='#4c5e75' font-size='13'>Conceptual scheme (method-specific)</text>"
        f"<rect x='22' y='62' width='300' height='210' rx='10' fill='#fbfdff' stroke='#d8e0ea'/>"
        f"{d['left_svg']}"
        f"<rect x='30' y='236' width='282' height='30' rx='6' fill='#eef3f9' stroke='#d2dbe7'/>"
        f"<text x='38' y='256' fill='#22354b' font-size='12'>Feature extraction  →  Hashing  →  Bit vector</text>"
        f"<rect x='340' y='18' width='286' height='262' rx='12' fill='#f7fafc' stroke='#d3dde8'/>"
        f"<text x='352' y='40' fill='#1f2b3b' font-size='16' font-weight='700'>How this method works</text>"
        f"<line x1='352' y1='48' x2='614' y2='48' stroke='{accent}' stroke-width='3'/>"
        f"{steps_svg}"
        f"<rect x='348' y='252' width='270' height='20' rx='6' fill='{accent}' fill-opacity='0.15'/>"
        f"<text x='354' y='266' fill='#22354b' font-size='12.5'>Fingerprint = compressed molecular feature signature</text>"
        f"</svg>"
    )
    return f"data:image/svg+xml;utf8,{quote(svg)}", d["caption"]


def label_with_default(text: str):
    return html.Span([text, " ", html.Span("(по умолчанию)", style={"color": "#4b4f55"})])


@app.callback(
    Output("fp-params-container", "children"),
    Input("fp-method", "value"),
)
def update_fp_params_ui(fp_method):
    if fp_method == "Morgan":
        return dbc.Row([
            dbc.Col([
                html.Div(label_with_default("Radius")),
                dcc.Slider(id="fp-radius", min=1, max=4, step=1, value=2, marks={i: str(i) for i in range(1, 5)}),
            ], md=6),
            dbc.Col([
                html.Div(label_with_default("Fingerprint bits")),
                dcc.Slider(id="fp-bits", min=256, max=4096, step=256, value=2048,
                           marks={256: "256", 1024: "1024", 2048: "2048", 4096: "4096"}),
            ], md=6),
        ])
    if fp_method in {"RDKit", "AtomPair", "TopologicalTorsion", "Pattern", "Layered", "Avalon"}:
        return dbc.Row([
            dbc.Col([
                html.Div(label_with_default("Fingerprint bits")),
                dcc.Slider(id="fp-bits", min=256, max=4096, step=256, value=2048,
                           marks={256: "256", 1024: "1024", 2048: "2048", 4096: "4096"}),
            ], md=12),
        ])
    return html.Div("No additional parameters for this fingerprint method (MACCS has fixed size).", className="text-muted")


@app.callback(
    Output("sim-params-container", "children"),
    Input("sim-method", "value"),
)
def update_similarity_params_ui(sim_method):
    cutoff_slider = html.Div([
        html.Div(label_with_default("Similarity cutoff")),
        dcc.Slider(
            id="sim-threshold",
            min=0.0,
            max=1.0,
            step=0.1,
            value=0.7,
            marks={round(i / 10, 1): f"{round(i / 10, 1)}" for i in range(0, 11)},
        ),
    ], style={"width": "100%", "marginBottom": "8px"})

    if sim_method == "Tversky":
        return html.Div([
            cutoff_slider,
            html.Div("Tversky alpha"),
            dcc.Slider(
                id="tversky-alpha",
                min=0.0,
                max=1.0,
                step=0.1,
                value=0.5,
                marks={round(i / 10, 1): f"{round(i / 10, 1)}" for i in range(0, 11)},
            ),
            html.Div("Tversky beta", style={"marginTop": "8px"}),
            dcc.Slider(
                id="tversky-beta",
                min=0.0,
                max=1.0,
                step=0.1,
                value=0.5,
                marks={round(i / 10, 1): f"{round(i / 10, 1)}" for i in range(0, 11)},
            ),
        ], style={"width": "100%"})

    return cutoff_slider


@app.callback(
    Output("embedding-params-container", "children"),
    Input("embedding-method", "value"),
)
def update_embedding_params_ui(embedding_method):
    if embedding_method == "PCA":
        return html.Div([
            html.Div(label_with_default("PCA: n_components")),
            dcc.Slider(id="pca-n-components", min=2, max=3, step=1, value=2, marks={2: "2", 3: "3"}),
            html.Div("PCA: whiten", style={"marginTop": "8px"}),
            dcc.Dropdown(id="pca-whiten", options=["False", "True"], value="False", style={"width": "100%"}),
        ], style={"width": "100%"})
    if embedding_method == "T-SNE":
        return html.Div([
            html.Div(label_with_default("T-SNE: perplexity")),
            dcc.Slider(id="tsne-perplexity", min=5, max=50, step=5, value=30, marks={5: "5", 10: "10", 20: "20", 30: "30", 40: "40", 50: "50"}),
            html.Div("T-SNE: learning rate", style={"marginTop": "8px"}),
            dcc.Slider(id="tsne-learning-rate", min=10, max=500, step=10, value=200, marks={10: "10", 100: "100", 200: "200", 300: "300", 500: "500"}),
            html.Div("T-SNE: n_iter", style={"marginTop": "8px"}),
            dcc.Slider(id="tsne-n-iter", min=250, max=3000, step=250, value=1000, marks={250: "250", 500: "500", 1000: "1000", 2000: "2000", 3000: "3000"}),
        ], style={"width": "100%"})
    if embedding_method == "UMAP":
        return html.Div([
            html.Div(label_with_default("UMAP: n_neighbors")),
            dcc.Slider(id="umap-n-neighbors", min=2, max=100, step=1, value=15, marks={2: "2", 5: "5", 15: "15", 30: "30", 50: "50", 100: "100"}),
            html.Div("UMAP: min_dist", style={"marginTop": "8px"}),
            dcc.Slider(id="umap-min-dist", min=0.0, max=0.99, step=0.01, value=0.1, marks={0.0: "0.0", 0.1: "0.1", 0.3: "0.3", 0.5: "0.5", 0.8: "0.8", 0.99: "0.99"}),
            html.Div("UMAP: metric", style={"marginTop": "8px"}),
            dcc.Dropdown(id="umap-metric", options=["euclidean", "cosine", "manhattan", "hamming"], value="euclidean", style={"width": "100%"}),
        ], style={"width": "100%"})
    return html.Div("Сжатие отключено: будет использована исходная проекция для кластерных графиков.", className="text-muted")


@app.callback(
    Output("cluster-params-container", "children"),
    Input("cluster-method", "value"),
)
def update_cluster_params_ui(cluster_method):
    if cluster_method == "KMeans":
        return html.Div([
            html.Div(label_with_default("KMeans: choose k mode"), style={"marginTop": "8px"}),
            dcc.Dropdown(id="kmeans-k-mode", options=["manual", "auto_elbow"], value="manual", style={"width": "100%"}),
            html.Div(id="kmeans-mode-extra", style={"marginTop": "8px"}),
        ], style={"width": "100%"})
    if cluster_method == "HDBSCAN":
        return html.Div([
            html.Div(label_with_default("HDBSCAN: min cluster size")),
            dcc.Slider(
                id="hdbscan-min-size",
                min=2,
                max=30,
                step=1,
                value=5,
                marks={2: "2", 5: "5", 10: "10", 20: "20", 30: "30"},
            ),
            html.Div("HDBSCAN: min samples", style={"marginTop": "8px"}),
            dcc.Slider(id="hdbscan-min-samples", min=1, max=20, step=1, value=5,
                       marks={1: "1", 5: "5", 10: "10", 15: "15", 20: "20"}),
            html.Div("HDBSCAN: metric", style={"marginTop": "8px"}),
            dcc.Dropdown(id="hdbscan-metric", options=["euclidean", "manhattan", "cosine"], value="euclidean", style={"width": "100%"}),
        ], style={"width": "100%"})
    if cluster_method == "DBSCAN":
        return html.Div([
            html.Div(label_with_default("DBSCAN: eps")),
            dcc.Slider(
                id="dbscan-eps",
                min=0.1,
                max=5.0,
                step=0.1,
                value=0.5,
                marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0", 2.0: "2.0", 5.0: "5.0"},
            ),
            html.Div("DBSCAN: min samples", style={"marginTop": "8px"}),
            dcc.Slider(id="dbscan-min-samples", min=1, max=20, step=1, value=5,
                       marks={1: "1", 3: "3", 5: "5", 10: "10", 20: "20"}),
            html.Div("DBSCAN: metric", style={"marginTop": "8px"}),
            dcc.Dropdown(id="dbscan-metric", options=["euclidean", "manhattan", "cosine", "hamming"], value="euclidean", style={"width": "100%"}),
        ], style={"width": "100%"})
    if cluster_method == "Hierarchical":
        return html.Div([
            html.Div(label_with_default("Hierarchical: linkage")),
            dcc.Dropdown(id="hier-linkage", options=["average", "complete", "single", "ward"], value="ward", style={"width": "100%"}),
            html.Div("Hierarchical: choose cluster mode", style={"marginTop": "8px"}),
            dcc.Dropdown(id="hier-cluster-mode", options=["manual", "auto_elbow"], value="manual", style={"width": "100%"}),
            html.Div("Hierarchical: criterion", style={"marginTop": "8px"}),
            dcc.Dropdown(id="hier-criterion", options=["distance", "maxclust"], value="distance", style={"width": "100%"}),
            html.Div(id="hier-criterion-extra", style={"marginTop": "8px"}),
        ], style={"width": "100%"})
    if cluster_method == "Butina":
        return html.Div("Butina uses Similarity cutoff from the similarity section.", className="text-muted")
    return html.Div("No extra parameters required for this clustering method.", className="text-muted")


@app.callback(
    Output("hier-criterion-extra", "children"),
    Input("hier-criterion", "value", allow_optional=True),
    Input("hier-cluster-mode", "value", allow_optional=True),
)
def update_hier_criterion_extra(hier_criterion, hier_cluster_mode):
    if hier_criterion is None and hier_cluster_mode is None:
        return no_update
    if (hier_cluster_mode or "manual") == "auto_elbow":
        return html.Div("Auto_elbow mode will estimate optimal cluster count automatically.", className="text-muted")
    if hier_criterion == "maxclust":
        return html.Div([
            html.Div("Hierarchical: max clusters"),
            dcc.Slider(
                id="hier-max-clusters",
                min=2,
                max=20,
                step=1,
                value=8,
                marks={2: "2", 5: "5", 10: "10", 15: "15", 20: "20"},
            ),
        ], style={"width": "100%"})
    return html.Div("Hierarchical distance mode uses Similarity cutoff slider above.", className="text-muted")


@app.callback(
    Output("kmeans-mode-extra", "children"),
    Input("kmeans-k-mode", "value", allow_optional=True),
)
def update_kmeans_mode_extra(kmeans_k_mode):
    if kmeans_k_mode is None:
        return no_update
    common = [
        html.Div("KMeans: init", style={"marginTop": "8px"}),
        dcc.Dropdown(id="kmeans-init", options=["k-means++", "random"], value="k-means++", style={"width": "100%"}),
        html.Div("KMeans: max iterations", style={"marginTop": "8px"}),
        dcc.Slider(id="kmeans-max-iter", min=50, max=1000, step=50, value=300,
                   marks={50: "50", 200: "200", 300: "300", 500: "500", 1000: "1000"}),
    ]
    if kmeans_k_mode == "auto_elbow":
        return html.Div([
            html.Div("KMeans: number of clusters is selected automatically (elbow).", className="text-muted"),
            *common
        ], style={"width": "100%"})
    return html.Div([
        html.Div("KMeans: number of clusters"),
        dcc.Slider(id="kmeans-k", min=2, max=12, step=1, value=8, marks={i: str(i) for i in range(2, 13)}),
        *common
    ], style={"width": "100%"})


@app.callback(
    Output("fp-method-image", "src"),
    Output("fp-method-caption", "children"),
    Input("fp-method", "value"),
)
def update_fp_method_visual(fp_method):
    return _fingerprint_visual_content(fp_method)


@app.callback(
    Output("main-table", "data"),
    Output("main-scatter", "figure"),
    Input("main-x", "value"),
    Input("main-y", "value"),
    Input("main-row-pct", "value"),
    Input("df-store", "data"),
    prevent_initial_call=True,
)
def update_main_page(x_col, y_col, pct, data):
    local_df = prepare_df(pd.DataFrame(data))
    n_rows = max(1, int(len(local_df) * pct / 100))
    tdf = local_df.head(n_rows)
    fig = px.scatter(tdf, x=x_col or "MolWt", y=y_col or "TPSA", hover_data=["ID", "SMILES"], opacity=0.65)
    fig.update_layout(title=f"{x_col} vs {y_col}")
    return tdf.to_dict("records"), fig


@app.callback(
    Output("sim-table", "data"),
    Output("sim-hist", "figure"),
    Output("cluster-scatter", "figure"),
    Output("cluster-bar", "figure"),
    Output("tsne-graph", "figure"),
    Output("hier-dendrogram", "figure"),
    Output("hier-cluster-hist", "figure"),
    Input("run-clustering", "n_clicks"),
    State("fp-method", "value"),
    State("sim-method", "value"),
    State("embedding-method", "value"),
    State("cluster-method", "value"),
    State("fp-radius", "value", allow_optional=True),
    State("fp-bits", "value", allow_optional=True),
    State("sim-threshold", "value", allow_optional=True),
    State("tversky-alpha", "value", allow_optional=True),
    State("tversky-beta", "value", allow_optional=True),
    State("kmeans-k", "value", allow_optional=True),
    State("kmeans-k-mode", "value", allow_optional=True),
    State("kmeans-init", "value", allow_optional=True),
    State("kmeans-max-iter", "value", allow_optional=True),
    State("hdbscan-min-size", "value", allow_optional=True),
    State("hdbscan-min-samples", "value", allow_optional=True),
    State("hdbscan-metric", "value", allow_optional=True),
    State("dbscan-eps", "value", allow_optional=True),
    State("dbscan-min-samples", "value", allow_optional=True),
    State("dbscan-metric", "value", allow_optional=True),
    State("hier-linkage", "value", allow_optional=True),
    State("hier-cluster-mode", "value", allow_optional=True),
    State("hier-criterion", "value", allow_optional=True),
    State("hier-max-clusters", "value", allow_optional=True),
    State("pca-n-components", "value", allow_optional=True),
    State("pca-whiten", "value", allow_optional=True),
    State("tsne-perplexity", "value", allow_optional=True),
    State("tsne-learning-rate", "value", allow_optional=True),
    State("tsne-n-iter", "value", allow_optional=True),
    State("umap-n-neighbors", "value", allow_optional=True),
    State("umap-min-dist", "value", allow_optional=True),
    State("umap-metric", "value", allow_optional=True),
    State("sample-pct", "value"),
    State("df-store", "data"),
    prevent_initial_call=True,
)
def run_similarity_clustering(_, fp_method, sim_method, embedding_method, cluster_method, radius, nbits, threshold, tversky_alpha, tversky_beta, kmeans_k, kmeans_k_mode, kmeans_init, kmeans_max_iter, hdbscan_min_size, hdbscan_min_samples, hdbscan_metric, dbscan_eps, dbscan_min_samples, dbscan_metric, hier_linkage, hier_cluster_mode, hier_criterion, hier_max_clusters, pca_n_components, pca_whiten, tsne_perplexity, tsne_learning_rate, tsne_n_iter, umap_n_neighbors, umap_min_dist, umap_metric, sample_pct, data):
    local_df = prepare_df(pd.DataFrame(data))
    sample_count = max(2, int(len(local_df) * (sample_pct or 60) / 100))
    local_df = local_df.head(sample_count).reset_index(drop=True)
    cfg = FPConfig(fp_method, int(radius or 2), int(nbits or 2048))

    mols = [Chem.MolFromSmiles(str(s)) for s in local_df["SMILES"]]
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    fps = {}
    for i in valid_idx:
        try:
            fps[i] = make_fp(mols[i], cfg)
        except Exception:
            continue

    if len(fps) < 2:
        blank = go.Figure()
        blank.add_annotation(text="Not enough valid molecules for clustering", showarrow=False)
        return [], blank, blank, blank, blank, blank, blank

    fps_list = [(idx, fps[idx]) for idx in sorted(fps.keys())]
    n_valid = len(fps_list)
    safe_threshold = float(threshold if threshold is not None else 0.55)
    alpha = float(tversky_alpha if tversky_alpha is not None else 0.5)
    beta = float(tversky_beta if tversky_beta is not None else 0.5)
    dists = []
    pair_sims = []
    for i in range(1, n_valid):
        sims = DataStructs.BulkTanimotoSimilarity(fps_list[i][1], [fp for _, fp in fps_list[:i]]) if sim_method == "Tanimoto" else [
            sim_score(fps_list[i][1], fps_list[j][1], sim_method, alpha=alpha, beta=beta) for j in range(i)
        ]
        dists.extend([1.0 - s for s in sims])
        pair_sims.extend(sims)

    cluster_map = {}
    if cluster_method == "Butina":
        clusters = Butina.ClusterData(dists, n_valid, 1.0 - safe_threshold, isDistData=True)
        for c_idx, cluster in enumerate(clusters, start=1):
            for member in cluster:
                row_idx = fps_list[member][0]
                cluster_map[row_idx] = f"Cluster {c_idx}"
    elif cluster_method == "Hierarchical":
        chosen_linkage = (hier_linkage or "average").lower()
        chosen_mode = (hier_cluster_mode or "manual").lower()
        chosen_criterion = (hier_criterion or "distance").lower()
        if chosen_linkage == "ward":
            fps_matrix = np.array([fp_numpy(fp) for _, fp in fps_list], dtype=float)
            Z_h = linkage(fps_matrix, method="ward")
        else:
            dist_matrix_h = np.zeros((n_valid, n_valid), dtype=float)
            for i in range(n_valid):
                for j in range(i):
                    sim_ij = sim_score(fps_list[i][1], fps_list[j][1], sim_method, alpha=alpha, beta=beta)
                    dist = 1.0 - sim_ij
                    dist_matrix_h[i, j] = dist
                    dist_matrix_h[j, i] = dist
            condensed_h = squareform(dist_matrix_h, checks=False)
            Z_h = linkage(condensed_h, method=chosen_linkage)
        if chosen_mode == "auto_elbow":
            heights = Z_h[:, 2]
            if len(heights) >= 3:
                second_diff = np.diff(heights, n=2)
                elbow_merge_idx = int(np.argmax(second_diff)) + 1
                auto_k = max(2, min(n_valid, n_valid - elbow_merge_idx))
            else:
                auto_k = 2
            h_clusters = fcluster(Z_h, t=auto_k, criterion="maxclust")
        elif chosen_criterion == "maxclust":
            h_clusters = fcluster(Z_h, t=int(hier_max_clusters or 6), criterion="maxclust")
        else:
            h_clusters = fcluster(Z_h, t=max(1e-6, 1.0 - safe_threshold), criterion="distance")
        for i, label in enumerate(h_clusters):
            cluster_map[fps_list[i][0]] = f"Cluster {int(label)}"
    elif cluster_method == "KMeans":
        try:
            from sklearn.cluster import KMeans
            fps_matrix = np.array([fp_numpy(fp) for _, fp in fps_list], dtype=float)
            k = int(kmeans_k or 4)
            if (kmeans_k_mode or "manual") == "auto_elbow":
                candidate_ks = list(range(2, min(10, len(fps_matrix)) + 1))
                inertias = []
                for kk in candidate_ks:
                    km_tmp = KMeans(n_clusters=kk, n_init=10, init=kmeans_init or "k-means++", max_iter=int(kmeans_max_iter or 300), random_state=42)
                    km_tmp.fit(fps_matrix)
                    inertias.append(float(km_tmp.inertia_))
                if len(inertias) >= 3:
                    curvature = []
                    for i in range(1, len(inertias) - 1):
                        curvature.append((inertias[i - 1] - inertias[i]) - (inertias[i] - inertias[i + 1]))
                    elbow_idx = int(np.argmax(curvature)) + 1
                    k = candidate_ks[elbow_idx]
                else:
                    k = candidate_ks[0]
            km_labels = KMeans(
                n_clusters=max(2, k),
                n_init=10,
                init=kmeans_init or "k-means++",
                max_iter=int(kmeans_max_iter or 300),
                random_state=42
            ).fit_predict(fps_matrix)
            for i, label in enumerate(km_labels):
                cluster_map[fps_list[i][0]] = f"Cluster {int(label) + 1}"
        except Exception:
            for i, _ in fps_list:
                cluster_map[i] = "Cluster 1"
    elif cluster_method == "HDBSCAN":
        try:
            import hdbscan
            fps_matrix = np.array([fp_numpy(fp) for _, fp in fps_list], dtype=float)
            min_size = int(hdbscan_min_size or 8)
            labels = hdbscan.HDBSCAN(
                min_cluster_size=max(2, min_size),
                min_samples=int(hdbscan_min_samples or 5),
                metric=hdbscan_metric or "euclidean"
            ).fit_predict(fps_matrix)
            for i, label in enumerate(labels):
                cluster_map[fps_list[i][0]] = "Noise" if int(label) < 0 else f"Cluster {int(label) + 1}"
        except Exception:
            for i, _ in fps_list:
                cluster_map[i] = "Cluster 1"
    elif cluster_method == "DBSCAN":
        try:
            from sklearn.cluster import DBSCAN
            fps_matrix = np.array([fp_numpy(fp) for _, fp in fps_list], dtype=float)
            eps = float(dbscan_eps or 1.0)
            labels = DBSCAN(
                eps=eps,
                min_samples=int(dbscan_min_samples or 3),
                metric=dbscan_metric or "euclidean"
            ).fit_predict(fps_matrix)
            for i, label in enumerate(labels):
                cluster_map[fps_list[i][0]] = "Noise" if int(label) < 0 else f"Cluster {int(label) + 1}"
        except Exception:
            for i, _ in fps_list:
                cluster_map[i] = "Cluster 1"

    similarities = [np.nan] * len(local_df)
    for i, (_, fp_i) in enumerate(fps_list):
        sims = [sim_score(fp_i, fp_j, sim_method, alpha=alpha, beta=beta) for _, fp_j in fps_list if fp_j is not fp_i]
        similarities[fps_list[i][0]] = float(np.mean(sims)) if sims else 1.0

    out = local_df.copy()
    out["Similarity"] = similarities
    out["Cluster"] = out.index.map(lambda idx: cluster_map.get(idx, "Invalid SMILES"))
    out.loc[out["Similarity"].isna(), "Cluster"] = "Invalid SMILES"
    out = out.sort_values("Similarity", ascending=False, na_position="last")

    hfig = px.histogram(out.dropna(subset=["Similarity"]), x="Similarity", nbins=40, color="Cluster", title="Similarity distribution")
    sfig = px.scatter(out.dropna(subset=["Similarity"]), x="MolWt", y="TPSA", color="Cluster", size="Similarity",
                      hover_data=["ID", "SMILES", "Similarity"], title=f"Clusters for {len(out)} molecules")
    bfig = px.bar(out["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"), x="Cluster", y="Count", color="Cluster", title="Cluster sizes")

    # ---------- Embedding (None / PCA / T-SNE / UMAP) ----------
    tsne_fig = go.Figure()
    try:
        fps_matrix = np.array([fp_numpy(fp) for _, fp in fps_list], dtype=float)
        emb_method = embedding_method or "T-SNE"
        if emb_method == "Не сжимать данные":
            emb = fps_matrix[:, :2] if fps_matrix.shape[1] >= 2 else np.column_stack([np.arange(len(fps_matrix)), np.zeros(len(fps_matrix))])
            title_embed = "Raw 2D projection (no dimensionality reduction)"
        elif emb_method == "PCA":
            from sklearn.decomposition import PCA
            pca = PCA(
                n_components=int(pca_n_components or 2),
                whiten=(str(pca_whiten or "False") == "True"),
                random_state=42
            )
            emb = pca.fit_transform(fps_matrix)
            if emb.shape[1] == 1:
                emb = np.column_stack([emb[:, 0], np.zeros(len(emb))])
            title_embed = "PCA projection of fingerprints"
        elif emb_method == "UMAP":
            import umap.umap_ as umap
            reducer = umap.UMAP(
                n_neighbors=int(umap_n_neighbors or 15),
                min_dist=float(umap_min_dist if umap_min_dist is not None else 0.1),
                metric=umap_metric or "euclidean",
                random_state=42,
            )
            emb = reducer.fit_transform(fps_matrix)
            title_embed = "UMAP projection of fingerprints"
        else:
            from sklearn.manifold import TSNE
            perplexity = int(tsne_perplexity or 30)
            perplexity = max(2, min(perplexity, len(fps_matrix) - 1))
            tsne = TSNE(
                n_components=2,
                random_state=42,
                init="random",
                learning_rate=float(tsne_learning_rate or 200),
                perplexity=perplexity,
                n_iter=int(tsne_n_iter or 1000),
            )
            emb = tsne.fit_transform(fps_matrix)
            title_embed = "T-SNE projection of fingerprints"

        tsne_df = pd.DataFrame({
            "x": emb[:, 0],
            "y": emb[:, 1],
            "Cluster": [cluster_map.get(idx, "Unknown") for idx, _ in fps_list],
            "ID": [str(local_df.iloc[idx]["ID"]) for idx, _ in fps_list],
        })
        tsne_fig = px.scatter(tsne_df, x="x", y="y", color="Cluster", hover_data=["ID"], title=title_embed)
    except Exception as exc:
        tsne_fig.add_annotation(text=f"Embedding unavailable: {exc}", showarrow=False)

    # ---------- Hierarchical clustering ----------
    dendro_fig = go.Figure()
    hier_hist_fig = go.Figure()
    try:
        dist_matrix = np.zeros((n_valid, n_valid), dtype=float)
        for i in range(n_valid):
            dist_matrix[i, i] = 0.0
            for j in range(i):
                sim_ij = sim_score(fps_list[i][1], fps_list[j][1], sim_method, alpha=alpha, beta=beta)
                dist = 1.0 - sim_ij
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method="average")
        labels = [str(local_df.iloc[idx]["ID"]) for idx, _ in fps_list]
        dendro_fig = ff.create_dendrogram(dist_matrix, labels=labels, linkagefun=lambda _: Z)
        dendro_fig.update_layout(title="Hierarchical clustering dendrogram")

        h_clusters = fcluster(Z, t=max(1e-6, 1.0 - safe_threshold), criterion="distance")
        h_sizes = pd.Series(h_clusters).value_counts().sort_index().reset_index()
        h_sizes.columns = ["HierCluster", "Count"]
        h_sizes["HierCluster"] = h_sizes["HierCluster"].astype(str)
        hier_hist_fig = px.bar(h_sizes, x="HierCluster", y="Count", title="Hierarchical cluster size histogram")
    except Exception as exc:
        dendro_fig.add_annotation(text=f"Hierarchical clustering unavailable: {exc}", showarrow=False)
        hier_hist_fig.add_annotation(text=f"Histogram unavailable: {exc}", showarrow=False)

    return out.to_dict("records"), hfig, sfig, bfig, tsne_fig, dendro_fig, hier_hist_fig


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8054)

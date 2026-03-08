import io
import json
import math
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ezdxf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Polygon as MplPolygon
from openai import OpenAI
from PIL import Image
from shapely.geometry import Point, Polygon


st.set_page_config(
    page_title="DXF + IA, Quantitativos Arquitetônicos",
    page_icon="📐",
    layout="wide",
)


# =========================================================
# Configuração
# =========================================================
MAX_ROOM_CANDIDATES = 80
MAX_TEXT_SAMPLES = 120
MAX_BLOCK_SAMPLES = 120


# =========================================================
# Modelos
# =========================================================
@dataclass
class TextEntity:
    text: str
    x: float
    y: float
    layer: str


@dataclass
class InsertEntity:
    name: str
    x: float
    y: float
    layer: str


@dataclass
class RoomCandidate:
    candidate_id: int
    source: str
    layer: str
    area_model_units2: float
    perimeter_model_units: float
    area_m2: float
    perimeter_m: float
    centroid_x: float
    centroid_y: float
    bbox_min_x: float
    bbox_min_y: float
    bbox_max_x: float
    bbox_max_y: float
    text_inside: List[str]
    polygon_xy: List[Tuple[float, float]]


# =========================================================
# Utilidades geométricas
# =========================================================
def polygon_from_points(points: List[Tuple[float, float]]) -> Optional[Polygon]:
    if len(points) < 3:
        return None
    try:
        poly = Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        if poly.geom_type != "Polygon":
            return None
        return poly
    except Exception:
        return None


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def linestring_length(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        total += distance(points[i], points[i + 1])
    return total


def convert_model_to_m(value: float, unit_factor_to_m: float, scale_1_to_x: float) -> float:
    return value * unit_factor_to_m * scale_1_to_x


def convert_area_to_m2(value: float, unit_factor_to_m: float, scale_1_to_x: float) -> float:
    factor = unit_factor_to_m * scale_1_to_x
    return value * factor * factor


# =========================================================
# Leitura do DXF
# =========================================================
def load_dxf_document(uploaded_file) -> ezdxf.document.Drawing:
    suffix = Path(uploaded_file.name).suffix.lower() or ".dxf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    doc = ezdxf.readfile(tmp_path)
    return doc


def get_modelspace_entities(doc: ezdxf.document.Drawing):
    return doc.modelspace()


def extract_layers(doc: ezdxf.document.Drawing) -> List[str]:
    names = []
    for layer in doc.layers:
        try:
            names.append(layer.dxf.name)
        except Exception:
            pass
    return sorted(list(set(names)))


def safe_layer(entity) -> str:
    try:
        return entity.dxf.layer
    except Exception:
        return "0"


def extract_text_entities(msp) -> List[TextEntity]:
    items: List[TextEntity] = []

    for e in msp:
        dxftype = e.dxftype()

        try:
            if dxftype == "TEXT":
                text = str(e.dxf.text).strip()
                x = float(e.dxf.insert.x)
                y = float(e.dxf.insert.y)
                if text:
                    items.append(TextEntity(text=text, x=x, y=y, layer=safe_layer(e)))

            elif dxftype == "MTEXT":
                text = str(e.text).replace("\\P", " ").strip()
                x = float(e.dxf.insert.x)
                y = float(e.dxf.insert.y)
                if text:
                    items.append(TextEntity(text=text, x=x, y=y, layer=safe_layer(e)))
        except Exception:
            continue

    return items


def extract_insert_entities(msp) -> List[InsertEntity]:
    items: List[InsertEntity] = []

    for e in msp:
        if e.dxftype() != "INSERT":
            continue
        try:
            items.append(
                InsertEntity(
                    name=str(e.dxf.name),
                    x=float(e.dxf.insert.x),
                    y=float(e.dxf.insert.y),
                    layer=safe_layer(e),
                )
            )
        except Exception:
            continue

    return items


def extract_polyline_points(entity) -> List[Tuple[float, float]]:
    dxftype = entity.dxftype()

    if dxftype == "LWPOLYLINE":
        pts = []
        for p in entity.get_points("xy"):
            pts.append((float(p[0]), float(p[1])))
        return pts

    if dxftype == "POLYLINE":
        pts = []
        try:
            for v in entity.vertices:
                pts.append((float(v.dxf.location.x), float(v.dxf.location.y)))
        except Exception:
            pass
        return pts

    return []


def is_closed_polyline(entity) -> bool:
    dxftype = entity.dxftype()

    if dxftype == "LWPOLYLINE":
        try:
            return bool(entity.closed)
        except Exception:
            return False

    if dxftype == "POLYLINE":
        try:
            return bool(entity.is_closed)
        except Exception:
            return False

    return False


def extract_room_candidates(
    msp,
    texts: List[TextEntity],
    unit_factor_to_m: float,
    scale_1_to_x: float,
    min_area_m2: float,
    max_area_m2: float,
) -> List[RoomCandidate]:
    candidates: List[RoomCandidate] = []
    candidate_id = 1

    text_points = []
    for t in texts:
        try:
            text_points.append((t, Point(t.x, t.y)))
        except Exception:
            continue

    for e in msp:
        if e.dxftype() not in {"LWPOLYLINE", "POLYLINE"}:
            continue

        if not is_closed_polyline(e):
            continue

        pts = extract_polyline_points(e)
        poly = polygon_from_points(pts)
        if poly is None:
            continue

        area_mu2 = float(poly.area)
        perim_mu = float(poly.length)

        area_m2 = convert_area_to_m2(area_mu2, unit_factor_to_m, scale_1_to_x)
        perim_m = convert_model_to_m(perim_mu, unit_factor_to_m, scale_1_to_x)

        if area_m2 < min_area_m2 or area_m2 > max_area_m2:
            continue

        bounds = poly.bounds
        centroid = poly.centroid

        texts_inside = []
        for t, p in text_points:
            try:
                if poly.contains(p):
                    texts_inside.append(t.text)
            except Exception:
                continue

        candidate = RoomCandidate(
            candidate_id=candidate_id,
            source=e.dxftype(),
            layer=safe_layer(e),
            area_model_units2=round(area_mu2, 4),
            perimeter_model_units=round(perim_mu, 4),
            area_m2=round(area_m2, 4),
            perimeter_m=round(perim_m, 4),
            centroid_x=round(float(centroid.x), 4),
            centroid_y=round(float(centroid.y), 4),
            bbox_min_x=round(bounds[0], 4),
            bbox_min_y=round(bounds[1], 4),
            bbox_max_x=round(bounds[2], 4),
            bbox_max_y=round(bounds[3], 4),
            text_inside=texts_inside[:10],
            polygon_xy=[(round(x, 4), round(y, 4)) for x, y in pts],
        )
        candidates.append(candidate)
        candidate_id += 1

    candidates.sort(key=lambda c: c.area_m2, reverse=True)
    return candidates[:MAX_ROOM_CANDIDATES]


def extract_linear_stats(
    msp,
    unit_factor_to_m: float,
    scale_1_to_x: float,
) -> Dict[str, Any]:
    by_layer_length_m = defaultdict(float)
    total_length_m = 0.0
    entity_counts = Counter()

    for e in msp:
        dxftype = e.dxftype()
        entity_counts[dxftype] += 1
        layer = safe_layer(e)

        try:
            if dxftype == "LINE":
                p1 = (float(e.dxf.start.x), float(e.dxf.start.y))
                p2 = (float(e.dxf.end.x), float(e.dxf.end.y))
                length_mu = distance(p1, p2)
                length_m = convert_model_to_m(length_mu, unit_factor_to_m, scale_1_to_x)
                by_layer_length_m[layer] += length_m
                total_length_m += length_m

            elif dxftype in {"LWPOLYLINE", "POLYLINE"}:
                pts = extract_polyline_points(e)
                if len(pts) >= 2:
                    length_mu = linestring_length(pts)
                    if is_closed_polyline(e):
                        length_mu += distance(pts[-1], pts[0])
                    length_m = convert_model_to_m(length_mu, unit_factor_to_m, scale_1_to_x)
                    by_layer_length_m[layer] += length_m
                    total_length_m += length_m
        except Exception:
            continue

    return {
        "entity_counts": dict(entity_counts),
        "by_layer_length_m": {k: round(v, 3) for k, v in sorted(by_layer_length_m.items(), key=lambda x: x[1], reverse=True)},
        "total_length_m": round(total_length_m, 3),
    }


def classify_blocks_heuristic(inserts: List[InsertEntity]) -> Dict[str, Any]:
    counts = Counter()
    door_candidates = []
    window_candidates = []
    other_candidates = []

    for ins in inserts:
        name = ins.name.upper()
        layer = ins.layer.upper()
        joined = f"{name} {layer}"

        if any(k in joined for k in ["PORTA", "DOOR"]):
            counts["doors"] += 1
            door_candidates.append(asdict(ins))
        elif any(k in joined for k in ["JANELA", "WINDOW", "VENEZIANA", "MAXIMAR"]):
            counts["windows"] += 1
            window_candidates.append(asdict(ins))
        else:
            counts["other_blocks"] += 1
            other_candidates.append(asdict(ins))

    return {
        "counts": dict(counts),
        "door_candidates": door_candidates[:MAX_BLOCK_SAMPLES],
        "window_candidates": window_candidates[:MAX_BLOCK_SAMPLES],
        "other_candidates": other_candidates[:MAX_BLOCK_SAMPLES],
    }


def heuristic_wall_length_m(linear_stats: Dict[str, Any]) -> float:
    total = 0.0
    for layer, value in linear_stats["by_layer_length_m"].items():
        key = layer.upper()
        if any(k in key for k in ["PAREDE", "WALL", "ALVEN", "DIVIS", "PAINEL"]):
            total += value
    return round(total, 3)


# =========================================================
# IA
# =========================================================
def build_ai_payload(
    layers: List[str],
    texts: List[TextEntity],
    inserts_summary: Dict[str, Any],
    room_candidates: List[RoomCandidate],
    linear_stats: Dict[str, Any],
    wall_height_m: float,
    project_notes: str,
) -> Dict[str, Any]:
    text_samples = [asdict(t) for t in texts[:MAX_TEXT_SAMPLES]]
    room_payload = [asdict(r) for r in room_candidates]

    payload = {
        "project_type": "arquitetura",
        "project_notes": project_notes,
        "wall_height_m_default": wall_height_m,
        "layers": layers,
        "text_samples": text_samples,
        "blocks_summary": inserts_summary,
        "linear_stats": linear_stats,
        "heuristic_wall_length_m": heuristic_wall_length_m(linear_stats),
        "room_candidates": room_payload,
    }
    return payload


def get_ai_schema() -> Dict[str, Any]:
    return {
        "name": "dxf_architecture_quantities",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary": {"type": "string"},
                "assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "warnings": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "rooms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "candidate_id": {"type": "integer"},
                            "room_name": {"type": "string"},
                            "classification_confidence": {"type": "number"},
                            "rationale": {"type": "string"},
                            "include_in_takeoff": {"type": "boolean"},
                            "room_type": {"type": "string"},
                            "floor_m2": {"type": "number"},
                            "ceiling_m2": {"type": "number"},
                            "wall_perimeter_m": {"type": "number"},
                            "wall_area_m2": {"type": "number"},
                            "doors_count": {"type": "integer"},
                            "windows_count": {"type": "integer"},
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "candidate_id",
                            "room_name",
                            "classification_confidence",
                            "rationale",
                            "include_in_takeoff",
                            "room_type",
                            "floor_m2",
                            "ceiling_m2",
                            "wall_perimeter_m",
                            "wall_area_m2",
                            "doors_count",
                            "windows_count",
                            "notes",
                        ],
                    },
                },
                "global_quantities": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "floor_m2": {"type": "number"},
                        "ceiling_m2": {"type": "number"},
                        "wall_area_m2": {"type": "number"},
                        "wall_linear_m": {"type": "number"},
                        "doors_count": {"type": "integer"},
                        "windows_count": {"type": "integer"},
                        "other_elements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "name": {"type": "string"},
                                    "quantity": {"type": "number"},
                                    "unit": {"type": "string"},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["name", "quantity", "unit", "rationale"],
                            },
                        },
                    },
                    "required": [
                        "floor_m2",
                        "ceiling_m2",
                        "wall_area_m2",
                        "wall_linear_m",
                        "doors_count",
                        "windows_count",
                        "other_elements",
                    ],
                },
            },
            "required": [
                "summary",
                "assumptions",
                "warnings",
                "rooms",
                "global_quantities",
            ],
        },
    }


def run_openai_analysis(
    api_key: str,
    model_name: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)

    system_prompt = """
Você é um analista técnico de DXF para arquitetura.
Seu papel é classificar polígonos fechados como ambientes prováveis e gerar um quantitativo inicial.
Use obrigatoriamente os dados estruturados recebidos.
Não invente precisão inexistente.
Se houver dúvida, sinalize em warnings e assumptions.
Considere, como regra inicial:
- área de piso = área do ambiente incluído
- área de forro = área do ambiente incluído, salvo indicação contrária
- área de parede = perímetro do ambiente * wall_height_m_default
- portas e janelas podem ser inferidas por blocos, layers e textos, com cautela
Exclua da lista final elementos que pareçam carimbo, quadro, legenda, áreas externas técnicas ou regiões sem caráter de ambiente interno, quando isso for claro.
"""

    user_prompt = f"""
Analise o DXF resumido abaixo e devolva apenas JSON válido no schema pedido.

DADOS:
{json.dumps(payload, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": get_ai_schema(),
        },
        temperature=0.1,
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("A IA não retornou conteúdo.")
    return json.loads(content)


# =========================================================
# Visualização
# =========================================================
def render_room_preview(room_candidates: List[RoomCandidate], ai_result: Dict[str, Any]) -> Image.Image:
    room_map = {r.candidate_id: r for r in room_candidates}
    ai_rooms = ai_result.get("rooms", [])

    included_ids = set()
    names_by_id = {}
    for item in ai_rooms:
        cid = item["candidate_id"]
        names_by_id[cid] = item["room_name"]
        if item.get("include_in_takeoff", False):
            included_ids.add(cid)

    xs = []
    ys = []
    for r in room_candidates:
        for x, y in r.polygon_xy:
            xs.append(x)
            ys.append(y)

    if not xs or not ys:
        raise ValueError("Sem polígonos para renderizar.")

    fig_w = 10
    fig_h = 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for r in room_candidates:
        poly = np.array(r.polygon_xy)
        include = r.candidate_id in included_ids
        face = "#99d8c9" if include else "#d9d9d9"
        edge = "#238b45" if include else "#636363"
        patch = MplPolygon(poly, closed=True, facecolor=face, edgecolor=edge, alpha=0.55, linewidth=1.5)
        ax.add_patch(patch)

        label_name = names_by_id.get(r.candidate_id, f"Amb {r.candidate_id}")
        label = f"{r.candidate_id} | {label_name}\n{r.area_m2:.2f} m²"
        ax.text(
            r.centroid_x,
            r.centroid_y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )

    ax.set_title("Ambientes candidatos, leitura inicial com IA")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.invert_yaxis()
    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# =========================================================
# Exportação
# =========================================================
def to_excel_bytes(
    rooms_df: pd.DataFrame,
    global_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        rooms_df.to_excel(writer, sheet_name="ambientes", index=False)
        global_df.to_excel(writer, sheet_name="quantitativos", index=False)
        diagnostics_df.to_excel(writer, sheet_name="diagnostico", index=False)
    return buffer.getvalue()


# =========================================================
# App
# =========================================================
st.title("📐 DXF + IA, Quantitativos Arquitetônicos")
st.caption("MVP com DXF como entrada principal e IA obrigatória para classificar ambientes e gerar quantitativos iniciais.")

with st.sidebar:
    st.header("Entrada")
    uploaded_file = st.file_uploader("Envie o arquivo DXF", type=["dxf"])

    st.header("Modelo de IA")
    api_key = st.text_input("OPENAI_API_KEY", type="password")
    model_name = st.text_input("Modelo", value="gpt-4.1-mini")

    st.header("Escala e unidade")
    drawing_scale = st.number_input("Escala 1 : x", min_value=1.0, value=1.0, step=1.0)
    dxf_unit = st.selectbox("Unidade do DXF", options=["mm", "cm", "m"], index=0)

    st.header("Parâmetros técnicos")
    wall_height_m = st.number_input("Altura padrão de parede, m", min_value=1.0, value=3.0, step=0.1)
    min_area_m2 = st.number_input("Área mínima candidata, m²", min_value=0.1, value=1.0, step=0.1)
    max_area_m2 = st.number_input("Área máxima candidata, m²", min_value=1.0, value=5000.0, step=1.0)

    st.header("Contexto opcional")
    project_notes = st.text_area(
        "Observações do projeto",
        value="Foco em arquitetura. Priorizar ambientes internos, piso, forro, paredes, portas e janelas.",
        height=120,
    )

    analyze_button = st.button("Analisar com IA", use_container_width=True)

if uploaded_file is None:
    st.info("Envie um DXF para começar.")
    st.stop()

if not api_key:
    st.warning("Informe a OPENAI_API_KEY para executar a análise, porque nesta versão o uso da IA é obrigatório.")
    st.stop()

unit_map = {"mm": 0.001, "cm": 0.01, "m": 1.0}
unit_factor_to_m = unit_map[dxf_unit]

try:
    doc = load_dxf_document(uploaded_file)
    msp = get_modelspace_entities(doc)
    layers = extract_layers(doc)
    texts = extract_text_entities(msp)
    inserts = extract_insert_entities(msp)
    linear_stats = extract_linear_stats(msp, unit_factor_to_m, drawing_scale)
    insert_summary = classify_blocks_heuristic(inserts)
    room_candidates = extract_room_candidates(
        msp=msp,
        texts=texts,
        unit_factor_to_m=unit_factor_to_m,
        scale_1_to_x=drawing_scale,
        min_area_m2=min_area_m2,
        max_area_m2=max_area_m2,
    )
except Exception as e:
    st.error(f"Erro ao ler o DXF: {e}")
    st.stop()

st.subheader("Resumo bruto do DXF")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Layers", len(layers))
c2.metric("Textos", len(texts))
c3.metric("Blocos INSERT", len(inserts))
c4.metric("Ambientes candidatos", len(room_candidates))

payload = build_ai_payload(
    layers=layers,
    texts=texts,
    inserts_summary=insert_summary,
    room_candidates=room_candidates,
    linear_stats=linear_stats,
    wall_height_m=wall_height_m,
    project_notes=project_notes,
)

if analyze_button:
    try:
        with st.spinner("Lendo o DXF, organizando a geometria e pedindo a interpretação da IA..."):
            ai_result = run_openai_analysis(
                api_key=api_key,
                model_name=model_name,
                payload=payload,
            )

        preview_img = None
        if room_candidates:
            preview_img = render_room_preview(room_candidates, ai_result)

        rooms_df = pd.DataFrame(ai_result["rooms"])
        global_q = ai_result["global_quantities"]
        global_rows = [
            {"item": "piso_total", "quantidade": global_q["floor_m2"], "unidade": "m²"},
            {"item": "forro_total", "quantidade": global_q["ceiling_m2"], "unidade": "m²"},
            {"item": "parede_area_total", "quantidade": global_q["wall_area_m2"], "unidade": "m²"},
            {"item": "parede_linear_total", "quantidade": global_q["wall_linear_m"], "unidade": "m"},
            {"item": "portas_total", "quantidade": global_q["doors_count"], "unidade": "un"},
            {"item": "janelas_total", "quantidade": global_q["windows_count"], "unidade": "un"},
        ]
        for item in global_q.get("other_elements", []):
            global_rows.append(
                {
                    "item": item["name"],
                    "quantidade": item["quantity"],
                    "unidade": item["unit"],
                }
            )
        global_df = pd.DataFrame(global_rows)

        diagnostics_rows = []
        for w in ai_result.get("warnings", []):
            diagnostics_rows.append({"tipo": "warning", "mensagem": w})
        for a in ai_result.get("assumptions", []):
            diagnostics_rows.append({"tipo": "assumption", "mensagem": a})
        diagnostics_df = pd.DataFrame(diagnostics_rows)

        excel_bytes = to_excel_bytes(rooms_df, global_df, diagnostics_df)
        json_bytes = json.dumps(ai_result, ensure_ascii=False, indent=2).encode("utf-8")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Planta interpretada", "Ambientes", "Quantitativos", "Diagnóstico da IA"]
        )
            
        with tab1:
            if preview_img is not None:
                st.image(preview_img, use_container_width=True)
            else:
                st.warning("Nenhum polígono fechado candidato foi encontrado para renderização.")
                st.write(ai_result["summary"])

        with tab2:
            st.dataframe(rooms_df, use_container_width=True)
            st.download_button(
                "Baixar JSON da análise",
                data=json_bytes,
                file_name="analise_dxf_ia.json",
                mime="application/json",
            )

        with tab3:
            st.dataframe(global_df, use_container_width=True)
            st.download_button(
                "Baixar Excel",
                data=excel_bytes,
                file_name="quantitativos_dxf_ia.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with tab4:
            if diagnostics_df.empty:
                st.success("A IA não registrou warnings nem assumptions.")
            else:
                st.dataframe(diagnostics_df, use_container_width=True)

        with st.expander("Ver resumo bruto enviado para a IA"):
            st.json(payload)

    except Exception as e:
        st.error(f"Erro na análise com IA: {e}")

else:
    st.info("Clique em “Analisar com IA” para gerar a interpretação inicial do DXF.")
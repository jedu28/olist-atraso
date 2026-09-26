"""Gera as figuras da documentação (EN e PT) e os cards para o LinkedIn.

    python scripts/make_figures.py            # recalcula tudo (~5 min)
    python scripts/make_figures.py --cached   # reaproveita os resultados já calculados

Saídas:
    docs/figures/{en,pt}/*.png   figuras da documentação
    docs/figures/linkedin/*.png  cards 1080×1350 (4:5) para o post
    docs/results.json            números usados no texto da documentação
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, PercentFormatter
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_predict, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    FIGURES_DIR,
    MIN_RECALL,
    N_TEMPORAL_SPLITS,
    PROCESSED_DIR,
    RANDOM_STATE,
    ROOT_DIR,
    TARGET,
    TIMESTAMP_COLUMN,
)
from src.data import load_raw
from src.features import add_seller_history, build_dataset, build_target
from src.metrics import classification_metrics
from src.modeling import build_tuned_xgb_pipeline, build_xgb_pipeline, split_xy, temporal_split
from src.plotting import (
    BLUE,
    CLASS_COLORS,
    DEEMPHASIS,
    GRID,
    INK,
    INK_2,
    METRIC_COLORS,
    MUTED,
    SURFACE,
    VERSION_COLORS,
    apply_style,
)
from src.thresholds import choose_threshold, temporal_oof_proba

CACHE_PATH = PROCESSED_DIR / "figure_results.joblib"
RESULTS_JSON = ROOT_DIR / "docs" / "results.json"
PCT = PercentFormatter(1, decimals=0)

# ---------------------------------------------------------------------------
# Textos
# ---------------------------------------------------------------------------
T = {
    "en": {
        "month_title": "Late-delivery rate by purchase month",
        "month_test": "test period",
        "month_avg": "average",
        "dist_title": "Where the model puts late and on-time orders (test set)",
        "dist_x": "predicted probability of delay",
        "dist_y": "share of orders (log scale)",
        "on_time": "on time",
        "late": "late",
        "dist_note": "≥ 0.5 catches {:.1%} of late orders",
        "chosen": "chosen",
        "tradeoff_title": "Precision, recall and F1 by threshold (temporal out-of-fold, train set)",
        "threshold": "threshold",
        "precision": "Precision",
        "recall": "Recall",
        "flagged": "Orders flagged",
        "min_recall": "minimum recall = {:.0%}",
        "promise_title": "Recall promised on train vs. delivered on test",
        "promised": "promised (train, out-of-fold)",
        "delivered": "delivered (test)",
        "random_oof": "Random CV",
        "temporal_oof": "Temporal CV",
        "versions_title": "Same ranking, different decisions: test-set metrics by version",
        "v_names": {"V1": "V1 · default, 0.5", "V2": "V2 · tuned, 0.5", "V3": "V3 · tuned, random-CV cut",
                    "V4": "V4 · tuned, temporal-CV cut"},
        "monthly_title": "The same threshold behaves differently each month (V4, test set)",
        "late_rate": "Late rate",
        "leak_title": "Seller late-rate feature on its own (ROC-AUC, train set)",
        "leak_before": "Counts orders\nstill in transit\n(leaky)",
        "leak_after": "Only outcomes known\nat purchase time\n(point-in-time)",
        "gap_title": "How optimistic is each validation? (ROC-AUC)",
        "gap_random": "Random CV\n(train)",
        "gap_temporal": "Temporal CV\n(train)",
        "gap_test": "Test\n(future months)",
        "base_rate": "dotted line = base rate",
        "tuning_title": "Tuning improved validation, not the future (ROC-AUC)",
        "tuning_cv": "Temporal CV (train)",
        "tuning_test": "Test (future months)",
        "tuning_default": "XGBoost defaults",
        "tuning_tuned": "Optuna-tuned",
    },
    "pt": {
        "month_title": "Taxa de atraso por mês de compra",
        "month_test": "período de teste",
        "month_avg": "média",
        "dist_title": "Onde o modelo coloca pedidos atrasados e no prazo (teste)",
        "dist_x": "probabilidade prevista de atraso",
        "dist_y": "fração dos pedidos (escala log)",
        "on_time": "no prazo",
        "late": "atrasado",
        "dist_note": "≥ 0.5 pega {:.1%} dos atrasos",
        "chosen": "escolhido",
        "tradeoff_title": "Precisão, recall e F1 por threshold (out-of-fold temporal, treino)",
        "threshold": "threshold",
        "precision": "Precisão",
        "recall": "Recall",
        "flagged": "Pedidos sinalizados",
        "min_recall": "recall mínimo = {:.0%}",
        "promise_title": "Recall prometido no treino × entregue no teste",
        "promised": "prometido (treino, out-of-fold)",
        "delivered": "entregue (teste)",
        "random_oof": "CV aleatória",
        "temporal_oof": "CV temporal",
        "versions_title": "Mesmo ranking, decisões diferentes: métricas no teste por versão",
        "v_names": {"V1": "V1 · padrão, 0.5", "V2": "V2 · Optuna, 0.5", "V3": "V3 · Optuna, corte CV aleatória",
                    "V4": "V4 · Optuna, corte CV temporal"},
        "monthly_title": "O mesmo threshold se comporta diferente a cada mês (V4, teste)",
        "late_rate": "Taxa de atraso",
        "leak_title": "Feature de taxa de atraso do vendedor, sozinha (ROC-AUC, treino)",
        "leak_before": "Conta pedidos\nainda em trânsito\n(vazada)",
        "leak_after": "Só desfechos conhecidos\nna data da compra\n(point-in-time)",
        "gap_title": "Quão otimista é cada validação? (ROC-AUC)",
        "gap_random": "CV aleatória\n(treino)",
        "gap_temporal": "CV temporal\n(treino)",
        "gap_test": "Teste\n(meses futuros)",
        "base_rate": "pontilhado = taxa base",
        "tuning_title": "O tuning melhorou a validação, não o futuro (ROC-AUC)",
        "tuning_cv": "CV temporal (treino)",
        "tuning_test": "Teste (meses futuros)",
        "tuning_default": "XGBoost padrão",
        "tuning_tuned": "Optuna",
    },
}


# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------
def _seller_rate_by_purchase(df: pd.DataFrame) -> pd.Series:
    """Versão antiga (vazada) da taxa do vendedor: conta todo pedido *comprado* antes."""
    df = df.sort_values(TIMESTAMP_COLUMN)
    grouped = df.groupby("seller_id")[TARGET]
    anteriores = grouped.cumcount()
    return ((grouped.cumsum() - df[TARGET]) / anteriores).fillna(0.0)


def seller_leak_auc() -> dict[str, float]:
    raw = load_raw()
    orders = build_target(raw["orders"]).merge(
        raw["items"].groupby("order_id")["seller_id"].first().reset_index(), on="order_id"
    )
    orders = add_seller_history(orders)  # ordena por data da compra
    orders["taxa_vazada"] = _seller_rate_by_purchase(orders)
    treino, _ = temporal_split(orders)
    return {
        "leaky": roc_auc_score(treino[TARGET], treino["taxa_vazada"]),
        "point_in_time": roc_auc_score(treino[TARGET], treino["seller_taxa_atraso_historica"]),
    }


def compute_results() -> dict:
    df = build_dataset()
    train_df, test_df = temporal_split(df)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    print("V1 (XGBoost padrão)...")
    proba_v1 = build_xgb_pipeline().fit(X_train, y_train).predict_proba(X_test)[:, 1]

    tuned = build_tuned_xgb_pipeline()
    print("OOF aleatório (10 folds)...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    oof_random = cross_val_predict(tuned, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    print("OOF temporal...")
    oof_temporal, y_oof_temporal = temporal_oof_proba(tuned, X_train, y_train)
    print("V2 (XGBoost Optuna)...")
    proba_v2 = tuned.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    print("CV temporal: padrão x tunado...")
    cv_temporal = TimeSeriesSplit(n_splits=N_TEMPORAL_SPLITS)
    cv_temporal_auc = {
        nome: cross_val_score(pipe, X_train, y_train, cv=cv_temporal, scoring="roc_auc").mean()
        for nome, pipe in [("default", build_xgb_pipeline()), ("tuned", build_tuned_xgb_pipeline())]
    }
    print("Vazamento da feature do vendedor...")
    leak = seller_leak_auc()

    monthly = df.set_index(TIMESTAMP_COLUMN)[TARGET].resample("MS").agg(["mean", "count"])
    return {
        "monthly": monthly[monthly["count"] >= 100],
        "test_start": test_df[TIMESTAMP_COLUMN].min(),
        "test_month": test_df[TIMESTAMP_COLUMN].dt.to_period("M").astype(str).to_numpy(),
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "proba_v1": proba_v1,
        "proba_v2": proba_v2,
        "oof_random": oof_random,
        "oof_temporal": oof_temporal,
        "y_oof_temporal": y_oof_temporal,
        "leak": leak,
        "cv_temporal_auc": cv_temporal_auc,
        "periods": {
            "train": [str(train_df[TIMESTAMP_COLUMN].min().date()), str(train_df[TIMESTAMP_COLUMN].max().date())],
            "test": [str(test_df[TIMESTAMP_COLUMN].min().date()), str(test_df[TIMESTAMP_COLUMN].max().date())],
        },
        "n_orders": len(df),
    }


def derive(r: dict) -> dict:
    """Thresholds, métricas por versão e por mês a partir das probabilidades."""
    t_v3 = choose_threshold(r["y_train"], r["oof_random"])
    t_v4 = choose_threshold(r["y_oof_temporal"], r["oof_temporal"])
    versions = {
        "V1": (r["proba_v1"], 0.5),
        "V2": (r["proba_v2"], 0.5),
        "V3": (r["proba_v2"], t_v3),
        "V4": (r["proba_v2"], t_v4),
    }
    test = {v: classification_metrics(r["y_test"], p, t) for v, (p, t) in versions.items()}
    promise = {
        "V3": {"promised": classification_metrics(r["y_train"], r["oof_random"], t_v3)["recall"],
               "delivered": test["V3"]["recall"]},
        "V4": {"promised": classification_metrics(r["y_oof_temporal"], r["oof_temporal"], t_v4)["recall"],
               "delivered": test["V4"]["recall"]},
    }
    months = {}
    for mes in np.unique(r["test_month"]):
        mask = r["test_month"] == mes
        if mask.sum() >= 1000:  # maio/2018 só tem os últimos dias no teste
            months[mes] = classification_metrics(r["y_test"][mask], r["proba_v2"][mask], t_v4)
    return {
        "thresholds": {"V3": t_v3, "V4": t_v4},
        "test": test,
        "promise": promise,
        "months": months,
        "auc": {
            "random_oof": roc_auc_score(r["y_train"], r["oof_random"]),
            "temporal_oof": roc_auc_score(r["y_oof_temporal"], r["oof_temporal"]),
            "test": test["V2"]["roc_auc"],
        },
    }


# ---------------------------------------------------------------------------
# Gráficos (cada função desenha num Axes/Figure recebido)
# ---------------------------------------------------------------------------
def _clean_x(ax):
    ax.grid(axis="x", visible=False)


def draw_monthly(ax, r, d, t):
    m = r["monthly"]
    ax.axvspan(r["test_start"], m.index.max() + pd.Timedelta(days=28), color=GRID, alpha=0.6, zorder=0, lw=0)
    ax.bar(m.index, m["mean"], width=20, align="edge", color=BLUE)
    media = (m["mean"] * m["count"]).sum() / m["count"].sum()
    ax.axhline(media, color=INK_2, linewidth=1)
    ax.text(m.index.min(), media, f" {t['month_avg']} {media:.1%}", color=INK_2, va="bottom")
    ax.text(r["test_start"] + pd.Timedelta(days=4), m["mean"].max() * 1.12, t["month_test"], color=INK_2, va="top")
    pico = m["mean"].idxmax()
    ax.annotate(f"{m['mean'].max():.1%}", (pico + pd.Timedelta(days=10), m["mean"].max()), xytext=(0, 4),
                textcoords="offset points", ha="center", color=INK)
    ax.set_ylim(0, m["mean"].max() * 1.15)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title(t["month_title"])
    _clean_x(ax)


def _label(ax, x, y, texto, **kwargs):
    """Texto com fundo da cor da superfície, para não brigar com as linhas embaixo."""
    ax.text(x, y, texto, color=INK, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2}, **kwargs)


def draw_distribution(ax, r, d, t):
    y, p = r["y_test"], r["proba_v2"]
    bins = np.linspace(0, 1, 41)
    for classe, nome in [(0, t["on_time"]), (1, t["late"])]:
        pesos = np.ones((y == classe).sum()) / (y == classe).sum()
        ax.hist(p[y == classe], bins=bins, weights=pesos, histtype="step", linewidth=2,
                color=CLASS_COLORS[classe], label=nome)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}%"))
    ax.axvline(0.5, color=INK, linestyle="--", linewidth=1)
    t4 = d["thresholds"]["V4"]
    ax.axvline(t4, color=VERSION_COLORS["V4"], linewidth=2)
    xt = ax.get_xaxis_transform()
    _label(ax, t4 + 0.01, 0.06, f"{t['chosen']}: {t4:.2f}", transform=xt)
    _label(ax, 0.51, 0.06, t["dist_note"].format(d["test"]["V2"]["recall"]), transform=xt)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel(t["dist_x"])
    ax.set_ylabel(t["dist_y"])
    ax.set_title(t["dist_title"])
    ax.legend(loc="upper right")


def draw_tradeoff(ax, r, d, t):
    prec, rec, cuts = precision_recall_curve(r["y_oof_temporal"], r["oof_temporal"])
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(rec), where=(prec + rec) > 0)
    visivel = cuts <= 0.7
    series = [(t["precision"], prec, METRIC_COLORS["Precisão"]), (t["recall"], rec, METRIC_COLORS["Recall"]),
              ("F1", f1, METRIC_COLORS["F1"])]
    for rotulo, valores, cor in series:
        ax.plot(cuts[visivel], valores[visivel], color=cor, label=rotulo)
    ax.axhline(MIN_RECALL, color=INK_2, linewidth=1, linestyle=":")
    t4 = d["thresholds"]["V4"]
    ax.text(0.01, MIN_RECALL - 0.015, t["min_recall"].format(MIN_RECALL), color=INK_2, va="top")
    for corte, rotulo, cor, estilo in [(0.5, "0.5", INK, "--"), (t4, f"{t['chosen']} {t4:.2f}", VERSION_COLORS["V4"], "-")]:
        ax.axvline(corte, color=cor, linestyle=estilo, linewidth=1.5)
        ax.text(corte - 0.008, 0.98, rotulo, color=INK, va="top", ha="right", rotation=90,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel(t["threshold"])
    ax.set_title(t["tradeoff_title"])
    ax.legend(loc="upper right")


def draw_promise(ax, r, d, t, big=False):
    s = 1.6 if big else 1
    linhas = [("V3", t["random_oof"]), ("V4", t["temporal_oof"])]
    for i, (v, _) in enumerate(linhas):
        prometido, entregue = d["promise"][v]["promised"], d["promise"][v]["delivered"]
        ax.plot([prometido, entregue], [i, i], color=GRID, linewidth=4 * s, zorder=1, solid_capstyle="round")
        ax.scatter(prometido, i, s=110 * s**2, facecolor=SURFACE, edgecolor=VERSION_COLORS[v], linewidth=2.5, zorder=2)
        ax.scatter(entregue, i, s=110 * s**2, color=VERSION_COLORS[v], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{prometido:.0%}", (prometido, i), xytext=(0, 14 * s), textcoords="offset points",
                    ha="center", color=INK_2)
        ax.annotate(f"{entregue:.0%}", (entregue, i), xytext=(0, 14 * s), textcoords="offset points",
                    ha="center", color=INK, fontweight="bold")
    ax.axvline(MIN_RECALL, color=INK_2, linewidth=1, linestyle=":")
    ax.set_yticks(range(len(linhas)), [nome for _, nome in linhas])
    ax.set_ylim(-0.6, len(linhas) - 0.3)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel(t["recall"])
    ax.grid(axis="y", visible=False)
    ax.set_title(t["promise_title"])
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_2, markersize=9,
               markeredgewidth=2, label=t["promised"]),
        Line2D([], [], marker="o", linestyle="", color=INK_2, markersize=9, label=t["delivered"]),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.18 if not big else -0.12), ncol=2)


def draw_versions(axes, r, d, t):
    metricas = [("recall", t["recall"]), ("precision", t["precision"]), ("flagged_rate", t["flagged"])]
    versoes = list(d["test"])
    for ax, (chave, titulo) in zip(axes, metricas):
        valores = [d["test"][v][chave] for v in versoes]
        barras = ax.barh(versoes[::-1], valores[::-1], height=0.6, color=[VERSION_COLORS[v] for v in versoes[::-1]])
        ax.bar_label(barras, labels=[f"{x:.1%}" for x in valores[::-1]], padding=4, color=INK)
        ax.set_xlim(0, max(valores) * 1.35)
        ax.xaxis.set_major_formatter(PCT)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(titulo)
        ax.grid(axis="y", visible=False)
        if chave == "precision":
            base = d["test"]["V1"]["base_rate"]
            ax.axvline(base, color=INK_2, linewidth=1, linestyle=":")
            ax.set_title(f"{titulo}  ·  {t['base_rate']} {base:.1%}")
    for ax in axes[1:]:
        ax.set_yticklabels([])


def draw_monthly_stability(axes, r, d, t):
    meses = list(d["months"])
    x = np.arange(len(meses))
    for ax, (chave, titulo) in zip(axes, [("base_rate", t["late_rate"]), ("recall", t["recall"]),
                                          ("flagged_rate", t["flagged"])]):
        valores = [d["months"][m][chave] for m in meses]
        cor = CLASS_COLORS[1] if chave == "base_rate" else VERSION_COLORS["V4"]
        barras = ax.bar(x, valores, width=0.55, color=cor)
        fmt = "{:.1%}" if chave == "base_rate" else "{:.0%}"
        ax.bar_label(barras, labels=[fmt.format(v) for v in valores], padding=3, color=INK)
        ax.set_xticks(x, meses)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_ylim(0, 1 if chave != "base_rate" else max(valores) * 1.3)
        ax.set_title(titulo)
        _clean_x(ax)
    axes[1].axhline(MIN_RECALL, color=INK_2, linewidth=1, linestyle=":")


def _auc_bars(ax, nomes, valores, cores, teto=None):
    """Barras de ROC-AUC crescendo a partir de 0.5 (o acaso), não de um eixo truncado."""
    barras = ax.bar(nomes, np.array(valores) - 0.5, bottom=0.5, width=0.5, color=cores)
    ax.bar_label(barras, labels=[f"{v:.3f}" for v in valores], padding=4, color=INK, fontweight="bold")
    ax.set_ylim(0.5, teto or max(valores) + 0.05)
    _clean_x(ax)
    return barras


def draw_leak(ax, r, d, t):
    _auc_bars(ax, [t["leak_before"], t["leak_after"]], [r["leak"]["leaky"], r["leak"]["point_in_time"]],
              [DEEMPHASIS, BLUE])
    ax.set_title(t["leak_title"])


def draw_gap(ax, r, d, t):
    a = d["auc"]
    _auc_bars(ax, [t["gap_random"], t["gap_temporal"], t["gap_test"]],
              [a["random_oof"], a["temporal_oof"], a["test"]], [DEEMPHASIS, BLUE, INK_2], teto=0.88)
    ax.set_title(t["gap_title"])


def draw_tuning(ax, r, d, t):
    grupos = [t["tuning_cv"], t["tuning_test"]]
    padrao = [r["cv_temporal_auc"]["default"], d["test"]["V1"]["roc_auc"]]
    tunado = [r["cv_temporal_auc"]["tuned"], d["test"]["V2"]["roc_auc"]]
    x = np.arange(len(grupos))
    largura = 0.34
    for desloc, valores, cor, nome in [(-largura / 2, padrao, VERSION_COLORS["V1"], t["tuning_default"]),
                                       (largura / 2, tunado, VERSION_COLORS["V2"], t["tuning_tuned"])]:
        barras = ax.bar(x + desloc, np.array(valores) - 0.5, bottom=0.5, width=largura - 0.02, color=cor, label=nome)
        ax.bar_label(barras, labels=[f"{v:.3f}" for v in valores], padding=4, color=INK, fontweight="bold")
    ax.set_xticks(x, grupos)
    ax.set_ylim(0.5, 0.82)
    ax.set_title(t["tuning_title"])
    ax.legend(loc="upper left")
    _clean_x(ax)


# ---------------------------------------------------------------------------
# Saída
# ---------------------------------------------------------------------------
def doc_figures(r, d, lang, out_dir: Path):
    t = T[lang]
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style(font_size=10)

    def save(fig, nome):
        fig.savefig(out_dir / f"{nome}.png", dpi=160)
        plt.close(fig)

    for nome, desenhar, tamanho in [
        ("01_late_rate_by_month", draw_monthly, (10, 3.6)),
        ("02_score_distribution", draw_distribution, (10, 4)),
        ("03_threshold_tradeoff", draw_tradeoff, (10, 4.2)),
        ("04_promised_vs_delivered", draw_promise, (8, 3.4)),
        ("07_seller_feature_leak", draw_leak, (6.5, 3.8)),
        ("08_validation_gap", draw_gap, (6.5, 3.8)),
        ("09_tuning_transfer", draw_tuning, (7.5, 3.8)),
    ]:
        fig, ax = plt.subplots(figsize=tamanho)
        desenhar(ax, r, d, t)
        save(fig, nome)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"wspace": 0.15})
    draw_versions(axes, r, d, t)
    axes[0].set_yticklabels([t["v_names"][v] for v in list(d["test"])[::-1]])
    fig.suptitle(t["versions_title"], x=0.01, ha="left", fontweight="bold", y=1.04)
    save(fig, "05_versions_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    draw_monthly_stability(axes, r, d, t)
    fig.suptitle(t["monthly_title"], x=0.01, ha="left", fontweight="bold", y=1.04)
    save(fig, "06_monthly_stability")


CARD_W, CARD_H, CARD_DPI = 7.2, 9, 150  # 1080×1350 px (4:5, formato retrato do feed)


def _card(headline: str, sub: str, footer: str = "Olist public e-commerce data · 96k orders · XGBoost"):
    """Figura 1080×1350 com título grande e subtítulo no topo; o gráfico vem abaixo de ~0.70."""
    apply_style(font_size=15)
    plt.rcParams["savefig.bbox"] = "standard"  # tamanho fixo: nada de recorte automático
    fig = plt.figure(figsize=(CARD_W, CARD_H), dpi=CARD_DPI)
    linhas_titulo = headline.count("\n") + 1
    fig.text(0.07, 0.945, headline, fontsize=27, fontweight="bold", color=INK, va="top", linespacing=1.15)
    fig.text(0.07, 0.945 - 0.052 * linhas_titulo - 0.018, sub, fontsize=14.5, color=INK_2, va="top",
             linespacing=1.45)
    fig.text(0.07, 0.03, footer, fontsize=11, color=MUTED)
    return fig


def linkedin_cards(r, d, out_dir: Path):
    t = T["en"]
    out_dir.mkdir(parents=True, exist_ok=True)
    v2, v4 = d["test"]["V2"], d["test"]["V4"]

    def save(fig, nome):
        fig.savefig(out_dir / f"{nome}.png", dpi=CARD_DPI)
        plt.close(fig)

    # 1 — capa: o número principal
    fig = _card("Predicting late deliveries\nbefore they happen",
                "Same model, same scores. The only thing that changed\nwas where I drew the line.")
    fig.text(0.07, 0.68, "Share of late orders caught on unseen future months", fontsize=14.5, color=INK_2)
    ax = fig.add_axes([0.07, 0.40, 0.86, 0.25])
    barras_info = [("Default cut (0.5)", v2["recall"], DEEMPHASIS), (f"Chosen cut ({v4['threshold']:.2f})", v4["recall"], VERSION_COLORS["V4"])]
    for y, (rotulo, valor, cor) in zip([1, 0], barras_info):
        ax.barh(y, valor, height=0.42, color=cor)
        ax.text(0, y + 0.3, rotulo, fontsize=15, color=INK, va="bottom")
        ax.text(valor + 0.02, y, f"{valor:.1%}", fontsize=30, fontweight="bold", color=INK, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.35, 1.65)
    ax.axis("off")
    for i, (valor, rotulo) in enumerate([
        (f"{v4['tp']:,}", f"of {v4['tp'] + v4['fn']:,} late orders\ncaught in the test months"),
        (f"{v4['precision'] / v4['base_rate']:.1f}×", "the base late rate among\nflagged orders"),
        (f"{v4['flagged_rate']:.0%}", "of orders flagged\nfor proactive action"),
    ]):
        x = 0.07 + i * 0.30
        fig.text(x, 0.30, valor, fontsize=28, fontweight="bold", color=INK, va="top")
        fig.text(x, 0.235, rotulo, fontsize=12.5, color=INK_2, va="top", linespacing=1.4)
    save(fig, "01_cover")

    # 2 — prometido x entregue
    fig = _card("A threshold picked on\nrandom CV breaks in production",
                "Both cuts promised 50% recall on the training data.\nOnly the time-aware one kept the promise.")
    ax = fig.add_axes([0.24, 0.25, 0.68, 0.4])
    draw_promise(ax, r, d, t, big=True)
    ax.set_title("")
    save(fig, "02_promised_vs_delivered")

    # 3 — trade-off do threshold
    fig = _card("0.5 is not a neutral default",
                f"With ~{r['y_train'].mean():.0%} late orders, few scores ever cross 0.5.\n"
                "Precision, recall and F1 across every possible cut:")
    ax = fig.add_axes([0.13, 0.12, 0.8, 0.58])
    draw_tradeoff(ax, r, d, t)
    ax.set_title("")
    save(fig, "03_threshold_tradeoff")

    # 4 — por que validação temporal
    m = r["monthly"]["mean"]
    fig = _card("Delays are not stationary",
                f"The late-delivery rate swings from {m.min():.1%} to {m.max():.1%} month to month.\n"
                "So every validation step here respects time.")
    ax = fig.add_axes([0.13, 0.13, 0.8, 0.56])
    draw_monthly(ax, r, d, t)
    ax.set_title("")
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    save(fig, "04_late_rate_by_month")

    # 5 — o vazamento
    fig = _card("The leak I found\nin my own feature",
                "\"Seller's historical late rate\" counted orders still in transit\n"
                "at purchase time. Fixed, it looks weaker, and it's honest.")
    ax = fig.add_axes([0.13, 0.15, 0.8, 0.46])
    draw_leak(ax, r, d, t)
    ax.set_title("ROC-AUC of the feature on its own (0.5 = coin flip)", fontsize=13.5, color=INK_2,
                 fontweight="normal")
    save(fig, "05_feature_leak")

    # 6 — o tuning que não se transferiu
    fig = _card("Tuning won the validation,\nlost the future",
                "Optuna improved the temporal CV score, yet the untuned\nmodel ranked better on the unseen months.")
    ax = fig.add_axes([0.13, 0.15, 0.8, 0.46])
    draw_tuning(ax, r, d, t)
    ax.set_title("ROC-AUC (0.5 = coin flip)", fontsize=13.5, color=INK_2, fontweight="normal")
    save(fig, "06_tuning_transfer")


def export_results(r, d):
    resumo = {
        "n_orders": r["n_orders"],
        "periods": r["periods"],
        "late_rate": {"train": float(r["y_train"].mean()), "test": float(r["y_test"].mean())},
        "thresholds": d["thresholds"],
        "test_metrics": d["test"],
        "promise": d["promise"],
        "auc": d["auc"],
        "months_v4": d["months"],
        "seller_feature_auc": r["leak"],
        "cv_temporal_auc": r["cv_temporal_auc"],
    }
    RESULTS_JSON.write_text(json.dumps(resumo, indent=2, default=float) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cached", action="store_true", help="reaproveita resultados já calculados")
    args = parser.parse_args()

    if args.cached and CACHE_PATH.exists():
        r = joblib.load(CACHE_PATH)
    else:
        r = compute_results()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(r, CACHE_PATH)
    d = derive(r)

    for lang in ("en", "pt"):
        doc_figures(r, d, lang, FIGURES_DIR / lang)
    linkedin_cards(r, d, FIGURES_DIR / "linkedin")
    export_results(r, d)
    print(f"Figuras em {FIGURES_DIR} | números em {RESULTS_JSON}")


if __name__ == "__main__":
    main()

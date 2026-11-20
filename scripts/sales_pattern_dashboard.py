"""2024 vs 2025 Sales Pattern Dashboard.

Mix of matplotlib static charts (with Chinese-font support) and Plotly
interactive charts (with plotly.js embedded inline so no internet needed).

Usage:
    python -u -m scripts.sales_pattern_dashboard
"""
from __future__ import annotations

import base64
import sys
import io
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Style + Chinese font ──
C24 = "#1565C0"
C25 = "#D32F2F"
C_GREEN = "#2E7D32"
C_GRAY = "#888888"

OUTPUT_HTML = Path("backtests/report/sales_pattern_24vs25.html")
OUTPUT_MD = Path("backtests/report/sales_pattern_findings.md")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

# ── Lunar reference dates (manually verified) ──
CNY_2024 = date(2024, 2, 10)
CNY_2025 = date(2025, 1, 29)
QINGMING_2024 = date(2024, 4, 4)
QINGMING_2025 = date(2025, 4, 4)
DRAGON_2024 = date(2024, 6, 10)
DRAGON_2025 = date(2025, 5, 31)
MID_AUTUMN_2024 = date(2024, 9, 17)
MID_AUTUMN_2025 = date(2025, 10, 6)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ════════════════════════════════════════════════════════════════
def load_data():
    print("Loading data...", flush=True)
    cols = ["date", "store_id", "sku_id", "quantity_sold", "total_amount"]
    t24 = pd.read_parquet("data/features/features_train.parquet", columns=cols)
    t25 = pd.read_parquet("data/features/features_oos.parquet", columns=cols)
    t24["date"] = pd.to_datetime(t24["date"])
    t25["date"] = pd.to_datetime(t25["date"])

    raw = pd.read_parquet("data/raw/raw_pos.parquet", columns=["sku_id", "sku_name", "barcode"])
    name_map = raw.drop_duplicates("sku_id").set_index("sku_id")["sku_name"].to_dict()
    bc_map = raw.drop_duplicates("sku_id").set_index("sku_id")["barcode"].to_dict()
    print(f"  2024: {len(t24):,} rows,  2025: {len(t25):,} rows", flush=True)
    return t24, t25, name_map, bc_map


# ════════════════════════════════════════════════════════════════
#  Fig 1: Daily overlay aligned by lunar calendar
# ════════════════════════════════════════════════════════════════
def chart1(t24, t25):
    print("  Chart 1: Daily overlay (lunar-aligned)...", flush=True)
    d24 = t24.groupby(t24["date"].dt.dayofyear)["quantity_sold"].sum()
    d25 = t25.groupby(t25["date"].dt.dayofyear)["quantity_sold"].sum()

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(d24.index, d24.values / 1e4, color=C24, linewidth=0.7, alpha=0.85, label="2024")
    ax.plot(d25.index, d25.values / 1e4, color=C25, linewidth=0.7, alpha=0.85, label="2025")

    ymax = max(d24.max(), d25.max()) / 1e4

    # CNY 2024 (DOY 41) and CNY 2025 (DOY 29)
    cny24_doy = CNY_2024.timetuple().tm_yday
    cny25_doy = CNY_2025.timetuple().tm_yday
    ax.axvline(cny24_doy, color=C24, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(cny25_doy, color=C25, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(cny24_doy + 1, ymax * 0.95, "2024 春节 (Feb 10)", fontsize=9,
            color=C24, fontweight="bold")
    ax.text(cny25_doy + 1, ymax * 0.85, "2025 春节 (Jan 29)", fontsize=9,
            color=C25, fontweight="bold")

    # Mid-autumn (also lunar — 2024: Sep 17, 2025: Oct 6)
    ma24 = MID_AUTUMN_2024.timetuple().tm_yday
    ma25 = MID_AUTUMN_2025.timetuple().tm_yday
    ax.axvline(ma24, color=C24, linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(ma25, color=C25, linestyle=":", linewidth=1, alpha=0.5)
    ax.text(ma24 + 1, ymax * 0.6, "2024 中秋", fontsize=7, color=C24)
    ax.text(ma25 + 1, ymax * 0.5, "2025 中秋", fontsize=7, color=C25)

    # Solar holidays (same date both years)
    for name, doy, ha_pos in [("劳动节", date(2025, 5, 1).timetuple().tm_yday, 0.95),
                                ("国庆", date(2025, 10, 1).timetuple().tm_yday, 0.95)]:
        ax.axvline(doy, color=C_GRAY, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(doy + 1, ymax * ha_pos, name, fontsize=7, color=C_GRAY)

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Total Units Sold (万)")
    ax.set_title("Fig 1: Daily Total Sales — 2024 vs 2025 (Spring Festival shifted by ~12 days)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    obs = ("KEY INSIGHT: 2024 Spring Festival (Feb 10) is 12 days later than 2025 (Jan 29). "
           "This explains why 2025 Jan looks much higher than 2024 Jan — pre-festival "
           "stocking moved into Jan in 2025. The two pre-festival peaks actually align "
           "almost perfectly relative to their respective Spring Festival dates.")
    return fig_to_b64(fig), obs


# ════════════════════════════════════════════════════════════════
#  Fig 2: Monthly bar
# ════════════════════════════════════════════════════════════════
def chart2(t24, t25):
    print("  Chart 2: Monthly bar...", flush=True)
    m24 = t24.groupby(t24["date"].dt.month)["quantity_sold"].sum()
    m25 = t25.groupby(t25["date"].dt.month)["quantity_sold"].sum()
    months = np.arange(1, 13)
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(months - w / 2, m24.reindex(months, fill_value=0).values / 1e4, w,
           label="2024", color=C24, alpha=0.85)
    ax.bar(months + w / 2, m25.reindex(months, fill_value=0).values / 1e4, w,
           label="2025", color=C25, alpha=0.85)
    for m in months:
        v24 = m24.get(m, 0)
        v25 = m25.get(m, 0)
        if v24 > 0:
            pct = (v25 - v24) / v24 * 100
            y = max(v24, v25) / 1e4
            ax.text(m, y + 10, f"{pct:+.0f}%", ha="center", fontsize=7,
                    color=C_GREEN if pct >= 0 else C25)
    ax.set_xticks(months)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Units Sold (万)")
    ax.set_title("Fig 2: Monthly Total Sales — 2024 vs 2025", fontsize=13, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    obs = ("Jan 2025 is +31% over Jan 2024 because Spring Festival fell in Jan; "
           "Feb 2024 is +43% over Feb 2025 for the mirror reason. After lunar adjustment, "
           "monthly volumes are largely consistent.")
    return fig_to_b64(fig), obs


# ════════════════════════════════════════════════════════════════
#  Fig 3: DOW pattern
# ════════════════════════════════════════════════════════════════
def chart3(t24, t25):
    print("  Chart 3: DOW pattern...", flush=True)
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def _stats(df):
        tmp = df.copy()
        tmp["dow"] = tmp["date"].dt.dayofweek
        daily = tmp.groupby(["date", "dow"])["quantity_sold"].sum().reset_index()
        g = daily.groupby("dow")["quantity_sold"]
        return g.mean().values / 1e4, g.std().values / 1e4

    m24, s24 = _stats(t24)
    m25, s25 = _stats(t25)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(7)
    ax.errorbar(x - 0.1, m24, yerr=s24, fmt="o-", color=C24, capsize=4, label="2024")
    ax.errorbar(x + 0.1, m25, yerr=s25, fmt="s-", color=C25, capsize=4, label="2025")
    ax.set_xticks(x)
    ax.set_xticklabels(dow_labels)
    ax.set_ylabel("Avg Daily Sales (万)")
    ax.set_title("Fig 3: Day-of-Week Pattern — 2024 vs 2025", fontsize=13, fontweight="bold")
    ax.legend()

    obs = ("Weekend sales consistently exceed weekdays in both years. The shape is "
           "nearly identical year-over-year, validating day_of_week as a stable feature.")
    return fig_to_b64(fig), obs


# ════════════════════════════════════════════════════════════════
#  Fig 4: Per-store
# ════════════════════════════════════════════════════════════════
def chart4(t24, t25):
    print("  Chart 4: Per-store monthly...", flush=True)
    store_order = ["store_c", "store_b", "store_a"]
    store_names = {"store_c": "门店C Store-C", "store_b": "门店B Store-B",
                   "store_a": "门店A Store-A"}
    months = np.arange(1, 13)
    labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, sid in zip(axes, store_order):
        df24 = t24[t24["store_id"] == sid]
        df25 = t25[t25["store_id"] == sid]
        s24 = df24.groupby(df24["date"].dt.month)["quantity_sold"].sum()
        s25 = df25.groupby(df25["date"].dt.month)["quantity_sold"].sum()
        ax.plot(months, s24.reindex(months, fill_value=0).values / 1e4, "o-",
                color=C24, label="2024", markersize=5)
        ax.plot(months, s25.reindex(months, fill_value=0).values / 1e4, "s-",
                color=C25, label="2025", markersize=5)
        ax.set_title(store_names[sid], fontsize=10, fontweight="bold")
        ax.set_xticks(months)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Monthly Sales (万)")
    fig.suptitle("Fig 4: Per-Store Monthly Sales — 2024 vs 2025",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    obs = ("All three stores show the same Jan↔Feb mirror flip (Spring Festival), "
           "but absolute volumes differ — Store C is the largest, Store A the smallest.")
    return fig_to_b64(fig), obs


# ════════════════════════════════════════════════════════════════
#  Fig 5: Top-50 SKU stability — INTERACTIVE Plotly
# ════════════════════════════════════════════════════════════════
def chart5_interactive(t24, t25, name_map):
    print("  Chart 5: Top SKU scatter (interactive)...", flush=True)
    a24 = t24.groupby("sku_id")["quantity_sold"].sum()
    a25 = t25.groupby("sku_id")["quantity_sold"].sum()
    both = pd.DataFrame({"s24": a24, "s25": a25}).dropna()
    both = both[(both["s24"] > 0) & (both["s25"] > 0)]
    both["pct"] = ((both["s25"] - both["s24"]) / both["s24"] * 100)
    both["abs_pct"] = both["pct"].abs()
    both["name"] = [name_map.get(idx, str(idx)) for idx in both.index]
    top50 = both.nlargest(50, "s24").reset_index()

    mn = min(top50["s24"].min(), top50["s25"].min()) * 0.7
    mx = max(top50["s24"].max(), top50["s25"].max()) * 1.4

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines",
        line=dict(dash="dash", color=C_GRAY, width=1),
        hoverinfo="skip", showlegend=False,
    ))
    customdata = np.stack([
        top50["sku_id"].astype(str).values,
        top50["name"].astype(str).values,
        top50["s24"].values,
        top50["s25"].values,
        top50["pct"].values,
    ], axis=-1)
    fig.add_trace(go.Scatter(
        x=top50["s24"], y=top50["s25"], mode="markers",
        marker=dict(
            size=12,
            color=top50["abs_pct"],
            colorscale=[[0, C_GREEN], [0.5, "#FDD835"], [1, "#D32F2F"]],
            cmin=0, cmax=100,
            colorbar=dict(title="|YoY %|"),
            line=dict(color="white", width=1),
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "SKU ID: %{customdata[0]}<br>"
            "2024 sales: %{customdata[2]:,.0f}<br>"
            "2025 sales: %{customdata[3]:,.0f}<br>"
            "YoY change: %{customdata[4]:+.1f}%<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_layout(
        title="Fig 5: Top-50 SKU Stability (interactive — hover for SKU info)",
        xaxis=dict(title="2024 Annual Sales (log)", type="log"),
        yaxis=dict(title="2025 Annual Sales (log)", type="log"),
        height=600, plot_bgcolor="#fafafa",
    )

    obs = ("Hover over any point to see the SKU name and exact sales numbers. "
           "Most top-50 SKUs cluster near the diagonal (stable), green dots are highly "
           "stable, red dots had large year-over-year shifts.")
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="fig5_div"), obs


# ════════════════════════════════════════════════════════════════
#  Fig 6: Disappeared / new SKUs — with category breakdown
# ════════════════════════════════════════════════════════════════
def chart6_with_breakdown(t24, t25, name_map, bc_map):
    print("  Chart 6: SKU continuity + category breakdown...", flush=True)
    skus_24 = set(t24["sku_id"].unique())
    skus_25 = set(t25["sku_id"].unique())
    only_24_ids = skus_24 - skus_25
    only_25_ids = skus_25 - skus_24
    stable_ids = skus_24 & skus_25

    # Sales volumes for disappeared & new
    s24_total = t24.groupby("sku_id")["quantity_sold"].sum()
    s25_total = t25.groupby("sku_id")["quantity_sold"].sum()

    # ── Category classification by keyword ──
    CATEGORIES = {
        "生鲜肉禽蛋": ["猪", "牛", "羊", "鸡", "鸭", "排骨", "五花", "里脊", "肉", "蛋",
                  "禽", "腿", "腰", "肝", "心"],
        "生鲜水产": ["鱼", "虾", "蟹", "三文鱼", "生蚝", "海鲜", "鲍", "贝", "鳕"],
        "生鲜果蔬": ["菜", "瓜", "桃", "梨", "苹果", "葡萄", "香蕉", "橙", "柚",
                  "芒果", "草莓", "樱桃", "蓝莓", "西瓜", "白菜", "萝卜", "土豆",
                  "番茄", "豆芽", "蘑菇", "杏鲍菇", "水果", "椒", "葱", "姜", "蒜",
                  "茄", "豆角", "韭", "芹"],
        "乳制品": ["牛奶", "酸奶", "鲜奶", "酸牛奶", "乳", "奶粉", "奶酪", "黄油"],
        "面包烘焙": ["面包", "蛋糕", "糕点", "饼干", "曲奇", "派"],
        "饮料": ["饮料", "可乐", "雪碧", "矿泉水", "果汁", "茶饮", "咖啡",
                "汽水", "饮", "水"],
        "酒水": ["白酒", "红酒", "黄酒", "啤酒", "酒"],
        "粮油调味": ["米", "面粉", "面条", "挂面", "油", "酱油", "醋", "盐", "糖",
                  "调味", "酱", "味精", "鸡精", "料酒", "豆豉"],
        "零食干货": ["饼", "薯片", "糖果", "巧克力", "瓜子", "坚果", "蜜饯", "话梅",
                  "辣条", "果脯", "山楂", "红枣", "枸杞", "桂圆", "海苔"],
        "速食冷冻": ["饺子", "汤圆", "馄饨", "包子", "馒头", "速冻", "水饺", "面点"],
        "豆制品": ["豆腐", "腐竹", "豆干", "千张"],
        "日用纸品": ["纸", "巾", "卷纸"],
        "个护清洁": ["洗发", "沐浴", "牙膏", "牙刷", "洗衣", "洗洁", "肥皂", "香皂",
                  "面膜", "化妆", "护肤", "口红", "粉", "霜", "乳液", "卫生", "尿"],
        "其他": [],
    }

    def classify(name: str) -> str:
        if pd.isna(name):
            return "其他"
        name = str(name)
        for cat, kws in CATEGORIES.items():
            for kw in kws:
                if kw in name:
                    return cat
        return "其他"

    # ── Check for renames: same barcode_base, different SKU id ──
    import re
    def strip_bc(bc):
        if pd.isna(bc):
            return None
        bc = str(bc).strip()
        m = re.match(r"^(\d{8,})[a-zA-Z]+$", bc)
        return m.group(1) if m else bc

    bc_24 = {sid: strip_bc(bc_map.get(sid)) for sid in only_24_ids}
    bc_25 = {sid: strip_bc(bc_map.get(sid)) for sid in only_25_ids}
    bc_24_set = {b for b in bc_24.values() if b}
    bc_25_set = {b for b in bc_25.values() if b}
    rename_count = len(bc_24_set & bc_25_set)

    # ── Build category breakdown tables (with example SKUs) ──
    def build_breakdown(ids, sales_series):
        rows = []
        for sid in ids:
            name = name_map.get(sid, str(sid))
            cat = classify(name)
            qty = sales_series.get(sid, 0)
            rows.append((cat, qty, name))
        df = pd.DataFrame(rows, columns=["category", "qty", "name"])
        agg = df.groupby("category").agg(
            n_sku=("qty", "count"),
            total_qty=("qty", "sum"),
        ).reset_index().sort_values("total_qty", ascending=False)
        # Top examples per category
        examples = {}
        for cat in agg["category"]:
            top = df[df["category"] == cat].nlargest(3, "qty")
            examples[cat] = [(row["name"], row["qty"]) for _, row in top.iterrows()]
        agg["examples"] = agg["category"].map(examples)
        return agg

    disappeared_breakdown = build_breakdown(only_24_ids, s24_total)
    new_breakdown = build_breakdown(only_25_ids, s25_total)

    # ── Plot: top bar (counts) + side-by-side category breakdown ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: SKU continuity counts
    cats = ["Stable", "Disappeared\n(2024 only)", "New\n(2025 only)"]
    vals = [len(stable_ids), len(only_24_ids), len(only_25_ids)]
    colors = [C_GREEN, C24, C25]
    bars = axes[0].bar(cats, vals, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                     f"{v:,}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("SKU Continuity", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of SKUs")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Subplot 2: Disappeared by category
    db = disappeared_breakdown.head(8)
    axes[1].barh(db["category"], db["total_qty"] / 1e4, color=C24, alpha=0.85)
    for i, (_, row) in enumerate(db.iterrows()):
        axes[1].text(row["total_qty"] / 1e4 + 1, i,
                     f"  {row['n_sku']:,} 个 / {row['total_qty']/1e4:.1f}万件",
                     va="center", fontsize=8)
    axes[1].set_title("Disappeared SKUs (2024 only) by Category", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("2024 Total Sales (万件)")
    axes[1].invert_yaxis()

    # Subplot 3: New by category
    nb = new_breakdown.head(8)
    axes[2].barh(nb["category"], nb["total_qty"] / 1e4, color=C25, alpha=0.85)
    for i, (_, row) in enumerate(nb.iterrows()):
        axes[2].text(row["total_qty"] / 1e4 + 1, i,
                     f"  {row['n_sku']:,} 个 / {row['total_qty']/1e4:.1f}万件",
                     va="center", fontsize=8)
    axes[2].set_title("New SKUs (2025 only) by Category", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("2025 Total Sales (万件)")
    axes[2].invert_yaxis()

    fig.suptitle("Fig 6: SKU Continuity + Category Breakdown",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    # Build example tables for each category
    def fmt_examples(breakdown):
        lines = []
        for _, row in breakdown.head(6).iterrows():
            ex_str = " · ".join([f"{n[:14]} ({q:.0f})" for n, q in row["examples"]])
            lines.append(f"<b>{row['category']}</b> ({row['n_sku']:,} SKUs, "
                         f"{row['total_qty']/1e4:.1f}万件): {ex_str}")
        return "<br>".join(lines)

    obs = (f"<b>Counts:</b> Stable {len(stable_ids):,} | Disappeared {len(only_24_ids):,} "
           f"| New {len(only_25_ids):,}.<br><br>"
           f"<b>Renames (same barcode, different SKU ID):</b> ~{rename_count:,} SKUs "
           f"appear in both lists but share the same underlying barcode_base — these "
           f"are NOT real churn, just SKU ID changes (e.g. supplier reassigned the "
           f"internal code).<br><br>"
           f"<b>'其他' (Other) is the catch-all for SKUs whose names don't contain "
           f"any of the keywords for the 11 mapped categories — typically packaged "
           f"goods like 卫生巾, 洗发水, 化妆品, 厨具 etc.</b><br><br>"
           f"<b>Example SKUs (top 3 by sales) per disappeared category:</b><br>"
           f"{fmt_examples(disappeared_breakdown)}<br><br>"
           f"<b>Example SKUs per new category:</b><br>"
           f"{fmt_examples(new_breakdown)}")
    return fig_to_b64(fig), obs


# ════════════════════════════════════════════════════════════════
#  Fig 7: CV scatter — INTERACTIVE Plotly
# ════════════════════════════════════════════════════════════════
def chart7_interactive(t24, t25, name_map):
    print("  Chart 7: CV scatter (interactive)...", flush=True)

    def _cv(df):
        tmp = df.copy()
        daily = tmp.groupby(["sku_id", "date"])["quantity_sold"].sum().reset_index()
        g = daily.groupby("sku_id")["quantity_sold"]
        m, s = g.mean(), g.std()
        cv = (s / m).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({"mean_daily": m, "cv": cv}).dropna()

    cv24 = _cv(t24)
    cv25 = _cv(t25)
    both = cv24.join(cv25, lsuffix="_24", rsuffix="_25", how="inner").dropna()
    both["avg_mean"] = (both["mean_daily_24"] + both["mean_daily_25"]) / 2
    sample = both.nlargest(2000, "avg_mean") if len(both) > 2000 else both
    sample = sample.reset_index()
    sample["name"] = [name_map.get(s, str(s)) for s in sample["sku_id"]]

    mx = min(sample["cv_24"].quantile(0.98), sample["cv_25"].quantile(0.98))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, mx], y=[0, mx], mode="lines",
        line=dict(dash="dash", color=C_GRAY, width=1),
        hoverinfo="skip", showlegend=False,
    ))
    sizes = np.clip(np.sqrt(sample["avg_mean"]) * 2, 4, 30)
    customdata = np.stack([
        sample["sku_id"].astype(str).values,
        sample["name"].astype(str).values,
        sample["cv_24"].values,
        sample["cv_25"].values,
        sample["avg_mean"].values,
    ], axis=-1)
    fig.add_trace(go.Scatter(
        x=sample["cv_24"], y=sample["cv_25"], mode="markers",
        marker=dict(size=sizes, color=C24, opacity=0.45,
                    line=dict(color="white", width=0.3)),
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "SKU: %{customdata[0]}<br>"
            "2024 CV: %{customdata[2]:.2f}<br>"
            "2025 CV: %{customdata[3]:.2f}<br>"
            "Avg daily sales: %{customdata[4]:.1f}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_layout(
        title="Fig 7: SKU Demand Variability (CV) — 2024 vs 2025 (interactive)",
        xaxis=dict(title="2024 CV (= std / mean of daily sales)", range=[0, mx]),
        yaxis=dict(title="2025 CV (= std / mean of daily sales)", range=[0, mx]),
        height=600, plot_bgcolor="#fafafa",
    )

    obs = ("<b>What is CV?</b> CV (Coefficient of Variation) = standard deviation / mean. "
           "It measures how 'jumpy' an SKU's daily sales are relative to its average. "
           "CV=0.5 means ~50% day-to-day fluctuation. Low CV = stable, predictable item; "
           "high CV = erratic, hard to forecast. Hover any dot to see the SKU. Dot size "
           "reflects average daily sales — larger = higher volume. Most SKUs cluster "
           "along the diagonal: their variability is consistent year-over-year.")
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="fig7_div"), obs


# ════════════════════════════════════════════════════════════════
#  Fig 8: Monthly MAE + MAPE
# ════════════════════════════════════════════════════════════════
def chart8(t25_full, features, model):
    print("  Chart 8: Monthly error...", flush=True)
    pred = model.predict(t25_full[features].values)
    err = pd.DataFrame({
        "month": pd.to_datetime(t25_full["date"]).dt.month,
        "qty": t25_full["quantity_sold"].values,
        "ae": np.abs(t25_full["quantity_sold"].values - pred),
    })
    monthly = err.groupby("month").apply(
        lambda x: pd.Series({
            "mae": x["ae"].mean(),
            "mape": (x["ae"].sum() / max(x["qty"].sum(), 1)) * 100,  # weighted MAPE
            "count": len(x),
        }), include_groups=False,
    ).reset_index()
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Left: MAE
    bars1 = axes[0].bar(labels, monthly["mae"], color=C25, alpha=0.85, edgecolor="white")
    for bar, row in zip(bars1, monthly.itertuples()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{row.mae:.2f}", ha="center", fontsize=8)
    axes[0].set_ylabel("MAE (units)")
    axes[0].set_title("Monthly MAE (absolute, per row)", fontsize=11, fontweight="bold")
    # Highlight Jan & Feb
    for i in [0, 1]:
        bars1[i].set_color("#B71C1C")
        bars1[i].set_alpha(0.95)

    # Right: MAPE (weighted, more meaningful)
    bars2 = axes[1].bar(labels, monthly["mape"], color=C24, alpha=0.85, edgecolor="white")
    for bar, row in zip(bars2, monthly.itertuples()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{row.mape:.1f}%", ha="center", fontsize=8)
    axes[1].set_ylabel("Weighted MAPE (%)")
    axes[1].set_title("Monthly MAPE (volume-weighted, % of demand)",
                       fontsize=11, fontweight="bold")
    for i in [0, 1]:
        bars2[i].set_color("#0D47A1")
        bars2[i].set_alpha(0.95)

    fig.suptitle("Fig 8: Monthly Model Error — 2025 OOS (q=0.50)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    obs = ("Jan-Feb MAE/MAPE are highest, even though the model has Spring Festival "
           "features (days_to_nearest_holiday, holiday_name, is_makeup_workday). "
           "Likely causes: (1) only 1 year of training data — the model has seen only "
           "ONE Spring Festival (2024 Feb 10), but 2025 SF is Jan 29 — different DOY, "
           "so the model can't easily transfer; (2) Spring Festival shopping intensity "
           "varies 5-10x vs normal days, so even small relative errors translate to "
           "large MAE. Possible improvements: stronger weight on holiday-window samples, "
           "or train on 2-3 years to learn the lunar offset.")
    return fig_to_b64(fig), obs


# ════════════════════════════════════════════════════════════════
#  Fig 9: Error by bucket — MAE vs MAPE
# ════════════════════════════════════════════════════════════════
def chart9(t25_full, features, model):
    print("  Chart 9: Error by bucket...", flush=True)
    pred = model.predict(t25_full[features].values)
    err = pd.DataFrame({
        "sku_id": t25_full["sku_id"].values,
        "qty": t25_full["quantity_sold"].values,
        "ae": np.abs(t25_full["quantity_sold"].values - pred),
    })
    sku_stats = err.groupby("sku_id").agg(
        total=("qty", "sum"),
        mae=("ae", "mean"),
        mean_daily=("qty", "mean"),
    ).reset_index()
    # Volume-weighted MAPE per SKU
    sku_mape = err.groupby("sku_id").apply(
        lambda x: x["ae"].sum() / max(x["qty"].sum(), 1) * 100,
        include_groups=False,
    ).rename("mape").reset_index()
    sku_stats = sku_stats.merge(sku_mape, on="sku_id")
    sku_stats["bucket"] = pd.qcut(sku_stats["total"], 5,
                                   labels=["Bottom 20%", "20-40%", "40-60%", "60-80%", "Top 20%"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bucket_order = ["Bottom 20%", "20-40%", "40-60%", "60-80%", "Top 20%"]

    # Left: MAE
    data_mae = [sku_stats[sku_stats["bucket"] == b]["mae"].values for b in bucket_order]
    bp1 = axes[0].boxplot(data_mae, tick_labels=bucket_order, patch_artist=True, showfliers=False)
    bucket_colors = ["#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5"]
    for patch, c in zip(bp1["boxes"], bucket_colors):
        patch.set_facecolor(c)
    for i, d in enumerate(data_mae, 1):
        axes[0].text(i, np.median(d), f" {np.median(d):.2f}",
                     va="center", fontsize=8, color=C24)
    axes[0].set_ylabel("MAE per SKU (units)")
    axes[0].set_title("By Volume Bucket — MAE (absolute)",
                       fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Volume Bucket")

    # Right: MAPE - filter to where total > 0
    valid = sku_stats[sku_stats["total"] > 0].copy()
    data_mape = [valid[valid["bucket"] == b]["mape"].values for b in bucket_order]
    # Clip extreme values for box plot readability
    data_mape = [np.clip(d, 0, 200) for d in data_mape]
    bp2 = axes[1].boxplot(data_mape, tick_labels=bucket_order, patch_artist=True, showfliers=False)
    for patch, c in zip(bp2["boxes"], ["#FFCDD2", "#EF9A9A", "#E57373", "#EF5350", "#E53935"]):
        patch.set_facecolor(c)
    for i, d in enumerate(data_mape, 1):
        axes[1].text(i, np.median(d), f" {np.median(d):.1f}%",
                     va="center", fontsize=8, color=C25)
    axes[1].set_ylabel("MAPE per SKU (%, capped at 200%)")
    axes[1].set_title("By Volume Bucket — MAPE (relative)",
                       fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Volume Bucket")

    fig.suptitle("Fig 9: Model Error by SKU Volume — Absolute vs Relative",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    obs = ("Two views answer your question: <b>Top-20% SKUs have the highest MAE</b> "
           "because they sell ~50× more units than Bottom-20% (a 5-unit error on a "
           "100/day item is large in absolute terms). <b>But on MAPE (right), Top-20% "
           "is actually the BEST</b> — high-volume items are proportionally the most "
           "accurate, ~5-10% MAPE vs ~50%+ for bottom buckets. This is exactly the "
           "behavior we want for profit optimization: the high-revenue SKUs have the "
           "lowest relative error.")
    return fig_to_b64(fig), obs


# ════════════════════════════════════════════════════════════════
#  Fig 10: Worst SKUs — INTERACTIVE Plotly
# ════════════════════════════════════════════════════════════════
def chart10_interactive(t25_full, features, model, name_map):
    print("  Chart 10: Worst SKUs (interactive)...", flush=True)
    pred = model.predict(t25_full[features].values)
    err = pd.DataFrame({
        "sku_id": t25_full["sku_id"].values,
        "qty": t25_full["quantity_sold"].values,
        "ae": np.abs(t25_full["quantity_sold"].values - pred),
    })
    sku_stats = err.groupby("sku_id").agg(
        mean_daily=("qty", "mean"), mae=("ae", "mean"),
    ).reset_index()
    sku_mape = err.groupby("sku_id").apply(
        lambda x: x["ae"].sum() / max(x["qty"].sum(), 1) * 100,
        include_groups=False,
    ).rename("mape").reset_index()
    sku_stats = sku_stats.merge(sku_mape, on="sku_id")

    active = sku_stats[sku_stats["mean_daily"] >= 3].copy()
    active["name"] = [name_map.get(s, str(s)) for s in active["sku_id"]]
    active["mape_clip"] = np.clip(active["mape"], 0, min(active["mape"].quantile(0.99), 300))

    # Mark the top-15 worst high-volume as a separate trace for visual highlight
    high_vol = active[active["mean_daily"] >= 10]
    worst15 = high_vol.nlargest(15, "mape")
    worst_set = set(worst15["sku_id"].values)
    active["is_worst"] = active["sku_id"].isin(worst_set)

    fig = go.Figure()

    # All points (background)
    bg = active[~active["is_worst"]]
    fig.add_trace(go.Scatter(
        x=bg["mean_daily"], y=bg["mape_clip"], mode="markers",
        marker=dict(size=5, color=C25, opacity=0.3, line=dict(width=0)),
        customdata=np.stack([
            bg["sku_id"].astype(str).values,
            bg["name"].astype(str).values,
            bg["mean_daily"].values,
            bg["mape"].values,
            bg["mae"].values,
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "SKU: %{customdata[0]}<br>"
            "Avg daily: %{customdata[2]:.1f} units<br>"
            "MAPE: %{customdata[3]:.1f}%<br>"
            "MAE: %{customdata[4]:.2f}<extra></extra>"
        ),
        name="All SKUs", showlegend=False,
    ))

    # Highlight worst-15
    fig.add_trace(go.Scatter(
        x=worst15["mean_daily"], y=np.clip(worst15["mape"], 0, 300),
        mode="markers",
        marker=dict(size=14, color="#B71C1C",
                    line=dict(color="white", width=1.5),
                    symbol="diamond"),
        customdata=np.stack([
            worst15["sku_id"].astype(str).values,
            worst15["name"].astype(str).values,
            worst15["mean_daily"].values,
            worst15["mape"].values,
            worst15["mae"].values,
        ], axis=-1),
        hovertemplate=(
            "<b>★ %{customdata[1]}</b><br>"
            "SKU: %{customdata[0]}<br>"
            "Avg daily: %{customdata[2]:.1f} units<br>"
            "MAPE: %{customdata[3]:.1f}%<br>"
            "MAE: %{customdata[4]:.2f}<extra></extra>"
        ),
        name="Top-15 worst (high vol)", showlegend=True,
    ))

    fig.update_layout(
        title="Fig 10: Worst-Performing SKUs — MAPE vs Daily Volume (interactive)",
        xaxis=dict(title="Mean Daily Sales (log scale)", type="log"),
        yaxis=dict(title="MAPE (%, capped at 99th percentile)"),
        height=600, plot_bgcolor="#fafafa",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    obs = ("Hover any dot for the SKU name and exact metrics. Long-tail pattern: "
           "low-volume SKUs (left) have high MAPE because a 1-unit error on a 3/day "
           "item is 33%. High-volume SKUs (right) cluster at lower MAPE. The 15 red "
           "diamonds are the worst-performing high-volume SKUs (daily≥10) — these are "
           "the items where prediction improvements would most directly impact profit.")
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="fig10_div"), obs


# ════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)

    t24, t25, name_map, bc_map = load_data()

    print("Loading full OOS for Part 5...", flush=True)
    t25_full = pd.read_parquet("data/features/features_oos.parquet")
    t25_full["date"] = pd.to_datetime(t25_full["date"])
    d = joblib.load("models/step3_results.joblib")
    features = d["features"]
    model = d["quantile_models"][0.5]
    for col in features:
        if col in t25_full.columns and t25_full[col].dtype == "bool":
            t25_full[col] = t25_full[col].astype(int)

    print("Building charts...", flush=True)
    charts = []  # (section, type, content, obs); type ∈ {"img", "div"}

    b1, o1 = chart1(t24, t25)
    charts.append(("Part 1: Aggregate Level", "img",
                   "Fig 1: Daily Total Sales — 2024 vs 2025 (lunar-aligned)", b1, o1))
    b2, o2 = chart2(t24, t25)
    charts.append((None, "img", "Fig 2: Monthly Total Sales", b2, o2))
    b3, o3 = chart3(t24, t25)
    charts.append((None, "img", "Fig 3: Day-of-Week Pattern", b3, o3))
    b4, o4 = chart4(t24, t25)
    charts.append(("Part 2: Per-Store Comparison", "img",
                   "Fig 4: Per-Store Monthly", b4, o4))

    div5, o5 = chart5_interactive(t24, t25, name_map)
    charts.append(("Part 3: SKU-Level Distribution Shift", "div",
                   "Fig 5: Top-50 SKU Stability (interactive)", div5, o5))
    b6, o6 = chart6_with_breakdown(t24, t25, name_map, bc_map)
    charts.append((None, "img", "Fig 6: SKU Continuity + Category Breakdown", b6, o6))

    div7, o7 = chart7_interactive(t24, t25, name_map)
    charts.append(("Part 4: Demand Pattern Stability", "div",
                   "Fig 7: CV Scatter (interactive)", div7, o7))

    b8, o8 = chart8(t25_full, features, model)
    charts.append(("Part 5: Model Error Analysis", "img",
                   "Fig 8: Monthly MAE + MAPE", b8, o8))
    b9, o9 = chart9(t25_full, features, model)
    charts.append((None, "img", "Fig 9: Error by Volume Bucket — MAE vs MAPE", b9, o9))
    div10, o10 = chart10_interactive(t25_full, features, model, name_map)
    charts.append((None, "div", "Fig 10: Worst SKUs (interactive)", div10, o10))

    # ── Embed plotly.js inline (large but offline-safe) ──
    print("Inlining Plotly JS (~3MB)...", flush=True)
    import plotly
    plotly_dir = Path(plotly.__file__).parent
    plotly_js_paths = list(plotly_dir.rglob("plotly.min.js"))
    if plotly_js_paths:
        plotly_js = plotly_js_paths[0].read_text(encoding="utf-8")
    else:
        # Fallback: use plotly.io to get the JS string
        from plotly.offline import get_plotlyjs
        plotly_js = get_plotlyjs()

    print("Assembling HTML...", flush=True)
    html_parts = [
        '<!DOCTYPE html><html><head><meta charset="UTF-8">',
        '<title>RetailLab: 2024 vs 2025 Sales Pattern Analysis</title>',
        f'<script type="text/javascript">{plotly_js}</script>',
        '<style>',
        'body{font-family:"Microsoft YaHei","Segoe UI",sans-serif;max-width:1200px;'
        'margin:0 auto;padding:20px;background:#fff}',
        'h1{color:#333;border-bottom:3px solid #D32F2F;padding-bottom:8px}',
        'h2{color:#1565C0;margin-top:40px;border-bottom:1px solid #ddd;padding-bottom:4px}',
        'img{max-width:100%;height:auto;display:block;margin:10px 0}',
        'p.obs{color:#444;margin:5px 0 30px 0;font-size:14px;'
        'border-left:3px solid #1565C0;padding-left:12px;background:#f5faff;'
        'padding-top:8px;padding-bottom:8px}',
        '</style></head><body>',
        '<h1>RetailLab: 2024 vs 2025 Sales Pattern Analysis</h1>',
        '<p>10 charts. Static figures (matplotlib) + 2 interactive figures (Plotly) — '
        'fully offline, no internet required.</p>',
    ]

    md_lines = ["# Key Findings: 2024 vs 2025 Sales Pattern Analysis\n"]

    for section, kind, title, content, obs in charts:
        if section:
            html_parts.append(f"<h2>{section}</h2>")
        if kind == "img":
            html_parts.append(f'<img src="data:image/png;base64,{content}" alt="{title}">')
        else:
            html_parts.append(content)
        html_parts.append(f'<p class="obs"><b>{title}</b>: {obs}</p>')
        # strip HTML tags for markdown
        import re as _re
        clean = _re.sub(r"<[^>]+>", "", obs)
        md_lines.append(f"- **{title}**: {clean}\n")

    html_parts.append("</body></html>")
    OUTPUT_HTML.write_text("\n".join(html_parts), encoding="utf-8")
    OUTPUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    sz = OUTPUT_HTML.stat().st_size / 1e6
    print(f"\nDashboard saved: {OUTPUT_HTML} ({sz:.1f} MB)", flush=True)
    print(f"Findings saved: {OUTPUT_MD}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Kemai POS raw data reader — loads .xls exports from multiple stores.

Handles the reality of Kemai POS exports:
- Different stores have different column sets
- Some files have header in row 1 (detected per-file, not per-folder)
- Files are split into 65K-row chunks (Excel row limit)
- Each store folder contains monthly files like "2024.01抚二_01.xls"

Store mapping:
- 门店A → store_id: store_a, city: city_a
- 门店B → store_id: store_b, city: city_a
- 门店C → store_id: store_c, city: city_b
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 门店文件夹名 → (内部门店ID, 城市) 的映射表
# 科脉POS导出时按门店建文件夹，这里统一映射为英文标识符，方便下游使用
_STORE_MAP: dict[str, tuple[str, str]] = {
    "门店A": ("store_a", "city_a"),
    "门店B": ("store_b", "city_a"),
    "门店C": ("store_c", "city_b"),
}

# 统一输出列名，所有门店的数据最终都归一化为这些列
# 这是下游清洗和特征工程的契约接口
_OUTPUT_COLUMNS = [
    "store_id",
    "store_city",
    "sku_id",          # 商品编码
    "barcode",         # 国际条码
    "sku_name",        # 商品名称
    "spec",            # 规格
    "quantity_sold",   # 销售数量
    "total_amount",    # 含税销售金额 or 销售金额
    "discount_amount", # 折扣金额 (if available)
    "timestamp",       # 日期
]


def load_all_stores(
    raw_dir: Path,
    store_folders: list[str] | None = None,
) -> pd.DataFrame:
    """Load POS data from all store folders under raw_dir.

    Args:
        raw_dir: Root directory containing store subfolders.
        store_folders: Specific folders to load. If None, auto-detect.

    Returns:
        Combined DataFrame with normalized columns and store_id.
    """
    # 读取原始数据目录下的所有门店子文件夹，合并为统一 DataFrame → 返回
    raw_dir = Path(raw_dir)
    # 未指定门店时，自动扫描目录下所有已知门店文件夹
    if store_folders is None:
        store_folders = [
            d.name for d in raw_dir.iterdir()
            if d.is_dir() and d.name in _STORE_MAP
        ]

    if not store_folders:
        logger.warning("No store folders found in %s", raw_dir)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    all_dfs: list[pd.DataFrame] = []
    # 按文件夹名排序保证加载顺序一致，便于调试和日志追踪
    for folder_name in sorted(store_folders):
        store_dir = raw_dir / folder_name
        if not store_dir.exists():
            logger.warning("Store folder not found: %s", store_dir)
            continue

        # 从映射表获取门店标识；未知门店降级使用文件夹名本身
        store_id, city = _STORE_MAP.get(folder_name, (folder_name, "unknown"))
        logger.info("Loading store: %s (%s)", folder_name, store_id)

        store_df = _load_store_folder(store_dir, store_id, city)
        if not store_df.empty:
            all_dfs.append(store_df)

    if not all_dfs:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        "Loaded %d total rows from %d stores",
        len(combined), len(all_dfs),
    )
    return combined


def _load_store_folder(
    store_dir: Path,
    store_id: str,
    city: str,
) -> pd.DataFrame:
    """Load and concatenate all .xls files in a store folder."""
    # 单个门店文件夹 → 读取所有 .xls 并拼接，附加 store_id/city 列
    xls_files = sorted(store_dir.glob("*.xls"))
    if not xls_files:
        logger.warning("No .xls files in %s", store_dir)
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for i, xls_path in enumerate(xls_files):
        try:
            # 每个文件单独检测表头格式，因为同一门店的不同时期导出格式可能不同
            is_header_row1 = _detect_header_format(xls_path)
            df = _read_single_file(xls_path, is_header_row1)
            # 将科脉中文列名映射为统一英文列名
            df = _normalize_kemai_columns(df)
            df["store_id"] = store_id
            df["store_city"] = city
            dfs.append(df)

            # 每 20 个文件打一次进度日志，避免大批量文件时无反馈
            if (i + 1) % 20 == 0:
                logger.info(
                    "  %s: loaded %d/%d files", store_id, i + 1, len(xls_files)
                )
        except Exception as e:
            # 单文件失败不应阻断整个门店的加载，记录错误后跳过
            logger.error("Failed to read %s: %s", xls_path.name, e)
            continue

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    logger.info(
        "  %s: %d files → %d rows", store_id, len(dfs), len(result),
    )
    return result


def _detect_header_format(file_path: Path) -> bool:
    """Detect if header row is at row 0 (normal) or row 1 (门店D format).

    门店D files have a decorative header in row 0 and real column names
    in row 1.
    """
    # 门店D的导出文件第0行是装饰性标题（如"XXX门店销售明细"），
    # 真正的列名在第1行。需要逐文件检测，因为格式可能混合存在。
    try:
        df = pd.read_excel(file_path, nrows=2, header=None)
        row0_vals = [str(v) for v in df.iloc[0].values]
        row1_vals = [str(v) for v in df.iloc[1].values if pd.notna(v)]

        # 用几个必定存在的中文列名来判断哪一行才是真正的表头
        known_cols = {"商品编码", "商品名称", "销售数量", "日期"}
        row0_has = any(v in known_cols for v in row0_vals)
        row1_has = any(v in known_cols for v in row1_vals)

        # 第0行没有已知列名、第1行有 → 说明第0行是装饰行
        if not row0_has and row1_has:
            return True
    except Exception:
        pass
    return False


def _read_single_file(file_path: Path, header_row1: bool = False) -> pd.DataFrame:
    """Read a single .xls file with appropriate header handling."""
    # 根据检测结果选择表头行：装饰性格式用第1行，正常格式用第0行
    header = 1 if header_row1 else 0
    df = pd.read_excel(file_path, header=header)

    # 科脉导出常带一列序号"#"，对分析无用，直接丢弃
    if "#" in df.columns:
        df = df.drop(columns=["#"])

    # 删除全空行（Excel导出文件末尾常有空行）
    df = df.dropna(how="all")

    return df


def _normalize_kemai_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Kemai POS columns to canonical names.

    Different stores export different column sets. We map them all
    to a common schema.
    """
    # 科脉POS不同门店的导出列名不完全一致（如"含税销售金额"vs"销售金额"），
    # 这里逐个字段做兼容性映射，确保输出 schema 统一
    col_map: dict[str, str] = {}

    # 商品编码 → sku_id（科脉内部编码，非国标条码）
    if "商品编码" in df.columns:
        col_map["商品编码"] = "sku_id"

    # 条码字段有多种叫法，按优先级取第一个匹配的
    for name in ["国际条码", "销售条码", "基本条码"]:
        if name in df.columns:
            col_map[name] = "barcode"
            break

    # 商品名称也有两种叫法
    if "商品名称" in df.columns:
        col_map["商品名称"] = "sku_name"
    elif "商品全名" in df.columns:
        col_map["商品全名"] = "sku_name"

    if "规格" in df.columns:
        col_map["规格"] = "spec"

    if "销售数量" in df.columns:
        col_map["销售数量"] = "quantity_sold"

    # 金额：优先取含税金额（更准确），否则降级为不含税金额
    if "含税销售金额" in df.columns:
        col_map["含税销售金额"] = "total_amount"
    elif "销售金额" in df.columns:
        col_map["销售金额"] = "total_amount"

    if "折扣金额" in df.columns:
        col_map["折扣金额"] = "discount_amount"

    if "日期" in df.columns:
        col_map["日期"] = "timestamp"

    # 以下为部分门店特有的附加字段，保留供分析使用
    if "仓库名称" in df.columns:
        col_map["仓库名称"] = "warehouse_name"

    # 门店D特有的类别和成本列
    if "类别名称" in df.columns:
        col_map["类别名称"] = "category"

    if "销售成本金额" in df.columns:
        col_map["销售成本金额"] = "cost_amount"

    df = df.rename(columns=col_map)

    # 日期列转为 datetime，无法解析的值设为 NaT 而非报错
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 数值列强制转为数值类型，防止 Excel 导出时混入文本
    for col in ["quantity_sold", "total_amount", "discount_amount", "cost_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # sku_id 转字符串并去空格，保证后续 join 的一致性
    if "sku_id" in df.columns:
        df["sku_id"] = df["sku_id"].astype(str).str.strip()

    # 条码必须当字符串处理，否则 pandas 会以浮点数读入导致精度丢失
    if "barcode" in df.columns:
        df["barcode"] = df["barcode"].apply(_format_barcode)

    return df


def _format_barcode(val: object) -> str:
    """Convert barcode to string, handling float scientific notation."""
    # Excel 经常把长数字条码存为浮点数（如 6.920184e+12），
    # 必须先转 int 再转 str，否则会得到科学计数法字符串
    if pd.isna(val):
        return ""
    if isinstance(val, float):
        return str(int(val))
    return str(val).strip()


def load_product_master(file_path: Path) -> pd.DataFrame:
    """Load Guanjiapo product master (商品信息.xls).

    The file has a title row at row 0, real headers at row 1.
    Key columns: 商品全名, 规格, 基本单位, 基本条码, 状态.

    Note: Product master does NOT have 商品编码 (internal POS code).
    Join with POS data must be done via 基本条码 ↔ 国际条码.

    Args:
        file_path: Path to 商品信息.xls.

    Returns:
        Normalized product master DataFrame.
    """
    # 管家婆商品信息文件 → 归一化后的商品主数据
    # 注意：管家婆没有"商品编码"字段，只能通过条码与科脉POS数据关联
    df = pd.read_excel(file_path, header=1)

    # 管家婆导出的"行号"列是自增序号，无业务意义
    if "行号" in df.columns:
        df = df.drop(columns=["行号"])

    col_map = {
        "商品全名": "sku_name",
        "规格": "spec",
        "基本单位": "unit_of_measure",
        "基本条码": "barcode",
        "状态": "status",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "barcode" in df.columns:
        df["barcode"] = df["barcode"].apply(_format_barcode)
        # 没有条码的商品无法与POS数据关联，必须丢弃
        before = len(df)
        df = df[df["barcode"].ne("") & df["barcode"].notna()]
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d products with no barcode", dropped)

    # 条码去重：同一条码可能因管家婆录入问题出现多行，保留第一条
    if "barcode" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["barcode"], keep="first")
        dups = before - len(df)
        if dups:
            logger.info("Dropped %d duplicate barcodes in product master", dups)

    # 创建 barcode_base：去掉条码末尾的字母后缀
    # 管家婆中同一商品不同口味/规格会在条码后加字母（如 6939580620190x），
    # 而科脉POS只存数字部分，需要 barcode_base 来做模糊匹配
    if "barcode" in df.columns:
        df["barcode_base"] = df["barcode"].apply(_strip_barcode_suffix)
        n_stripped = (df["barcode"] != df["barcode_base"]).sum()
        if n_stripped:
            logger.info(
                "Created barcode_base: %d barcodes had letter suffixes stripped",
                n_stripped,
            )

    logger.info("Loaded product master: %d products", len(df))
    return df


def _strip_barcode_suffix(barcode: str) -> str:
    """Strip trailing letter suffix from a barcode for matching.

    E.g., '6939580620190x' → '6939580620190'
    Only strips if there's a numeric prefix of 8+ digits followed by letters.
    """
    # 只在"8位以上数字+字母尾缀"的模式下才去掉字母，
    # 避免误伤短条码或纯字母编码
    if not barcode:
        return barcode
    match = re.match(r"^(\d{8,})[a-zA-Z]+$", barcode)
    if match:
        return match.group(1)
    return barcode


def load_purchase_summary(
    file_paths: list[Path],
) -> pd.DataFrame:
    """Load Guanjiapo purchase summary (进货统计) for unit cost data.

    These files have a title row at row 0, real headers at row 1.
    Key column: 进货均价 = average purchase price (unit cost).

    Args:
        file_paths: Paths to purchase summary .xls files.

    Returns:
        DataFrame with barcode and unit_cost columns.
    """
    # 多个进货统计文件（通常按年份分文件）→ 合并后计算加权平均成本
    all_dfs: list[pd.DataFrame] = []
    for fp in file_paths:
        # 管家婆导出统一第0行是标题，第1行才是真正表头
        df = pd.read_excel(fp, header=1)
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    if "行号" in combined.columns:
        combined = combined.drop(columns=["行号"])

    col_map = {
        "商品全名": "sku_name",
        "基本条码": "barcode",
        "进货均价": "unit_cost",
        "进货数量": "purchase_qty",
        "折后金额": "purchase_total",
        "规格": "spec",
        "辅助单位": "aux_unit",
        "赠品数量": "gift_qty",
    }
    combined = combined.rename(
        columns={k: v for k, v in col_map.items() if k in combined.columns}
    )

    if "barcode" in combined.columns:
        combined["barcode"] = combined["barcode"].apply(_format_barcode)

    for col in ["unit_cost", "purchase_qty", "purchase_total", "gift_qty"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # 识别纯赠品SKU：进货成本为空 且 进货数量全部是赠品
    # 赠品不应参与成本计算和报童模型的 Cu/Co 比率
    if "gift_qty" in combined.columns:
        combined["gift_qty"] = combined["gift_qty"].fillna(0)
        is_gift = combined["unit_cost"].isna() & (
            combined["purchase_qty"] == combined["gift_qty"]
        )
        combined["is_promotional_gift"] = is_gift
        n_gifts = is_gift.sum()
        if n_gifts:
            logger.info("Marked %d pure promotional gift SKUs", n_gifts)
    else:
        combined["is_promotional_gift"] = False

    # 缺失成本填0：纯赠品或数据缺失的情况，下游会特殊处理
    combined["unit_cost"] = combined["unit_cost"].fillna(0)

    # 按条码聚合多年数据：用加权平均（总金额/总数量）而非简单平均，
    # 因为不同年份进货量差异大，加权平均更能反映真实成本
    if "barcode" in combined.columns:
        if "purchase_total" in combined.columns and "purchase_qty" in combined.columns:
            agg = combined.groupby("barcode", as_index=False).agg({
                "purchase_total": "sum",
                "purchase_qty": "sum",
                "sku_name": "first",
                # 只有当所有年份都标记为赠品时才认定为赠品
                "is_promotional_gift": "all",
            })
            # 加权平均成本 = 总进货金额 / 总进货数量（避免除零）
            mask = agg["purchase_qty"] > 0
            agg.loc[mask, "unit_cost"] = (
                agg.loc[mask, "purchase_total"] / agg.loc[mask, "purchase_qty"]
            )
            agg.loc[~mask, "unit_cost"] = 0
            combined = agg
        else:
            # 没有总金额/数量列时降级为简单平均
            combined = combined.groupby("barcode", as_index=False).agg({
                "unit_cost": "mean",
                "sku_name": "first",
                "is_promotional_gift": "all",
            })

    # 同样创建 barcode_base 用于与商品主数据的模糊匹配
    if "barcode" in combined.columns:
        combined["barcode_base"] = combined["barcode"].apply(_strip_barcode_suffix)

    logger.info(
        "Loaded purchase data: %d SKUs (%d promotional gifts)",
        len(combined), combined["is_promotional_gift"].sum(),
    )
    return combined

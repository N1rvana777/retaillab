"""SKU category mapping based on 商品编码 4-digit prefix.

Hand-classified by inspection of representative SKUs from the actual
purchase data (data/features/prefix_samples.json). Each prefix is
assigned a fine-grained category (cat_l2) and a coarse category (cat_l1).

Coverage: 72 prefixes covering all 77,198 自营 SKUs from purchase data.
SKUs without a 商品编码 (POS-only, mostly 联营/特价/购物袋) are handled
separately in classify_pos_only().
"""
from __future__ import annotations

import re

# ────────────────────────────────────────────────────────────────────
# 4-digit prefix → (coarse, fine) category labels
# ────────────────────────────────────────────────────────────────────

PREFIX_MAP: dict[str, tuple[str, str]] = {
    # ── 01XX 食品 / 生鲜 ──
    "0101": ("速冻冷藏", "速冻面食水饺"),       # 三全/安井/虾仁/虾滑/灌汤水饺
    "0104": ("生鲜水产", "鲜活水产"),           # 鲜黄花鱼/花蚬子/活虾/生蚝
    "0105": ("加工肉品", "火腿肠/汉堡"),        # 双汇/立杰汉堡/烤肠
    "0106": ("乳制品/烘焙", "鲜奶+包装面包"),    # 辉山鲜奶+桃李面包
    "0107": ("休闲零食", "散装糕点零食"),       # 散糕点/瓜子/蜜枣
    "0108": ("生鲜豆制品", "豆腐豆制品"),       # 干豆腐/大豆腐/素鸡
    "0109": ("生鲜蛋禽", "禽蛋/咸蛋/血制品"),    # 鸭腿/咸鸭蛋/血肠
    "0110": ("生鲜进口", "进口水果/月饼"),       # 榴莲/月饼礼盒
    "0112": ("调味干货", "火锅料/干货"),         # 海底捞蘸料/银耳/腐竹
    "0113": ("礼盒杂粮", "杂粮/礼盒"),          # 玉米杂粮/窝头/礼盒
    "0114": ("烘焙糕点", "甜品小点"),           # 雪媚娘/毛巾卷

    # ── 02XX 包装食品 / 酒饮 ──
    "0201": ("饮料酒水", "饮料/液态奶"),        # 娃哈哈水/农夫山泉/骑士酸奶
    "0202": ("休闲零食", "方便面/速食面"),      # 康师傅/统一汤达人
    "0203": ("米面粮油", "米面油杂粮"),         # 大米/面粉/豆油/红小豆
    "0204": ("乳制品/麦片", "燕麦/奶粉/麦片"),   # 西麦燕麦/飞鹤奶粉
    "0205": ("调味品", "酱油醋蚝油"),           # 海天/六月鲜/红梅味精
    "0206": ("茶叶礼盒", "茶叶"),               # 大红袍/铁观音/白茶
    # ── 03XX 个护 / 家清 / 百货 ──
    "0301": ("个护洗护", "洗发沐浴洗护"),       # 洗发露/沐浴/洗手
    "0302": ("家庭清洁", "洗衣液/洗洁精"),      # 立白/雕牌/威露士
    "0303": ("家庭清洁", "厕清/消毒液"),        # 蓝月亮/立白厕清
    "0304": ("家居电池", "电池/杀虫剂"),         # 南孚/雷达蚊香/超威
    "0305": ("家居用品", "保鲜袋膜/纸杯/购物袋/一次性手套"),
    "0306": ("家居用品", "保鲜袋膜/餐盒/收纳"),
    "0307": ("纸品卫生", "卫生纸/卫生巾/纸抽"),
    "0308": ("服饰袜帽", "袜子/家居服"),
    "0310": ("花卉鲜花", "鲜花/插花"),
    "0311": ("文具办公", "文具/红包"),
    "0312": ("数码电器", "插排/转换器/电器小件"),

    # ── 04XX 批发 / 积分 / 玩具 ──
    "0401": ("积分兑换", "积分换购"),
    "0402": ("玩具礼品", "毛绒玩具"),
    "0408": ("中转批发", "中转大宗(米油调味)"),
    "0411": ("大客户批发", "大客户/批发"),

    # ── 05XX 自营烘焙熟食 / 联营 ──
    "0501": ("自营烘焙", "主食面包/槽子糕"),
    "0522": ("乳制品", "羊奶/羊乳"),
    "0530": ("联营熟食", "小吃城烤鸡/烧饼"),
    "0543": ("联营杂项", "小吃城纸巾/果篮/电器"),
    "0544": ("品牌鲜蛋", "咯咯哒鸡蛋"),
    "0549": ("腊味", "腊肉"),
    "0568": ("腊味", "腊肠"),
    "0569": ("自营烘焙", "蛋糕/瑞士卷/盒子"),
    "0571": ("自营烘焙", "雪媚娘/泡芙/小蛋糕"),
    "0572": ("玩具礼品", "招财猫等杂项"),

    # ── 06XX 服饰 ──
    "0657": ("服饰内衣", "保暖内衣/驼老大"),

    # ── 09XX 包装 / 联营 ──
    "0901": ("包装容器", "塑料盒/果切杯/鸡蛋托"),
    "0902": ("包装容器", "购物袋/包装袋/餐盒/果蔬盒"),
    "0903": ("联营熟食", "小吃城/寿司熟食"),
    "0904": ("联营熟食", "小吃城面食"),

    # ── 11XX 自营生鲜 / 烘焙 ──
    "1101": ("自营熟食", "自营熟食/水饺"),
    "1104": ("生鲜禽肉", "鸡肉分割"),
    "1105": ("自营烘焙", "桃李面包/月饼"),
    "1106": ("乳制品", "酸奶/液态奶"),
    "1108": ("生鲜蔬菜", "精品蔬菜"),
    "1109": ("生鲜蛋", "鲜鸡蛋"),
    "1110": ("礼盒糕点", "月饼礼盒"),

    # ── 22XX 品牌零食 / 酒水 / 调味 ──
    "2201": ("饮料酒水", "啤酒/天湖"),
    "2202": ("休闲零食", "进口零食/薯片"),
    "2203": ("米面粮油", "大宗米面油"),
    "2204": ("乳制品/麦片", "智力麦片/芝麻糊"),
    "2205": ("调味品", "海天调味系列"),

    # ── 33XX 家清 / 纸品 / 家电 ──
    "3301": ("家庭清洁", "蓝月亮/家清品牌"),
    "3307": ("纸品卫生", "心相印/维达"),
    "3309": ("家居小家电", "电饭煲/破壁机"),
    "3010": ("个护洗护", "化妆/洁颜"),

    # ── 异常编码（少量数据，单独标注） ──
    "1-01": ("其他/异常码", "异常码"),
    "1-03": ("个护洗护", "洗发水(异常码)"),
    "1-11": ("生鲜进口", "进口水果(异常码)"),
    "1-33": ("个护洗护", "湿巾(异常码)"),
    "3-03": ("家居电池", "灭蚊灯(异常码)"),
    "4401": ("自营熟食", "联营品牌E冰淇淋"),
    "8888": ("VIP活动", "VIP拼团"),
    "QS01": ("其他/异常码", "QS异常码"),
    "寿司耗0": ("联营耗材", "寿司一次性耗材"),
}

# ────────────────────────────────────────────────────────────────────
# Coarse-category list (cat_l1)
# ────────────────────────────────────────────────────────────────────
ALL_COARSE_CATEGORIES = sorted(set(v[0] for v in PREFIX_MAP.values()))


# ────────────────────────────────────────────────────────────────────
# 联营 / 特殊 SKU detection (POS-only, no 商品编码)
# ────────────────────────────────────────────────────────────────────

# These keywords identify 联营 (consignment) SKUs that should be EXCLUDED
# from model training (we don't have purchase cost data for them).
LIANYING_KEYWORDS = (
    # 联营烘焙/熟食柜台
    "联营品牌A", "美食城", "小吃城", "寿司", "特价签", "特价品",
    "特价食品", "特价馒头", "特价鸡",
    # 联营品牌 (specific vendor names from observed data)
    "联营品牌B", "联营品牌C", "联营品牌D", "联营品牌E", "联营品牌F",
    "联营品牌G", "联营品牌H", "联营品牌I", "联营品牌J",
    "联营品牌K", "联营品牌L", "联营品牌M",
    # 联营整份计价 / 称重熟食
    "整份计价销售",
    # 服务类（非商品但应排除）
    "员工套餐", "门票", "停车费", "充值",
)

# Non-product items (packaging, fees) that should be excluded entirely.
NON_PRODUCT_KEYWORDS = (
    "购物袋", "包装袋", "果蔬盒", "鸡蛋托", "餐盒", "杯子",
    "扎口绳", "收银小票", "停车", "纸杯",
)


def classify_lianying(name: str) -> bool:
    """True if SKU name suggests 联营 (consignment)."""
    if not isinstance(name, str):
        return False
    return any(kw in name for kw in LIANYING_KEYWORDS)


def classify_non_product(name: str) -> bool:
    """True if SKU name suggests non-product (packaging, fees, etc.)."""
    if not isinstance(name, str):
        return False
    return any(kw in name for kw in NON_PRODUCT_KEYWORDS)


# ────────────────────────────────────────────────────────────────────
# Fallback keyword classifier for POS-only SKUs that have no 商品编码
# ────────────────────────────────────────────────────────────────────

POS_ONLY_KEYWORDS: list[tuple[tuple[str, str], tuple[str, ...]]] = [
    # ── 联营熟食/烘焙 (specific brands first - more specific) ──
    (("联营熟食", "联营品牌A系列"), ("联营品牌A",)),
    (("联营熟食", "美食城系列"), ("美食城",)),
    (("联营熟食", "小吃城系列"), ("小吃城",)),
    (("联营熟食", "寿司类"), ("寿司",)),
    (("联营熟食", "联营品牌B系列"), ("联营品牌B",)),
    (("联营烘焙", "联营品牌C系列"), ("联营品牌C",)),
    (("联营熟食", "联营品牌D系列"), ("联营品牌D",)),
    (("联营熟食", "联营品牌K系列"), ("联营品牌K",)),
    (("联营熟食", "联营品牌L系列"), ("联营品牌L",)),
    (("联营熟食", "联营品牌M系列"), ("联营品牌M",)),
    (("联营熟食", "其他联营品牌"),
        ("联营品牌E", "联营品牌F", "联营品牌G", "联营品牌H", "联营品牌I")),
    # 清真肉品（自营 但需独立分类）
    (("生鲜禽肉", "清真肉品"),
        ("清真",)),
    # 水吧饮品（联营）
    (("联营熟食", "水吧饮品"),
        ("水吧",)),
    # 大客户批发
    (("大客户批发", "大客户散装"),
        ("大客户",)),
    # 服务/票务（非商品）
    (("包装/非商品", "服务票务"),
        ("员工套餐", "门票", "套餐（后厨", "充值")),
    # 散装腊肠/肉枣
    (("加工肉品", "散装腊肠"),
        ("粉肠", "干肠", "枣肠", "肉皮", "炒台肠", "实蛋")),
    # 散装水产
    (("生鲜水产", "称重水产"),
        ("三文鱼", "海蜇", "鲜虾", "活虾")),

    # ── 称重生鲜 (PLU-coded, no barcode) ──
    # 猪肉分割（含称重 + 单品牌散装）
    (("生鲜禽肉", "称重猪肉"),
        ("前槽", "后丘", "下五花", "上五花", "通长五花", "大五花",
         "里脊条", "里脊", "外脊", "排骨", "净排", "猪头肉", "猪肉",
         "猪肝", "猪心", "猪耳", "猪手", "肘子", "夹心肉", "净肉",
         "杂骨", "大骨", "胸软骨", "腿肉", "肋排", "棒骨", "梅肉",
         "肉馅", "刚子", "脊骨", "颈骨", "猪蹄", "肥肉", "五花肉",
         "前肘", "后肘", "前腿", "后腿")),
    (("生鲜禽肉", "称重牛羊"),
        ("牛腱", "牛腩", "牛肋", "牛排", "牛肉", "鲜牛", "限量鲜牛",
         "羊腿", "羊肩", "羊排", "羊蝎子", "羊肉", "鲜羊")),
    (("生鲜禽肉", "称重鸡鸭"),
        ("鸡腿", "鸡翅", "鸡胸", "鸡爪", "鸡心", "鸭脖", "鸭翅",
         "琵琶腿", "鸡架", "鸡胗")),
    # 称重菜
    (("生鲜蔬菜", "称重蔬菜/水果"),
        ("特价菜", "菜（", "蔬菜", "果（")),
    # 加工肉
    (("加工肉品", "称重肉肠"),
        ("风味肠", "腊肠", "腊肉", "香肠", "肉枣", "肉脯", "肉松")),

    # ── 称重熟食 ──
    (("自营熟食", "卤味熟食"),
        ("卤", "酱", "凉拌", "凉菜", "刺身", "拼盘",
         "三明治", "奥尔良", "锅包肉", "溜肉段", "红烧狮子头",
         "现场制作", "肉段")),

    # ── 联营主食/快餐 ──
    (("联营熟食", "联营主食"),
        ("米饭", "卷饼", "面条", "盖饭", "盒饭", "炒饭")),

    # ── 包装/非商品 ──
    (("包装/非商品", "购物袋/包装"),
        ("购物袋", "包装袋", "塑料袋", "餐盒", "纸杯", "鸡蛋托", "果蔬盒")),

    # ── 自营烘焙类 (last - generic) ──
    (("自营烘焙", "面包蛋糕"),
        ("面包", "蛋糕", "馒头", "花卷", "面食", "烧饼", "饺子皮")),

    # ── 鸡蛋 ──
    (("生鲜蛋", "散装鸡蛋"),
        ("鸡蛋", "鸭蛋", "咸蛋")),

    # ── 豆制品 ──
    (("生鲜豆制品", "称重豆制品"),
        ("豆腐", "豆干", "腐竹", "千张", "豆皮")),
]


def classify_pos_only(name: str) -> tuple[str, str]:
    """Classify a POS-only SKU (no 商品编码) by name keywords.

    Returns (cat_l1, cat_l2). Defaults to ("其他", "未分类") if no match.
    """
    if not isinstance(name, str) or not name:
        return ("其他", "未分类")
    for (l1, l2), kws in POS_ONLY_KEYWORDS:
        if any(kw in name for kw in kws):
            return (l1, l2)
    return ("其他", "未分类")


# ────────────────────────────────────────────────────────────────────
# Main classifier
# ────────────────────────────────────────────────────────────────────

def classify_sku(commodity_code: str | None, sku_name: str | None) -> tuple[str, str, bool, bool]:
    """Classify one SKU.

    Returns:
        (cat_l1, cat_l2, is_lianying, is_pos_only_non_product)

    Priority (revised):
    1. If 商品编码 has known prefix → Tier 1 (precise) - use PREFIX_MAP.
       This wins even for items whose name contains "购物袋" etc, because
       the existence of a 商品编码 with valid prefix means the item has
       real purchase tracking.
    2. Else, mark is_lianying via name keywords.
    3. Else, classify by POS-only keywords.
    4. Only mark as non-product if it's POS-only AND name suggests it.
    """
    name_str = str(sku_name) if sku_name is not None else ""

    # Step 1: Prefix lookup wins (true Tier-1 items have purchase tracking)
    if commodity_code:
        code = str(commodity_code).replace(".0", "")
        # Try original code (preserves leading zero) and stripped version
        candidates = [code, code.lstrip("0") or "0"]
        # Build set of unique prefixes to try
        for orig in candidates:
            for plen in [6, 5, 4, 3, 2]:
                if len(orig) >= plen:
                    p = orig[:plen]
                    if p in PREFIX_MAP:
                        l1, l2 = PREFIX_MAP[p]
                        # 联营 flag still applies even for prefix-matched items
                        is_lianying = classify_lianying(name_str)
                        return (l1, l2, is_lianying, False)

    # Step 2-4: POS-only path
    is_lianying = classify_lianying(name_str)

    # Non-product filter only applies to POS-only items (no purchase tracking)
    if classify_non_product(name_str):
        return ("包装/非商品", "非商品(POS-only)", False, True)

    l1, l2 = classify_pos_only(name_str)
    return (l1, l2, is_lianying, False)


if __name__ == "__main__":
    # Quick test
    tests = [
        ("0101010037", "500g三全上海风味馄饨"),
        ("0201010001", "550ml农夫山泉"),
        (None, "联营品牌A）主食特价签"),
        (None, "大购物袋"),
        ("8888001", "VIP)维纳斯苹果"),
    ]
    for code, name in tests:
        r = classify_sku(code, name)
        print(f"{code} | {name}\n  -> {r}")

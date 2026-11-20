"""Holiday and special event calendar — Chinese holidays + lunar festivals.

Chinese holidays cause massive demand shifts:
- Spring Festival: 5-10x spike in snacks, gifts, meat
- National Day: foot traffic drops but travel food increases
- Lunar festivals (Lantern, Laba, etc.) affect specific categories

Uses chinese-calendar for official holidays (including makeup workdays)
and lunardate for lunar calendar festivals. Covers 2024-2050.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# ── 农历节日定义 ──
# 格式：(农历月, 农历日, 节日名称, 对零售的影响说明)
# 农历日期每年对应的公历不同，需要用 lunardate 库逐年转换
_LUNAR_FESTIVALS: list[tuple[int, int, str]] = [
    (1, 1, "spring_festival"),     # 春节（大年初一）
    (1, 15, "lantern_festival"),   # 元宵节（正月十五）——汤圆、花灯
    (2, 2, "longtaitou"),          # 龙抬头（二月初二）——猪头肉、理发
    (5, 5, "dragon_boat"),         # 端午节（五月初五）——粽子
    (7, 7, "qixi"),               # 七夕（七月初七）——巧克力、鲜花
    (8, 15, "mid_autumn"),         # 中秋节（八月十五）——月饼
    (9, 9, "chongyang"),           # 重阳节（九月初九）——保健品
    (12, 8, "laba"),               # 腊八节（腊月初八）——腊八粥原料
    (12, 23, "xiaonian"),          # 小年（腊月二十三）——年货采购开始
]

# ── 公历固定节日 ──
_SOLAR_FESTIVALS: list[tuple[int, int, str]] = [
    (1, 1, "new_year"),            # 元旦
    (2, 14, "valentines"),         # 情人节——巧克力、鲜花
    (3, 8, "womens_day"),          # 妇女节——部分品类促销
    (5, 1, "labor_day"),           # 劳动节
    (6, 1, "childrens_day"),       # 儿童节——零食、玩具
    (10, 1, "national_day"),       # 国庆节
    (12, 25, "christmas"),         # 圣诞节——年轻消费者礼品
]

# ── 按"第N个周几"定义的节日 ──
# 格式：(月, 第几个, 星期几(0=Mon,6=Sun), 节日名称)
_NTH_WEEKDAY_FESTIVALS: list[tuple[int, int, int, str]] = [
    (5, 2, 6, "mothers_day"),   # 母亲节：5月第2个周日——鲜花、蛋糕、保健品
    (6, 3, 6, "fathers_day"),   # 父亲节：6月第3个周日——烟酒、保健品
]


def _nth_weekday(year: int, month: int, nth: int, weekday: int) -> date:
    """计算某年某月第 N 个星期几的日期。
    weekday: 0=周一, 6=周日。
    """
    # 从该月1号开始找第一个目标星期几
    first = date(year, month, 1)
    # 距第一个目标weekday的天数
    days_ahead = weekday - first.weekday()
    if days_ahead < 0:
        days_ahead += 7
    first_target = first + timedelta(days=days_ahead)
    # 第 nth 个 = 第一个 + (nth-1) 周
    return first_target + timedelta(weeks=nth - 1)


def get_holiday_features(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate holiday features for a date range.

    Features:
    - is_holiday: True if official public holiday (from chinese-calendar)
    - is_festival: True if it's a traditional festival (lunar or solar)
    - holiday_name: Name of the holiday/festival
    - days_to_nearest_holiday: Days until the next holiday (continuous, for model to learn pre-holiday effects)
    - nearest_holiday_name: Which holiday is coming next (so model can differentiate Spring Festival vs others)
    - is_makeup_workday: True if this is a weekend that's been designated as a workday (调休)

    Args:
        date_range: Pandas DatetimeIndex to generate features for.

    Returns:
        DataFrame with date and holiday feature columns.
    """
    # 构建所有已知节假日/节日的日期字典
    all_holidays = _build_holiday_dict(date_range)

    # 获取官方假日和调休信息
    official_holidays, makeup_workdays = _get_official_holidays(date_range)

    records = []
    for dt in date_range:
        d = dt.date()

        # 是否官方假日（国务院公布的放假日）
        is_holiday = d in official_holidays

        # 是否传统节日（农历+公历）
        is_festival = d in all_holidays
        holiday_name = all_holidays.get(d, "")

        # 如果是官方假日但不在节日字典里（比如调休产生的额外假日），标记节日名
        if is_holiday and not holiday_name:
            holiday_name = official_holidays.get(d, "holiday")

        # 距最近节假日天数 + 最近节假日名称（连续特征，让模型自己学备货提前量）
        days_to, nearest_name = _days_to_nearest(d, all_holidays)

        # 是否调休工作日（周末但要上班）——影响客流
        is_makeup = d in makeup_workdays

        records.append({
            "date": dt,
            "is_holiday": is_holiday,
            "is_festival": is_festival,
            "holiday_name": holiday_name,
            "days_to_nearest_holiday": days_to,
            "nearest_holiday_name": nearest_name,
            "is_makeup_workday": is_makeup,
        })

    df = pd.DataFrame(records)
    n_holidays = df["is_holiday"].sum()
    n_festivals = df["is_festival"].sum()
    logger.info(
        "Generated holiday features for %d days (%d official holidays, %d festival days)",
        len(df), n_holidays, n_festivals,
    )
    return df


def _build_holiday_dict(date_range: pd.DatetimeIndex) -> dict[date, str]:
    """Build a dict mapping dates to holiday/festival names.

    Combines lunar festivals (converted to solar dates) and fixed solar festivals.
    """
    holidays: dict[date, str] = {}

    # 确定需要覆盖的年份范围
    # 前后各扩展1年：农历腊月节日（如腊八、小年）跨公历年，
    # 例如 LunarDate(2025, 12, 8) 对应公历 2026-01-26
    min_year = date_range.min().year - 1
    max_year = date_range.max().year + 1

    # 农历节日：每年转换一次公历日期
    min_date = date_range.min().date()
    max_date = date_range.max().date()
    try:
        from lunardate import LunarDate

        for year in range(min_year, max_year + 1):
            for lunar_month, lunar_day, name in _LUNAR_FESTIVALS:
                try:
                    solar = LunarDate(year, lunar_month, lunar_day).toSolarDate()
                    # 只保留在数据日期范围内的节日
                    if min_date <= solar <= max_date:
                        holidays[solar] = name
                except ValueError:
                    # 某些年份农历日期不存在（如闰月差异），跳过
                    continue
    except ImportError:
        logger.warning("lunardate not installed, skipping lunar festivals")

    # 公历固定节日
    for year in range(min_year, max_year + 1):
        for month, day, name in _SOLAR_FESTIVALS:
            try:
                holidays[date(year, month, day)] = name
            except ValueError:
                continue

    # 按"第N个周几"计算的节日（母亲节、父亲节等）
    for year in range(min_year, max_year + 1):
        for month, nth, weekday, name in _NTH_WEEKDAY_FESTIVALS:
            try:
                d = _nth_weekday(year, month, nth, weekday)
                if min_date <= d <= max_date:
                    holidays[d] = name
            except ValueError:
                continue

    return holidays


def _get_official_holidays(
    date_range: pd.DatetimeIndex,
) -> tuple[dict[date, str], set[date]]:
    """Get official public holidays and makeup workdays from chinese-calendar.

    Returns:
        Tuple of (official_holidays dict, makeup_workdays set).
    """
    official: dict[date, str] = {}
    makeup: set[date] = set()

    try:
        import chinese_calendar as cc

        for dt in date_range:
            d = dt.date()
            try:
                if cc.is_holiday(d):
                    # chinese-calendar 能返回节日名称
                    detail = cc.get_holiday_detail(d)
                    if detail[0]:  # is_holiday
                        name = detail[1].name if detail[1] else "holiday"
                        official[d] = name

                # 调休工作日：周六日但要上班
                if d.weekday() >= 5 and cc.is_workday(d):
                    makeup.add(d)
            except Exception:
                continue

    except ImportError:
        logger.warning("chinese-calendar not installed, using fallback holiday list")

    return official, makeup


def _days_to_nearest(
    d: date,
    holidays: dict[date, str],
) -> tuple[int, str]:
    """Find days to nearest future holiday and its name."""
    future = [(h, n) for h, n in holidays.items() if h > d]
    if not future:
        return 365, ""
    nearest_date, nearest_name = min(future, key=lambda x: x[0])
    return (nearest_date - d).days, nearest_name

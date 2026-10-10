"""事件类型的对照表：语料里的中文名 / `Data` 表的英文列名 → 本项目统一的 snake_case。

语料有两个标签空间，说的是同一件事：

- `Sheet1["事件类型"]` —— 中文（"美国对中国加征关税"）
- `Data` 的 25 个事件列 —— 英文标题式（"US_Tariff_Increases_on_China"）

第 2 周要把 445 条新闻落成事件库，两套名字都得能对上，所以对照集中放在这里，
别在各自的脚本里各写一份。
"""

from __future__ import annotations

from typing import Dict

# Sheet1 的中文事件类型 -> snake_case
CHINESE_TO_SNAKE: Dict[str, str] = {
    "中美对话": "bilateral_dialogue",
    "美国对中国加征关税": "us_tariff_increase",
    "美国对中国放宽关税政策": "us_tariff_relief",
    "美国对中国企业的限制": "us_restrict_cn_firms",
    "美国对中国企业降低限制": "us_ease_restrictions_cn_firms",
    "美国对中国发起诉讼或调查": "us_litigation_cn",
    "美国指责中国": "us_criticize_cn",
    "美国将中国列为汇率操纵国": "us_designate_cn_fx_manipulator",
    "美国将中国从汇率操纵名单剔除": "us_remove_cn_fx_manipulator",
    "中国加大对美国采购": "cn_increase_purchases",
    "中国对美国加征关税": "cn_tariff_increase",
    "中国对美国放宽关税政策": "cn_tariff_relief",
    "中国对美国企业的限制": "cn_restrict_us_firms",
    "中国对美国企业降低限制": "cn_ease_restrictions_us_firms",
    "中国对美国发起诉讼或调查": "cn_litigation_us",
    "中国指责美国": "cn_criticize_us",
}

# `Data` 表的英文列名 -> snake_case（名字与 `src/core/research_dataset.py` 规范后的一致）
COLUMN_TO_SNAKE: Dict[str, str] = {
    "Sino_US_Dialogue": "bilateral_dialogue",
    "US_Tariff_Increases_on_China": "us_tariff_increase",
    "US_Tariff_Relief_Policies_for_China": "us_tariff_relief",
    "US_Restrictions_on_Chinese_Enterprises": "us_restrict_cn_firms",
    "US_Reducing_Restrictions_on_Chinese_Enterprises": "us_ease_restrictions_cn_firms",
    "US_Litigation_or_Investigation_against_China": "us_litigation_cn",
    "US_Criticizing_China": "us_criticize_cn",
    "US_Designating_China_as_Currency_Manipulator": "us_designate_cn_fx_manipulator",
    "US_Removing_China_from_Currency_Manipulator_List": "us_remove_cn_fx_manipulator",
    "China_Increasing_Purchases_of_US_Goods": "cn_increase_purchases",
    "China_Tariff_Increases_on_the_US": "cn_tariff_increase",
    "China_Tariff_Relief_Policies_for_the_US": "cn_tariff_relief",
    "China_Restrictions_on_US_Enterprises": "cn_restrict_us_firms",
    "China_Reducing_Restrictions_on_US_Enterprises": "cn_ease_restrictions_us_firms",
    "China_Litigation_or_Investigation_against_the_US": "cn_litigation_us",
    "China_Criticizing_the_US": "cn_criticize_us",
}

SNAKE_TO_CHINESE: Dict[str, str] = {
    snake: chinese for chinese, snake in CHINESE_TO_SNAKE.items()
}


def label(name: str) -> str:
    """snake_case → 中文，给界面用。认不出就原样返回。"""
    return SNAKE_TO_CHINESE.get(name, name)


# 粗分组。**不是替换，是并存**——`event_type` 保留语料原本的 16 类不动，
# 另加一列 `event_family` 给样本量小的地方用。
#
# 加它的依据是第 1 周的审计：`us_restrict_cn_firms`(160) 和 `us_litigation_cn`(13)
# 之间的边界连大模型都分不准（"反倾销终裁" 被标成前者、而"发起诉讼或调查"是另一个
# 标签），更别说关键词。与其去猜哪边对，不如把语料的标签原样留着，
# 另外给一个粗到不会错的粒度。
FAMILY_LABELS: Dict[str, str] = {
    "tariff": "关税",
    "entity_restrictions": "企业限制",
    "litigation": "调查诉讼",
    "fx_manipulator": "汇率操纵认定",
    "rhetoric": "表态指责",
    "dialogue": "双边对话",
    "purchases": "采购承诺",
    "other": "其他",
}


def event_family(snake: str) -> str:
    """snake_case 事件类型 → 粗分组。

    用「哪一侧 + 什么动作」里的**动作**分组，不区分发起方——
    美方加征和中方加征在检索上属于同一类事件。
    """
    if snake == "bilateral_dialogue":
        return "dialogue"
    if "tariff" in snake:
        return "tariff"
    if "restrict" in snake:
        return "entity_restrictions"
    if "litigation" in snake:
        return "litigation"
    if "fx_manipulator" in snake:
        return "fx_manipulator"
    if "criticize" in snake:
        return "rhetoric"
    if "purchases" in snake:
        return "purchases"
    return "other"


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family)

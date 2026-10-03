"""Conservative fact-type guard supplementing multilingual LLM evidence selection."""

import re

QUANTITY = re.compile(
    r"\b(?:how much|what (?:is|was|were)|what['’]s|estimated|projected|expected|"
    r"berapa(?:kah)?)\b|多少|金额|金額|预计|預計|估计|估計|预估|預估",
    re.I,
)
MONEY = re.compile(
    r"\b(?:funding|funds|budget|costs?|prices?|profits?|revenues?|investment|salary|amount|"
    r"dana|peruntukan|belanjawan|kos|keuntungan|pelaburan|gaji)\b"
    r"|资金|資金|拨款|撥款|预算|預算|成本|利润|利潤|营收|營收|收入|价格|價格|投资|投資|金额|金額|费用|費用",
    re.I,
)
FINANCIAL_METRICS = (
    re.compile(r"\b(?:profits?|earnings|net income|keuntungan)\b|利润|利潤|盈利|净利|淨利", re.I),
    re.compile(r"\b(?:revenues?|sales|pendapatan)\b|营收|營收|收入|销售额|銷售額", re.I),
)
CURRENCY = re.compile(
    r"\b(?:RM|MYR|USD|SGD|CNY|RMB|EUR|GBP|ringgit|dollars?|yuan|euros?|pounds?)\b"
    r"|[$¥￥€£]|令吉|美元|人民币|人民幣|欧元|歐元|英镑|英鎊",
    re.I,
)
NUMBER = re.compile(
    r"\d|[零一二三四五六七八九十百千万亿〇]|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|million|billion|"
    r"satu|dua|tiga|empat|lima|enam|tujuh|lapan|sembilan|sepuluh|juta|bilion)\b",
    re.I,
)


def required_fact_present(question: str, excerpts: list[str]) -> bool:
    if QUANTITY.search(question) and MONEY.search(question):
        requested_metrics = [metric for metric in FINANCIAL_METRICS if metric.search(question)]
        # A product price is not evidence of company profit or revenue. Require each
        # requested metric alongside an amount in at least one selected excerpt.
        return all(
            any(
                CURRENCY.search(excerpt)
                and NUMBER.search(excerpt)
                and (metric is None or metric.search(excerpt))
                for excerpt in excerpts
            )
            for metric in requested_metrics or [None]
        )
    return True

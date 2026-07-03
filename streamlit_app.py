from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
import os
import random
import re
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from twin_core import (
    LINDA_OPTS,
    LINDA_TEXT,
    PROFILE_VARIANTS,
    SYSTEM_MESSAGE,
    accuracy,
    build_user_prompt,
    call_llm,
    candidate_eval_items,
    choose_eval_question,
    evaluate_items,
    find_question_by_id,
    get_persona_text,
    get_persona_variant,
    inspect_persona_rows,
    iter_answer_questions,
    load_data,
    mock_llm,
    persona_blocks,
    parse_option_number,
    simulate_answer,
    single_item_evaluation,
    summarize_prediction_table,
)


ROOT = Path(__file__).resolve().parent
ZH_CACHE_PATH = ROOT / "tmp" / "zh_translation_cache.json"
ZH_SYSTEM_MESSAGE = (
    "你是一名擅长预测特定个体如何回答问卷问题的人工智能专家。"
    "你会看到这个人的画像资料，也就是他/她过去的问卷作答记录，以及一个新的问题和回答格式要求。"
    "请严格根据画像推断该个体会如何作答，保持与画像一致，并严格遵守输出格式。"
)
ZH_FORMAT_INSTRUCTION = "只返回一个阿拉伯数字选项编号，不要输出任何其他文字、标点或解释。"
ZH_OPEN_FORMAT_INSTRUCTION = "直接给出该被试最可能写下的回答，不要解释推理过程。"
EMPTY_MODEL_OUTPUT = "（模型没有返回可展示内容；已自动重试仍为空。请提高 token 上限或关闭 thinking 后重试。）"
CHOICE_MAX_TOKENS = 30
SEGMENT_DIMENSIONS = ["价格敏感", "健康导向", "环保导向", "便利导向", "社会证明", "尝新倾向"]
MARKET_RESEARCH_DEFAULT_QUESTIONS = [
    "这个产品或概念最吸引你的地方是什么？",
    "你会在什么场景下考虑尝试它？",
    "你最大的顾虑或拒绝理由是什么？",
    "什么信息、优惠或证明会提高你的购买意向？",
    "你更希望通过什么渠道了解或购买它？",
]
ZH_VARIANT_LABELS = {
    "full_or_summary": "优先摘要画像",
    "full": "完整问卷画像",
    "demographics_only": "只用人口统计",
    "empty": "无画像基线",
}
ZH_TEXT_REPLACEMENTS = {
    "The following is a description of a person.": "以下是一位被试的描述。",
    "The person's demographics are the following": "这位被试的人口统计信息如下",
    "Geographic region": "地理区域",
    "Gender": "性别",
    "Age": "年龄",
    "Education level": "教育水平",
    "Race": "种族",
    "Citizen of the US": "美国公民身份",
    "Marital status": "婚姻状况",
    "Question Type": "题型",
    "Single Choice": "单选题",
    "Multiple Choice": "多选题",
    "Text Entry": "文本填答",
    "Formatted Text": "说明文本",
    "Mr. A": "A先生",
    "Mr. B": "B先生",
    "Mr. C": "C先生",
    "PERSONA PROFILE": "人物画像",
    "QUESTION": "问题",
    "Options": "选项",
    "FORMAT INSTRUCTIONS": "格式要求",
    "Q:": "题目：",
    "Answer:": "作答：",
    "No individual persona information is provided. Predict the answer using only the question and options.": "不提供任何个人画像信息。请只根据问题和选项预测回答。",
    "Demographics": "人口统计",
    "Personality": "人格",
    "Cognitive tests": "认知测试",
    "Economic preferences - intro": "经济偏好 - 引导",
    "Economic preferences": "经济偏好",
    "Forward Flow": "自由联想",
    "Female": "女性",
    "Male": "男性",
    "True": "真",
    "False": "假",
    "[no answer]": "未作答",
    "Which part of the United States do you currently live in?": "你目前居住在美国哪个地区？",
    "What is the sex that you were assigned at birth?": "你出生时被指定的性别是什么？",
    "How old are you?": "你的年龄是多少？",
    "South": "南部",
    "Northeast": "东北部",
    "Midwest": "中西部",
    "West": "西部",
    "Some college, no degree": "上过大学但未获得学位",
    "High school graduate": "高中毕业",
    "Bachelor's degree": "学士学位",
    "White": "白人",
    "Black or African American": "黑人或非裔美国人",
    "Asian": "亚裔",
    "Never married": "从未结婚",
    "Married": "已婚",
    "Very liberal": "非常自由派",
    "Liberal": "自由派",
    "Moderate": "中间派",
    "Conservative": "保守派",
    "Very conservative": "非常保守派",
    "Strongly disagree": "非常不同意",
    "Disagree": "不同意",
    "Neutral": "中立",
    "Neither agree nor disagree": "既不同意也不反对",
    "Agree a little": "有点同意",
    "Disagree a little": "有点不同意",
    "Agree strongly": "强烈同意",
    "Disagree strongly": "强烈不同意",
    "Agree": "同意",
    "Strongly agree": "非常同意",
    "No": "否",
    "Yes": "是",
    "TRUE": "真",
    "FALSE": "假",
    "Extremely improbable": "极不可能",
    "Very improbable": "很不可能",
    "Somewhat probable": "有点可能",
    "Moderately probable": "中等可能",
    "Very probable": "很可能",
    "Extremely probable": "极有可能",
    LINDA_TEXT: "琳达31岁，单身，坦率而且很聪明；她大学主修哲学，关心歧视与社会正义。琳达是一名银行柜员并积极参与女性主义运动的可能性是___。",
    "Imagine that there will be a deadly flu going around your area next winter. Your doctor says that you have a 10% chance (10 out of 100) of dying from this flu. However, a new flu vaccine has been developed and tested. If taken, the vaccine prevents you from catching the deadly flu. However, there is one serious risk involved with taking this vaccine. The vaccine is made from a somewhat weaker type of flu virus, and there is a 5% (5 out of 100) risk of the vaccine causing you to die from the weaker type of flu. Imagine that this vaccine is completely covered by health insurance. If you had to decide now, which would you choose?": "设想明年冬天你所在地区会流行一种致命流感。医生说，如果感染这种流感，你有10%（100人中10人）的概率死亡。现在一种新的流感疫苗已经研发并测试完成；接种后可以避免感染这种致命流感。但接种也有一个严重风险：疫苗由一种较弱的流感病毒制成，有5%（100人中5人）的概率会导致你死于这种较弱病毒。假设该疫苗完全由医保覆盖。如果你现在必须做决定，你会选择哪一项？",
    "I would definitely not take the vaccine. I would thus accept the 10% chance of dying from this flu.": "我肯定不会接种疫苗，因此接受死于这种流感的10%风险。",
    "I would probably not take the vaccine. I would thus accept the 10% chance of dying from this flu.": "我大概不会接种疫苗，因此接受死于这种流感的10%风险。",
    "I would probably take the vaccine. I would thus accept the 5% chance of dying from the weaker flu in the vaccine": "我大概会接种疫苗，因此接受死于疫苗中较弱流感病毒的5%风险。",
    "I would definitely take the vaccine. I would thus accept the 5% chance of dying from the weaker flu in the vaccine.": "我肯定会接种疫苗，因此接受死于疫苗中较弱流感病毒的5%风险。",
    "Assume that you are presented with two trays of black and white marbles, a large tray that contains 100 marbles and a small tray that contains 10 marbles. The marbles are spread in a single layer in each tray. You must draw out one marble (without peeking, of course) from either tray. If you draw a black marble you win $2. Consider a condition in which the small tray contains 1 black marble and 9 white marbles, and the large tray contains 8 black marbles and 92 white marbles. From which tray would you prefer to select a marble in a real situation?": "假设你面前有两个装有黑白弹珠的托盘：一个大托盘有100颗弹珠，一个小托盘有10颗弹珠。每个托盘中的弹珠都平铺成一层。你必须从其中一个托盘中抽出一颗弹珠（当然不能偷看）。如果抽到黑色弹珠，你将赢得2美元。现在小托盘中有1颗黑色弹珠和9颗白色弹珠，大托盘中有8颗黑色弹珠和92颗白色弹珠。在真实情境下，你更愿意从哪个托盘抽取弹珠？",
    "the small tray": "小托盘",
    "the large tray": "大托盘",
    "Yes, I would purchase the product": "是，我会购买该商品",
    "No, I would not purchase the product": "否，我不会购买该商品",
    "Coca-Cola Classic Soda, 12 fl oz, 12-pack": "可口可乐经典汽水，12液量盎司，12罐装",
    "money; dollar; cent; dime; nickel; penny": "钱；美元；美分；一角硬币；五美分硬币；一美分硬币",
    "myself; themselves; ourself; ourselves; self; other": "我自己；他们自己；我们自己；我们自己；自我；他人",
    "A panel of psychologist have interviewed and administered personality tests to 70 engineers and 30 lawyers, all successful in their respective fields. On the basis of this information, thumbnail descriptions of the 70 engineers and 30 lawyers have been written. Below is one description, chosen at random from the 100 available descriptions. Jack is a 45-year-old man. He is married and has four children. He is generally conservative, careful, and ambitious. He shows no interest in political and social issues and spends most of his free time on his many hobbies which include home carpentry, sailing, and mathematical puzzles. The probability that Jack is one of the 70 engineers in the sample of 100 is ___%. Please indicate the probability on a scale from 0 to 100.": "一组心理学家访谈了70名工程师和30名律师，并对他们进行了人格测试；这些人都在各自领域很成功。根据这些信息，研究者为这100人写下了简短描述。下面是一段从100份描述中随机抽取的描述。杰克45岁，已婚，有4个孩子。他总体上保守、谨慎、有抱负。他对政治和社会议题没有兴趣，大部分闲暇时间都花在许多爱好上，包括家庭木工、帆船和数学谜题。你认为杰克是这100人样本中70名工程师之一的概率是多少？请在0到100的范围内给出百分比。",
    "How many African countries do you think are in the United Nations?": "你认为联合国中有多少个非洲国家？",
    "How tall do you think the tallest redwood tree in the world is? Enter a number of feet.": "你认为世界上最高的红杉树有多高？请输入英尺数。",
    "Imagine that you just paid $50 for a Coffee Connection discount card that allows you to buy coffee for 50% off the regular price of $3.00 (i.e., you pay $1.50). Soon after you purchased the Coffee Connection discount card, Java Coffee, a competitor, opened a new store that sells coffee for just $2.00 per cup. Although the Coffee Connection store is ten minutes away by car, Java Coffee is only about 1/2 block from your apartment. Assuming that you only buy coffee from these two places and that you like the coffee sold in both places the same, how many of your next 20 coffee purchases would be from Java Coffee? Enter a number between 0 and 20.": "设想你刚花50美元购买了一张 Coffee Connection 折扣卡，可以用常规价格3.00美元的五折购买咖啡，也就是每杯1.50美元。不久之后，竞争对手 Java Coffee 开了一家新店，每杯咖啡只卖2.00美元。Coffee Connection 门店开车需要10分钟，而 Java Coffee 离你的公寓只有大约半个街区。假设你只从这两家店买咖啡，并且你同样喜欢两家的咖啡，那么接下来20次购买咖啡中，你会有多少次从 Java Coffee 购买？请输入0到20之间的数字。",
    "What percentage of the public do you think supports the following policies? For each policy, choose a number from 0% to 100%.": "你认为公众中有百分之多少的人支持以下政策？请为每项政策选择0%到100%之间的数字。",
}


st.set_page_config(
    page_title="Twin-2K-500 Interactive",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2.4rem; }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; }
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(49, 51, 63, 0.12); }
    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
    .wrapped-prompt {
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 6px;
        padding: 0.85rem 1rem;
        background: rgba(250, 250, 252, 0.88);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_data(num_personas: int, use_hub: bool):
    return load_data(num_personas=num_personas, root=ROOT, use_hub=use_hub)


def llm_kwargs(api_key: str, base_url: str, model: str, max_tokens: int, thinking_mode: str | None = None) -> dict:
    if thinking_mode is None:
        thinking_mode = st.session_state.get("thinking_mode", "disabled")
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "max_tokens": max_tokens,
        "thinking": thinking_mode,
    }


def choice_llm_kwargs(kwargs: dict) -> dict:
    fast = dict(kwargs)
    fast["max_tokens"] = CHOICE_MAX_TOKENS
    fast["strict_max_tokens"] = True
    return fast


def default_api_key() -> str:
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets.get("DEEPSEEK_API_KEY", "")).strip()
    except Exception:
        return ""


def option_editor(default_options: list[str]) -> list[str]:
    data = pd.DataFrame({"option": default_options})
    edited = st.data_editor(
        data,
        hide_index=True,
        width="stretch",
        num_rows="dynamic",
        key="custom_options",
    )
    return [str(value).strip() for value in edited["option"].tolist() if str(value).strip()]


def zh_variant_label(variant_name: str) -> str:
    return ZH_VARIANT_LABELS.get(str(variant_name), str(variant_name))


def persona_block_summary(blocks: list[dict]) -> pd.DataFrame:
    records = []
    for block in blocks:
        questions = block.get("Questions", [])
        q_types = sorted({str(question.get("QuestionType", "?")) for question in questions})
        sample = questions[0].get("QuestionText") if questions else ""
        records.append(
            {
                "block": block.get("BlockName", "Section"),
                "questions": len(questions),
                "types": ", ".join(q_types),
                "sample_question": sample,
            }
        )
    return pd.DataFrame(records)


def profile_size_table(row) -> pd.DataFrame:
    variants = ["full_or_summary", "full", "demographics_only", "empty"]
    records = []
    for name in variants:
        text = get_persona_variant(row, name)
        records.append(
            {
                "variant": name,
                "label": zh_variant_label(name),
                "characters": len(text),
                "lines": len(text.splitlines()),
                "preview": text.replace("\n", " ")[:180],
            }
        )
    return pd.DataFrame(records)


def zh_question_types(types_text: str) -> str:
    mapping = {
        "MC": "选择题",
        "TE": "文本题",
        "Matrix": "矩阵题",
        "DB": "说明页",
        "Slider": "滑杆题",
        "?": "未知",
    }
    return ", ".join(mapping.get(part.strip(), part.strip()) for part in str(types_text).split(","))


def zh_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "pid": "被试ID",
            "qid": "题目ID",
            "variant": "画像版本",
            "predictor": "预测器",
            "truth": "真实答案",
            "retest": "人类重测",
            "twin": "数字孪生",
            "twin_label": "孪生答案",
            "random": "随机基线",
            "raw_twin": "模型原始输出",
            "truth_text": "真实答案",
            "retest_text": "人类重测",
            "twin_answer": "数字孪生回答",
            "question_type": "题型",
            "acc_twin": "孪生准确率",
            "acc_retest": "人类重测准确率",
            "lo": "下界",
            "hi": "上界",
            "options": "选项数",
            "question": "题干",
            "metric": "指标",
            "accuracy": "准确率",
            "correlation": "相关性",
            "mean_diff_sd": "均值偏差",
            "variance_ratio": "方差比例",
            "value": "值",
        }
    )


def demand_line_chart(demand_df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(demand_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("price_ratio:Q", title="价格 / 常规价格"),
            y=alt.Y("P(buy):Q", title="购买比例", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("price_ratio:Q", format=".2f"),
                alt.Tooltip("price_$:Q", format=".2f"),
                alt.Tooltip("P(buy):Q", format=".3f"),
            ],
        )
        .properties(height=360)
    )


@st.cache_resource(show_spinner=False)
def translation_cache(cache_mtime: float) -> dict:
    if ZH_CACHE_PATH.exists():
        try:
            return json.loads(ZH_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_translation_cache(cache: dict) -> None:
    ZH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ZH_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def translation_cache_mtime() -> float:
    return ZH_CACHE_PATH.stat().st_mtime if ZH_CACHE_PATH.exists() else 0.0


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def code_like_answer(text: str) -> bool:
    return bool(re.fullmatch(r"\s*[A-G](?:\s*,\s*(?:[A-G]|\d+))*\s*", text or ""))


def needs_translation(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or "")) and not has_chinese(text) and not code_like_answer(text)


def english_residue_words(text: str) -> list[str]:
    cleaned = re.sub(r"https?://\S+|score_[A-Za-z0-9_-]+|wave\d*_[A-Za-z0-9_-]+|QID\d+", " ", text or "")
    cleaned = re.sub(r"\$?[0-9]+(?:\.[0-9]+)?%?|\b[A-Z]{1,3}\b", " ", cleaned)
    allow = {
        "api",
        "coffee",
        "connection",
        "crt",
        "deepseek",
        "green",
        "html",
        "java",
        "json",
        "llm",
        "mock",
        "thaler",
        "twin",
        "wason",
        "wave",
    }
    return [
        word
        for word in re.findall(r"\b[A-Za-z][A-Za-z'’-]{2,}\b", cleaned)
        if word.lower().strip("'’-") not in allow
    ]


def translation_is_usable(source: str, translated: str) -> bool:
    if not needs_translation(source):
        return True
    if not has_chinese(translated):
        return False
    return len(english_residue_words(translated)) <= 8


def local_zh_fallback(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    product_match = re.fullmatch(
        r"Please consider the following product category: (.+?)\. Suppose you are in a grocery store, and you see the following product in that category: (.+?)\. The product is priced at: \$([0-9.]+)\. Would you or would you not purchase this product\?",
        text.strip(),
    )
    if product_match:
        category, product, price = product_match.groups()
        return f"请考虑以下商品类别：{category}。假设你在一家杂货店，看到了该类别中的以下商品：{product}。该商品价格为：${price}。你会购买这件商品吗？"
    result = text
    for source, target in sorted(ZH_TEXT_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        if source in {"No", "Yes"}:
            result = re.sub(rf"\b{re.escape(source)}\b", target, result)
        else:
            result = result.replace(source, target)
    return result


def split_text_for_translation(text: str, max_chars: int = 2200) -> list[str]:
    parts = re.split(r"(\n\s*\n)", text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) <= max_chars:
            current += part
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(part) <= max_chars:
            current = part
            continue
        lines = part.splitlines(keepends=True) or [part]
        for line in lines:
            if len(current) + len(line) > max_chars and current:
                chunks.append(current)
                current = ""
            if len(line) > max_chars:
                for i in range(0, len(line), max_chars):
                    chunks.append(line[i : i + max_chars])
            else:
                current += line
    if current:
        chunks.append(current)
    return chunks


def translate_to_chinese(text: str, api_key: str, base_url: str, model: str, *, context: str = "") -> str:
    if not isinstance(text, str) or not text.strip() or has_chinese(text):
        return text
    fallback = local_zh_fallback(text)

    cache = translation_cache(translation_cache_mtime())
    out_chunks = []
    changed = False
    for chunk in split_text_for_translation(text):
        cache_key = hashlib.sha256(("zh-v2|" + chunk).encode("utf-8")).hexdigest()
        if cache_key in cache:
            cached = cache[cache_key]
            if translation_is_usable(chunk, cached):
                out_chunks.append(cached)
                continue
            if not api_key.strip() or not st.session_state.get("auto_translate_uncached", False):
                out_chunks.append(local_zh_fallback(cached if has_chinese(cached) else chunk))
                continue
        elif not api_key.strip() or not st.session_state.get("auto_translate_uncached", False):
            out_chunks.append(local_zh_fallback(chunk))
            continue
        try:
            translated = call_llm(
                "你是严谨的英译中助手。请把用户提供的内容翻译成自然、准确的简体中文；保留编号、换行、JSON键名、题目ID和选项顺序；不要添加解释。",
                chunk,
                api_key.strip(),
                base_url,
                model,
                max_tokens=min(3500, max(500, int(len(chunk) * 1.4) + 200)),
            )
        except Exception as exc:
            st.session_state["translation_error"] = str(exc)
            translated = local_zh_fallback(chunk)
        if not translation_is_usable(chunk, translated):
            translated = local_zh_fallback(translated if has_chinese(translated) else chunk)
        cache[cache_key] = translated
        changed = True
        out_chunks.append(translated)
    if changed:
        save_translation_cache(cache)
    return "".join(out_chunks)


def zh_text(text: str, *, context: str = "") -> str:
    if not st.session_state.get("use_chinese_text", True):
        return text
    return translate_to_chinese(text, st.session_state.get("api_key", ""), st.session_state.get("base_url", ""), st.session_state.get("model", ""), context=context)


def zh_options(options: list[str], *, context: str = "") -> list[str]:
    return [zh_text(str(option), context=f"{context}:option:{i}") for i, option in enumerate(options)]


def visible_model_output(raw: str | None) -> str:
    text = "" if raw is None else str(raw).strip()
    return text or EMPTY_MODEL_OUTPUT


def safe_error_message(exc: Exception) -> str:
    return re.sub(r"sk-[A-Za-z0-9]{12,}", "sk-***", f"{type(exc).__name__}: {exc}")


def clean_question_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def answer_text(question: dict | None) -> str | None:
    if not question:
        return None
    answers = question.get("Answers") or {}
    if isinstance(answers.get("Text"), str) and answers["Text"].strip():
        return answers["Text"].strip()
    values = answers.get("Values")
    if isinstance(values, list) and values:
        return ", ".join(str(value).strip() for value in values if str(value).strip()) or None
    selected = answers.get("SelectedText")
    if isinstance(selected, list) and selected:
        return ", ".join(str(value).strip() for value in selected if str(value).strip()) or None
    if isinstance(selected, str) and selected.strip():
        return selected.strip()
    return None


def candidate_open_answer_items(rows: list[dict], max_items: int = 80) -> list[dict]:
    common = None
    examples = {}
    for row in rows:
        gt_ids, rt_ids = set(), set()
        for source, target in [(row["ground_truth"], gt_ids), (row["retest"], rt_ids)]:
            for question in iter_answer_questions(source):
                qid = question.get("QuestionID")
                q_type = question.get("QuestionType")
                text = clean_question_text(question.get("QuestionText"))
                if q_type in {"TE", "Slider"} and qid and text and answer_text(question) is not None:
                    target.add(qid)
                    examples.setdefault(qid, question)
        ids = gt_ids & rt_ids
        common = ids if common is None else common & ids
    items = []
    for qid in sorted(common or set())[:max_items]:
        question = examples[qid]
        items.append(
            {
                "qid": qid,
                "text": clean_question_text(question.get("QuestionText")),
                "question_type": question.get("QuestionType", "TE"),
            }
        )
    return items


def build_user_prompt_for_language(persona_text: str, question_text: str, options: list[str]) -> str:
    if st.session_state.get("use_chinese_text", True):
        opt_block = "选项：\n" + "\n".join(f"  {i + 1} = {option}" for i, option in enumerate(options)) + "\n"
        return (
            f"人物画像：\n{persona_text}\n\n"
            f"问题：{question_text}\n\n"
            f"{opt_block}\n"
            f"格式要求：{ZH_FORMAT_INSTRUCTION}"
        )
    return build_user_prompt(persona_text, question_text, options)


def build_open_prompt_for_language(persona_text: str, question_text: str) -> str:
    if st.session_state.get("use_chinese_text", True):
        return (
            f"人物画像：\n{persona_text}\n\n"
            f"开放题：{question_text}\n\n"
            f"格式要求：{ZH_OPEN_FORMAT_INSTRUCTION}"
        )
    return (
        f"PERSONA PROFILE:\n{persona_text}\n\n"
        f"OPEN-ENDED QUESTION: {question_text}\n\n"
        "FORMAT INSTRUCTIONS: Directly provide the answer this persona would most likely write. "
        "Do not explain your reasoning."
    )


def simulate_answer_for_language(persona_text: str, question_text: str, options: list[str], **kwargs):
    fast_kwargs = choice_llm_kwargs(kwargs)
    if st.session_state.get("use_chinese_text", True):
        user_prompt = build_user_prompt_for_language(persona_text, question_text, options)
        if fast_kwargs.get("api_key", "").strip():
            raw = call_llm(
                ZH_SYSTEM_MESSAGE,
                user_prompt,
                fast_kwargs["api_key"].strip(),
                fast_kwargs["base_url"],
                fast_kwargs["model"],
                max_tokens=fast_kwargs["max_tokens"],
                thinking=fast_kwargs["thinking"],
                strict_max_tokens=fast_kwargs.get("strict_max_tokens", False),
            )
        else:
            raw = mock_llm(ZH_SYSTEM_MESSAGE, user_prompt, n_options=len(options) if options else None)
        raw = visible_model_output(raw)
        return parse_option_number(raw, len(options) if options else None), raw
    twin, raw = simulate_answer(persona_text, question_text, options, **fast_kwargs)
    return twin, visible_model_output(raw)


def simulate_open_answer_for_language(persona_text: str, question_text: str, **kwargs) -> str:
    user_prompt = build_open_prompt_for_language(persona_text, question_text)
    if kwargs.get("api_key", "").strip():
        system = ZH_SYSTEM_MESSAGE if st.session_state.get("use_chinese_text", True) else SYSTEM_MESSAGE
        raw = call_llm(
            system,
            user_prompt,
            kwargs["api_key"].strip(),
            kwargs["base_url"],
            kwargs["model"],
            max_tokens=max(80, kwargs.get("max_tokens", 80)),
            thinking=kwargs.get("thinking", "disabled"),
        ).strip()
        return visible_model_output(raw)
    if st.session_state.get("use_chinese_text", True):
        return "我会根据自己的经历和偏好，给出一个简短、直接的回答。"
    return "I would give a brief answer based on my own experiences and preferences."


def simulate_single_choice_answer(persona_text: str, question_text: str, options: list[str], use_chinese: bool, kwargs: dict) -> tuple[int | None, str]:
    fast_kwargs = dict(kwargs) if kwargs.get("thinking") == "enabled" else choice_llm_kwargs(kwargs)
    if fast_kwargs.get("thinking") == "enabled":
        fast_kwargs["strict_max_tokens"] = False
    user_prompt = demand_prompt(persona_text, question_text, options, use_chinese)
    n_options = len(options) if options else None
    if fast_kwargs.get("api_key", "").strip():
        raw = call_llm(
            ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE,
            user_prompt,
            fast_kwargs["api_key"].strip(),
            fast_kwargs["base_url"],
            fast_kwargs["model"],
            max_tokens=fast_kwargs["max_tokens"],
            thinking=fast_kwargs["thinking"],
            strict_max_tokens=fast_kwargs.get("strict_max_tokens", False),
        )
    else:
        raw = mock_llm(ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE, user_prompt, n_options=n_options)
    raw = visible_model_output(raw)
    return parse_option_number(raw, n_options), raw


def simulate_single_open_answer(persona_text: str, question_text: str, use_chinese: bool, kwargs: dict) -> str:
    if use_chinese:
        user_prompt = (
            f"人物画像：\n{persona_text}\n\n"
            f"开放题：{question_text}\n\n"
            f"格式要求：{ZH_OPEN_FORMAT_INSTRUCTION}"
        )
    else:
        user_prompt = (
            f"PERSONA PROFILE:\n{persona_text}\n\n"
            f"OPEN-ENDED QUESTION: {question_text}\n\n"
            "FORMAT INSTRUCTIONS: Directly provide the answer this persona would most likely write. "
            "Do not explain your reasoning."
        )
    if kwargs.get("api_key", "").strip():
        raw = call_llm(
            ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE,
            user_prompt,
            kwargs["api_key"].strip(),
            kwargs["base_url"],
            kwargs["model"],
            max_tokens=max(80, kwargs.get("max_tokens", 80)),
            thinking=kwargs.get("thinking", "disabled"),
        )
        return visible_model_output(raw)
    return "我会根据自己的经历和偏好，给出一个简短、直接的回答。" if use_chinese else "I would give a brief answer based on my own experiences and preferences."


def demand_prompt(persona_text: str, question_text: str, options: list[str], use_chinese: bool) -> str:
    if use_chinese:
        opt_block = "选项：\n" + "\n".join(f"  {i + 1} = {option}" for i, option in enumerate(options)) + "\n"
        return (
            f"人物画像：\n{persona_text}\n\n"
            f"问题：{question_text}\n\n"
            f"{opt_block}\n"
            f"格式要求：{ZH_FORMAT_INSTRUCTION}"
        )
    return build_user_prompt(persona_text, question_text, options)


def simulate_demand_answer(persona_text: str, question_text: str, options: list[str], use_chinese: bool, kwargs: dict) -> tuple[int | None, str]:
    fast_kwargs = choice_llm_kwargs(kwargs)
    user_prompt = demand_prompt(persona_text, question_text, options, use_chinese)
    n_options = len(options) if options else None
    if fast_kwargs.get("api_key", "").strip():
        system = ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE
        raw = call_llm(
            system,
            user_prompt,
            fast_kwargs["api_key"].strip(),
            fast_kwargs["base_url"],
            fast_kwargs["model"],
            max_tokens=fast_kwargs["max_tokens"],
            thinking=fast_kwargs["thinking"],
            strict_max_tokens=fast_kwargs.get("strict_max_tokens", False),
        )
    else:
        raw = mock_llm(ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE, user_prompt, n_options=n_options)
    return parse_option_number(raw, n_options), raw


def app_stable_rng(text: str) -> random.Random:
    seed = int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def coerce_yes_no_choice(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return 2 if value else 1
    if isinstance(value, (int, float)) and not pd.isna(value):
        number = int(round(value))
        if number in {1, 2}:
            return number
        if number == 0:
            return 1
        return default
    text = str(value).strip().lower()
    match = re.search(r"(?<!\d)([12])(?:\.0+)?(?!\d)", text)
    if match:
        return int(match.group(1))
    no_markers = ["不会", "不买", "不购买", "否", "不是", "no", "false", "reject", "would not", "not buy"]
    yes_markers = ["会购买", "会买", "购买", "是", "yes", "true", "accept", "would buy", "buy"]
    if any(marker in text for marker in no_markers):
        return 1
    if any(marker in text for marker in yes_markers):
        return 2
    return default


def build_demand_curve_prompt(persona_text: str, product: str, regular_price: float, ratios: list[float], use_chinese: bool) -> str:
    price_lines = "\n".join(
        f"- price_ratio={float(ratio):.2f}, price=${float(ratio) * float(regular_price):.2f}"
        for ratio in ratios
    )
    if use_chinese:
        return (
            f"人物画像：\n{persona_text}\n\n"
            f"商品：{product}\n"
            f"常规价格：${float(regular_price):.2f}\n"
            f"要评估的价格点：\n{price_lines}\n\n"
            "请一次性预测这个人在每个价格点是否会购买。"
            "请保持同一个人的偏好一致：价格越高通常不应更容易购买，除非画像给出特殊理由。"
            "每个价格点只允许用 1=不会购买，2=会购买。"
            "只输出一个JSON对象，格式为："
            "{\"answers\":[{\"price_ratio\":0.0,\"answer\":2},{\"price_ratio\":0.2,\"answer\":2}]}。"
        )
    return (
        f"PERSONA PROFILE:\n{persona_text}\n\n"
        f"Product: {product}\n"
        f"Regular price: ${float(regular_price):.2f}\n"
        f"Price points:\n{price_lines}\n\n"
        "Predict whether this same person would buy at each price point. Keep preferences internally consistent; higher prices should usually not increase purchase likelihood. "
        "Use only 1=No and 2=Yes for each price point. Output only one JSON object: "
        "{\"answers\":[{\"price_ratio\":0.0,\"answer\":2},{\"price_ratio\":0.2,\"answer\":2}]}."
    )


def parse_demand_curve_response(raw: str, ratios: list[float]) -> dict[float, int]:
    text = (raw or "").strip()
    parsed = {}
    payload = text
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        payload = match.group(0)
    else:
        match = re.search(r"\[.*\]", text, flags=re.S)
        if match:
            payload = match.group(0)
    try:
        data = json.loads(payload)
    except Exception:
        data = None

    entries = None
    if isinstance(data, dict):
        entries = payload_value(data, ["answers", "results", "decisions", "回答", "结果", "购买决策"])
        if entries is None:
            for key, value in data.items():
                try:
                    ratio = round(float(key), 6)
                except Exception:
                    continue
                answer = coerce_yes_no_choice(value)
                if answer in {1, 2}:
                    parsed[ratio] = answer
    elif isinstance(data, list):
        entries = data

    if isinstance(entries, dict):
        entries = [{"price_ratio": key, "answer": value} for key, value in entries.items()]
    if isinstance(entries, list):
        for index, item in enumerate(entries):
            ratio = None
            answer = None
            if isinstance(item, dict):
                ratio_value = payload_value(item, ["price_ratio", "ratio", "价格比例", "价格倍数"])
                if ratio_value is not None:
                    try:
                        ratio = round(float(ratio_value), 6)
                    except Exception:
                        ratio = None
                answer = coerce_yes_no_choice(payload_value(item, ["answer", "decision", "choice", "buy", "would_buy", "购买", "会购买"]))
            else:
                answer = coerce_yes_no_choice(item)
            if ratio is None and index < len(ratios):
                ratio = round(float(ratios[index]), 6)
            if ratio is not None and answer in {1, 2}:
                parsed[ratio] = answer
    return parsed


def fallback_demand_curve_answers(persona_text: str, product: str, regular_price: float, ratios: list[float]) -> dict[float, int]:
    rng = app_stable_rng(f"{persona_text}\n{product}\n{regular_price}")
    reservation_ratio = 0.45 + rng.random() * 1.25
    if re.search(r"健康|低糖|运动|营养|health|diet|fitness", persona_text + product, flags=re.I):
        reservation_ratio += 0.15
    if re.search(r"价格|省钱|便宜|折扣|price|cheap|discount|budget", persona_text, flags=re.I):
        reservation_ratio -= 0.15
    reservation_ratio = min(max(reservation_ratio, 0.25), 2.2)
    return {
        round(float(ratio), 6): (2 if float(ratio) <= reservation_ratio else 1)
        for ratio in ratios
    }


def simulate_demand_curve_for_persona(
    persona_text: str,
    product: str,
    regular_price: float,
    ratios: list[float],
    use_chinese: bool,
    kwargs: dict,
) -> list[dict]:
    raw = ""
    source = "备用规则"
    parsed = {}
    if kwargs.get("api_key", "").strip():
        prompt = build_demand_curve_prompt(persona_text, product, regular_price, ratios, use_chinese)
        try:
            raw = call_llm(
                ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE,
                prompt,
                kwargs["api_key"].strip(),
                kwargs["base_url"],
                kwargs["model"],
                max_tokens=max(260, kwargs.get("max_tokens", 260)),
                thinking=kwargs.get("thinking", "disabled"),
            )
            parsed = parse_demand_curve_response(raw, ratios)
            source = "真实API" if len(parsed) == len(ratios) else "真实API+备用规则"
        except Exception as exc:
            raw = f"调用失败：{safe_error_message(exc)}"
            source = "备用规则"
    else:
        raw = "Mock：本地规则生成"
        source = "Mock"

    fallback = fallback_demand_curve_answers(persona_text, product, regular_price, ratios)
    rows = []
    for ratio in ratios:
        ratio_key = round(float(ratio), 6)
        answer = parsed.get(ratio_key, fallback[ratio_key])
        buy = 1 if answer == 2 else 0
        rows.append(
            {
                "price_ratio": float(ratio),
                "price_$": round(float(ratio) * float(regular_price), 2),
                "answer": answer,
                "decision": "是" if buy and use_chinese else ("否" if use_chinese else ("Yes" if buy else "No")),
                "buy": buy,
                "raw": raw,
                "来源": source if ratio_key in parsed else ("备用规则" if kwargs.get("api_key", "").strip() else source),
            }
        )
    return rows


def normalize_payload_key(key: object) -> str:
    return re.sub(r"[\s_\-:：/（）()]+", "", str(key or "").strip().lower())


def payload_value(data: dict, keys: list[str]):
    for key in keys:
        if key in data:
            return data[key]
    normalized = {normalize_payload_key(key): value for key, value in data.items()}
    for key in keys:
        value = normalized.get(normalize_payload_key(key))
        if value is not None:
            return value
    return None


def coerce_scale_score(value, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and not pd.isna(value):
        return min(max(int(round(value)), 1), 5)
    text = str(value).strip()
    match = re.search(r"(?<!\d)([1-5])(?:\.0+)?(?:\s*/\s*5|分|级)?(?!\d)", text)
    if not match:
        return default
    return min(max(int(match.group(1)), 1), 5)


def build_strategy_prompt(persona_text: str, context: str, strategy_name: str, strategy_message: str, use_chinese: bool) -> str:
    if use_chinese:
        return (
            f"人物画像：\n{persona_text}\n\n"
            f"管理情境：{context}\n"
            f"策略名称：{strategy_name}\n"
            f"策略方案：{strategy_message}\n\n"
            "请预测这个人看到该策略后的真实反应。请像管理学中的选择实验一样权衡：价格/成本、便利性、信任、风险、个人价值观、替代品和既有偏好。\n"
            "评分必须校准，并尽量拉开不同人和不同策略之间的差异，不要把3当成默认答案：\n"
            "1=明确不会接受；2=倾向不会接受；3=犹豫或需要更多信息；4=倾向接受；5=强烈接受。\n"
            "如果画像中有价格、健康、环保、口碑、便利、尝新等正向线索，可以给4；只有强正向线索才给5。"
            "如果画像证据不足但没有明显反感，给3；如果有反向线索，给1或2。\n"
            "请只输出一个JSON对象，格式为：{\"score\": 1-5, \"reason\": \"主要理由\", \"concern\": \"主要顾虑\", \"implication\": \"管理启示\"}。"
        )
    return (
        f"PERSONA PROFILE:\n{persona_text}\n\n"
        f"Management context: {context}\n"
        f"Strategy name: {strategy_name}\n"
        f"Strategy details: {strategy_message}\n\n"
        "Predict this person's realistic response as in a management choice experiment. Balance price/cost, convenience, trust, risk, values, alternatives, and existing preferences.\n"
        "Calibrate the score and separate people/strategies when evidence differs: 1=definitely reject, 2=probably reject, 3=uncertain/needs more information, 4=probably accept, 5=strongly accept. Use 4 when persona evidence supports the strategy, 5 only for strong evidence, and 1-2 for counter-evidence.\n"
        "Only output one JSON object: {\"score\": 1-5, \"reason\": \"main reason\", \"concern\": \"main concern\", \"implication\": \"management implication\"}."
    )


def parse_strategy_response(raw: str) -> dict:
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    payload = match.group(0) if match else text
    try:
        data = json.loads(payload)
    except Exception:
        data = {}
    score = coerce_scale_score(payload_value(data, ["score", "评分", "接受度", "意向", "acceptance", "rating"]), default=None)
    if score is None:
        score = coerce_scale_score(text, default=None)
    return {
        "score": score,
        "reason": str(data.get("reason") or data.get("主要理由") or "").strip(),
        "concern": str(data.get("concern") or data.get("主要顾虑") or "").strip(),
        "implication": str(data.get("implication") or data.get("管理启示") or "").strip(),
        "raw": raw,
    }


def simulate_strategy_answer(
    persona_text: str,
    context: str,
    strategy_name: str,
    strategy_message: str,
    use_chinese: bool,
    kwargs: dict,
) -> dict:
    prompt = build_strategy_prompt(persona_text, context, strategy_name, strategy_message, use_chinese)
    if kwargs.get("api_key", "").strip():
        raw = call_llm(
            ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE,
            prompt,
            kwargs["api_key"].strip(),
            kwargs["base_url"],
            kwargs.get("model", "deepseek-v4-pro"),
            max_tokens=max(180, kwargs.get("max_tokens", 180)),
            thinking=kwargs.get("thinking", "disabled"),
        )
    else:
        rng = random.Random(hash(persona_text + strategy_name + strategy_message))
        score = min(max(3 + rng.choice([-1, 0, 0, 1]), 1), 5)
        raw = json.dumps(
            {
                "score": score,
                "reason": "画像信息显示该策略可能部分匹配其偏好。",
                "concern": "仍可能受到价格、信任或替代选择影响。",
                "implication": "需要进一步细分人群并测试具体文案。",
            },
            ensure_ascii=False,
        )
    return parse_strategy_response(raw)


def summarize_strategy_results(results_df: pd.DataFrame, high_threshold: int) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame(columns=["策略", "平均接受度", "高意向比例", "样本数", "有效评分数"])
    data = results_df.copy()
    if "策略" not in data.columns:
        data["策略"] = "未命名策略"
    data["接受度"] = pd.to_numeric(data.get("接受度"), errors="coerce")
    data["高意向"] = data["接受度"].ge(high_threshold).where(data["接受度"].notna())
    return (
        data.groupby("策略", as_index=False)
        .agg(平均接受度=("接受度", "mean"), 高意向比例=("高意向", "mean"), 样本数=("被试ID", "count"), 有效评分数=("接受度", "count"))
        .sort_values(["高意向比例", "平均接受度"], ascending=False)
    )


def fallback_strategy_summary(summary_df: pd.DataFrame, results_df: pd.DataFrame, context: str) -> str:
    if summary_df.empty:
        return "暂无可总结的沙盘结果。"
    ordered = summary_df.sort_values(["高意向比例", "平均接受度"], ascending=False)
    best = ordered.iloc[0]
    worst = ordered.iloc[-1]
    top_concerns = (
        results_df.get("主要顾虑", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.strip().ne("")]
        .head(5)
        .tolist()
    )
    concerns = "；".join(top_concerns) if top_concerns else "需要进一步查看个体级理由。"
    return (
        f"在「{context}」情境下，当前模拟中表现最好的策略是「{best['策略']}」，"
        f"高意向比例约为 {best['高意向比例']:.1%}，平均接受度为 {best['平均接受度']:.2f}。"
        f"相对较弱的策略是「{worst['策略']}」。主要顾虑集中在：{concerns} "
        "课堂上可以据此说明数字孪生的管理价值：它不是替代真实实验，而是在真实A/B测试前帮助管理者快速发现策略排序、顾客异质性和需要补充验证的假设。"
    )


def generate_strategy_summary(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    context: str,
    api_key: str,
    base_url: str,
    model: str,
    use_chinese: bool,
) -> str:
    fallback = fallback_strategy_summary(summary_df, results_df, context)
    if not api_key.strip() or summary_df.empty:
        return fallback
    reason_cols = ["策略", "被试ID", "接受度", "主要理由", "主要顾虑", "管理启示"]
    reason_sample = results_df.reindex(columns=reason_cols).dropna(how="all").tail(24)
    prompt = (
        "你是一名管理学教师。请根据数字孪生策略沙盘结果，写一段适合课堂展示的中文总结。\n"
        "要求：1）指出表现最好的策略和相对较弱的策略；2）解释可能原因；3）总结顾客异质性和主要顾虑；"
        "4）给出下一步真实管理实验建议；5）不要夸大，说明这是预实验/反事实模拟。\n\n"
        f"管理情境：{context}\n\n"
        f"策略汇总表：\n{summary_df.round({'平均接受度': 3, '高意向比例': 3}).to_csv(index=False)}\n\n"
        f"个体理由样例：\n{reason_sample.to_csv(index=False)}"
    )
    try:
        return call_llm(
            ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE,
            prompt,
            api_key.strip(),
            base_url,
            model,
            max_tokens=650,
            thinking=st.session_state.get("thinking_mode", "disabled"),
        ).strip()
    except Exception as exc:
        return f"{fallback}\n\n（自动总结调用失败，已使用规则总结：{type(exc).__name__}: {exc}）"


def build_market_research_prompt(persona_text: str, research_goal: str, product_concept: str, questions: list[str], use_chinese: bool) -> str:
    question_lines = "\n".join(f"{i + 1}. {question}" for i, question in enumerate(questions) if str(question).strip())
    if use_chinese:
        return (
            f"人物画像：\n{persona_text}\n\n"
            f"调研目标：{research_goal}\n"
            f"产品/概念：{product_concept}\n\n"
            f"访谈问题：\n{question_lines}\n\n"
            "请扮演这位画像本人，像真实市场调研受访者一样回答。"
            "需要结合画像中的消费偏好、价值观、价格敏感性、健康/环保/便利取向和生活背景。"
            "请不要把所有人都回答成高意向，证据不足时给中等或偏低分。\n"
            "字段要求：购买意向和支付意愿均为1到5分；核心卖点从 健康、省钱、便利、口味、环保、社交、尝新、品质信任、其他 中选一个；"
            "主要顾虑从 价格偏高、口味不确定、健康疑虑、信任不足、购买不便、不符合习惯、信息不足、其他 中选一个；"
            "推荐渠道从 校园便利店、商超货架、电商平台、社交媒体、朋友推荐、线下试饮、其他 中选一个。\n"
            "只输出一个JSON对象，格式为："
            "{\"购买意向\":1-5,\"支付意愿\":1-5,\"核心卖点\":\"...\",\"主要顾虑\":\"...\","
            "\"推荐渠道\":\"...\",\"文案角度\":\"...\",\"用户原话\":\"...\",\"访谈回答\":{\"问题1\":\"回答\"}}。"
        )
    return (
        f"PERSONA PROFILE:\n{persona_text}\n\n"
        f"Research goal: {research_goal}\n"
        f"Product/concept: {product_concept}\n\n"
        f"Interview questions:\n{question_lines}\n\n"
        "Answer as this persona in a realistic market research interview. Use their preferences, values, price sensitivity, health/eco/convenience orientation, and life context. "
        "Do not make everyone high intent; use moderate or low scores when evidence is weak. "
        "Output only one JSON object with intent and willingness_to_pay from 1 to 5, one core_need, one main_barrier, one preferred_channel, one message_angle, one quote, and answers."
    )


def parse_market_research_response(raw: str) -> dict:
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    payload = match.group(0) if match else text
    try:
        data = json.loads(payload)
    except Exception:
        data = {}

    def text_field(keys: list[str]) -> str:
        value = payload_value(data, keys)
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value).strip()

    answers = payload_value(data, ["访谈回答", "回答", "answers", "interview_answers", "question_answers"])
    if isinstance(answers, dict):
        answer_text = "\n".join(f"{key}: {value}" for key, value in answers.items())
    elif isinstance(answers, list):
        answer_lines = []
        for item in answers:
            if isinstance(item, dict):
                question = payload_value(item, ["问题", "question", "q"]) or ""
                answer = payload_value(item, ["回答", "answer", "a"]) or ""
                answer_lines.append(f"{question}: {answer}".strip(": "))
            else:
                answer_lines.append(str(item))
        answer_text = "\n".join(line for line in answer_lines if line.strip())
    else:
        answer_text = "" if answers is None else str(answers).strip()

    intent = coerce_scale_score(payload_value(data, ["购买意向", "意向", "购买兴趣", "intent", "purchase_intent", "purchase intent"]), default=None)
    willingness = coerce_scale_score(payload_value(data, ["支付意愿", "价格接受度", "willingness_to_pay", "willingness to pay", "wtp"]), default=None)
    return {
        "购买意向": intent,
        "支付意愿": willingness,
        "核心卖点": text_field(["核心卖点", "主要卖点", "吸引点", "core_need", "core need", "main_benefit", "main benefit"]),
        "主要顾虑": text_field(["主要顾虑", "最大顾虑", "拒绝理由", "main_barrier", "main barrier", "barrier", "concern"]),
        "推荐渠道": text_field(["推荐渠道", "购买渠道", "preferred_channel", "preferred channel", "channel"]),
        "文案角度": text_field(["文案角度", "沟通角度", "message_angle", "message angle", "positioning"]),
        "用户原话": text_field(["用户原话", "受访者原话", "quote", "verbatim"]),
        "访谈回答": answer_text,
        "模型原始输出": raw,
    }


def fallback_market_research_answer(persona_text: str, research_goal: str, product_concept: str, questions: list[str]) -> str:
    rng = app_stable_rng(f"{persona_text}\n{research_goal}\n{product_concept}\n{'|'.join(questions)}")
    combined = f"{persona_text}\n{product_concept}\n{research_goal}"
    intent = min(max(3 + rng.choice([-1, 0, 0, 1]), 1), 5)
    willingness = min(max(intent + rng.choice([-1, 0, 0, 1]), 1), 5)
    if re.search(r"价格|省钱|便宜|折扣|budget|cheap|discount|price", combined, flags=re.I):
        intent = max(1, intent - 1)
        willingness = max(1, willingness - 1)
        barrier = "价格偏高"
        core_need = "省钱"
    elif re.search(r"健康|低糖|营养|运动|health|diet|fitness|sugar", combined, flags=re.I):
        core_need = "健康"
        barrier = rng.choice(["健康疑虑", "口味不确定", "信息不足"])
    elif re.search(r"环保|可持续|回收|eco|green|sustain", combined, flags=re.I):
        core_need = "环保"
        barrier = rng.choice(["信任不足", "价格偏高", "信息不足"])
    else:
        core_need = rng.choice(["便利", "口味", "尝新", "品质信任"])
        barrier = rng.choice(["口味不确定", "不符合习惯", "信息不足", "购买不便"])
    channel = rng.choice(["校园便利店", "商超货架", "电商平台", "社交媒体", "朋友推荐", "线下试饮"])
    angle = {
        "健康": "突出低负担和日常饮用场景",
        "省钱": "强调试用优惠和性价比",
        "环保": "用可验证的环保承诺建立信任",
        "便利": "强调随手可得和省时间",
        "口味": "先让消费者相信口感不会牺牲",
        "尝新": "突出新鲜感和限时体验",
        "品质信任": "展示配方、品牌背书和真实评价",
    }.get(core_need, "用清晰证据降低尝试门槛")
    quote = f"如果{barrier}能解决，我会愿意先试一次，尤其是它真的能做到{core_need}。"
    answers = {f"问题{i + 1}": f"我会关注{core_need}，但也会先看{barrier}。" for i, _ in enumerate(questions)}
    return json.dumps(
        {
            "购买意向": intent,
            "支付意愿": willingness,
            "核心卖点": core_need,
            "主要顾虑": barrier,
            "推荐渠道": channel,
            "文案角度": angle,
            "用户原话": quote,
            "访谈回答": answers,
        },
        ensure_ascii=False,
    )


def simulate_market_research_answer(
    persona_text: str,
    research_goal: str,
    product_concept: str,
    questions: list[str],
    use_chinese: bool,
    kwargs: dict,
) -> dict:
    source = "Mock"
    raw = fallback_market_research_answer(persona_text, research_goal, product_concept, questions)
    if kwargs.get("api_key", "").strip():
        prompt = build_market_research_prompt(persona_text, research_goal, product_concept, questions, use_chinese)
        try:
            raw = call_llm(
                ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE,
                prompt,
                kwargs["api_key"].strip(),
                kwargs["base_url"],
                kwargs.get("model", "deepseek-v4-pro"),
                max_tokens=max(360, kwargs.get("max_tokens", 360)),
                thinking=kwargs.get("thinking", "disabled"),
            )
            source = "真实API"
        except Exception as exc:
            raw = f"调用失败：{safe_error_message(exc)}"
            source = "备用规则"
    parsed = parse_market_research_response(raw)
    fallback = parse_market_research_response(fallback_market_research_answer(persona_text, research_goal, product_concept, questions))
    for key, value in fallback.items():
        if key == "模型原始输出":
            continue
        if parsed.get(key) in (None, ""):
            parsed[key] = value
            if source == "真实API":
                source = "真实API+备用规则"
    parsed["来源"] = source
    parsed["模型原始输出"] = raw
    return parsed


def top_label_counts(series: pd.Series, label: str) -> pd.DataFrame:
    values = series.fillna("").astype(str).str.strip()
    values = values[values.ne("")]
    if values.empty:
        return pd.DataFrame(columns=[label, "人数", "占比"])
    counts = values.value_counts().rename_axis(label).reset_index(name="人数")
    counts["占比"] = counts["人数"] / len(values)
    return counts


def fallback_market_research_summary(results_df: pd.DataFrame, product_concept: str, research_goal: str, high_threshold: int) -> str:
    if results_df is None or results_df.empty:
        return "暂无可总结的市场调研结果。"
    data = results_df.copy()
    data["购买意向"] = pd.to_numeric(data.get("购买意向"), errors="coerce")
    data["支付意愿"] = pd.to_numeric(data.get("支付意愿"), errors="coerce")
    intent_mean = data["购买意向"].mean()
    high_share = data["购买意向"].ge(high_threshold).mean()
    top_need = top_label_counts(data.get("核心卖点", pd.Series(dtype=str)), "核心卖点").head(1)
    top_barrier = top_label_counts(data.get("主要顾虑", pd.Series(dtype=str)), "主要顾虑").head(1)
    top_channel = top_label_counts(data.get("推荐渠道", pd.Series(dtype=str)), "推荐渠道").head(1)
    need = top_need.iloc[0]["核心卖点"] if not top_need.empty else "暂无"
    barrier = top_barrier.iloc[0]["主要顾虑"] if not top_barrier.empty else "暂无"
    channel = top_channel.iloc[0]["推荐渠道"] if not top_channel.empty else "暂无"
    return (
        f"围绕「{product_concept}」的调研目标「{research_goal}」，当前样本平均购买意向为 {intent_mean:.2f}/5，"
        f"高意向比例约为 {high_share:.1%}。最常见核心卖点是「{need}」，主要顾虑集中在「{barrier}」，"
        f"推荐优先触达渠道为「{channel}」。建议下一步把高意向人群和主要顾虑人群拆开，分别测试价格、试饮和信任背书。"
    )


def generate_market_research_summary(
    results_df: pd.DataFrame,
    product_concept: str,
    research_goal: str,
    api_key: str,
    base_url: str,
    model: str,
    use_chinese: bool,
    high_threshold: int,
) -> str:
    fallback = fallback_market_research_summary(results_df, product_concept, research_goal, high_threshold)
    if not api_key.strip() or results_df is None or results_df.empty:
        return fallback
    sample_cols = ["被试ID", "购买意向", "支付意愿", "核心卖点", "主要顾虑", "推荐渠道", "文案角度", "用户原话"]
    sample = results_df.reindex(columns=sample_cols).dropna(how="all").tail(30)
    prompt = (
        "你是一名市场调研分析师。请根据数字孪生访谈结果写一段适合课堂展示的中文调研结论。\n"
        "要求：1）说明总体购买意向和支付意愿；2）提炼主要卖点、顾虑和渠道；3）引用1-2条用户原话；"
        "4）给出下一步真实调研或营销实验建议；5）明确这是预实验模拟，不要夸大。\n\n"
        f"调研目标：{research_goal}\n"
        f"产品/概念：{product_concept}\n"
        f"高意向阈值：{high_threshold}\n\n"
        f"结果样例：\n{sample.to_csv(index=False)}"
    )
    try:
        return call_llm(
            ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE,
            prompt,
            api_key.strip(),
            base_url,
            model,
            max_tokens=700,
            thinking=st.session_state.get("thinking_mode", "disabled"),
        ).strip()
    except Exception as exc:
        return f"{fallback}\n\n（自动总结调用失败，已使用规则总结：{type(exc).__name__}: {exc}）"


def build_segment_prompt(persona_text: str, context: str, use_chinese: bool) -> str:
    if use_chinese:
        return (
            f"人物画像：\n{persona_text}\n\n"
            f"管理情境：{context}\n\n"
            "请根据画像判断这个人作为消费者/管理对象的细分特征。"
            "请为以下维度分别给1到5分：价格敏感、健康导向、环保导向、便利导向、社会证明、尝新倾向。"
            "其中“社会证明”指同伴评价、口碑推荐、从众影响、他人使用证据对这个人的影响程度。"
            "1表示很弱，3表示中等或信息不足，5表示很强。请校准评分，不要全部给高分。"
            "然后选择一个最合适的分群标签：价格敏感型、健康价值型、环保责任型、便利效率型、社交影响型、探索尝新型、低涉入观望型。"
            "只输出一个JSON对象，格式为："
            "{\"价格敏感\":1-5,\"健康导向\":1-5,\"环保导向\":1-5,\"便利导向\":1-5,"
            "\"社会证明\":1-5,\"尝新倾向\":1-5,\"分群\":\"...\",\"理由\":\"...\"}。"
        )
    return (
        f"PERSONA PROFILE:\n{persona_text}\n\n"
        f"Management context: {context}\n\n"
        "Score this person's consumer/managerial segmentation profile from 1 to 5 on: price sensitivity, health orientation, eco orientation, convenience orientation, social proof, novelty seeking. Social proof means peer reviews, word of mouth, popularity cues, and other people using the product. "
        "Use 1=weak, 3=moderate or insufficient evidence, 5=strong. Calibrate scores and avoid making all scores high. "
        "Choose one segment label. Output only one JSON object."
    )


def parse_segment_response(raw: str) -> dict:
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    payload = match.group(0) if match else text
    try:
        data = json.loads(payload)
    except Exception:
        data = {}
    result = {}
    aliases = {
        "价格敏感": ["价格敏感", "价格敏感度", "价格导向", "价格意识", "价格关注", "成本敏感", "成本意识", "price_sensitivity", "price sensitivity", "price"],
        "健康导向": ["健康导向", "健康意识", "健康价值", "健康关注", "health_orientation", "health orientation", "health"],
        "环保导向": ["环保导向", "环保意识", "环保价值", "可持续导向", "可持续意识", "eco_orientation", "eco orientation", "environmental_orientation", "environmental orientation", "sustainability"],
        "便利导向": ["便利导向", "便利性", "效率导向", "方便程度", "convenience_orientation", "convenience orientation", "convenience"],
        "社会证明": ["社会证明", "社会认同", "社会影响", "社交影响", "同伴影响", "从众倾向", "口碑影响", "口碑导向", "他人评价", "朋友推荐", "social_proof", "social proof", "social_influence", "social influence", "peer_influence", "peer influence", "word_of_mouth", "word of mouth", "popularity cues"],
        "尝新倾向": ["尝新倾向", "尝新", "探索倾向", "新奇偏好", "创新接受", "novelty_seeking", "novelty seeking", "novelty", "innovativeness"],
    }
    for dim in SEGMENT_DIMENSIONS:
        result[dim] = coerce_scale_score(payload_value(data, aliases[dim]), default=3)
    result["分群"] = str(payload_value(data, ["分群", "细分", "segment", "segment_label", "segment label"]) or "未分类").strip()
    result["理由"] = str(payload_value(data, ["理由", "原因", "reason", "main_reason", "main reason"]) or "").strip()
    result["模型原始输出"] = raw
    return result


def simulate_segment_answer(persona_text: str, context: str, use_chinese: bool, kwargs: dict) -> dict:
    prompt = build_segment_prompt(persona_text, context, use_chinese)
    if kwargs.get("api_key", "").strip():
        raw = call_llm(
            ZH_SYSTEM_MESSAGE if use_chinese else SYSTEM_MESSAGE,
            prompt,
            kwargs["api_key"].strip(),
            kwargs["base_url"],
            kwargs.get("model", "deepseek-v4-pro"),
            max_tokens=max(220, kwargs.get("max_tokens", 220)),
            thinking=kwargs.get("thinking", "disabled"),
        )
    else:
        rng = random.Random(hash(persona_text + context))
        values = {dim: min(max(3 + rng.choice([-1, 0, 0, 1]), 1), 5) for dim in SEGMENT_DIMENSIONS}
        label = max(values, key=values.get)
        raw = json.dumps({**values, "分群": f"{label}型", "理由": "画像信息显示该维度相对突出。"}, ensure_ascii=False)
    return parse_segment_response(raw)


def radar_chart(avg_df: pd.DataFrame) -> alt.Chart:
    import math

    dims = avg_df["维度"].tolist()
    n = max(1, len(dims))
    points = []
    for i, row in avg_df.reset_index(drop=True).iterrows():
        angle = 2 * math.pi * i / n
        radius = float(row["平均分"])
        points.append({"维度": row["维度"], "平均分": radius, "x": radius * math.cos(angle), "y": radius * math.sin(angle), "order": i})
    if points:
        first = dict(points[0])
        first["order"] = len(points)
        points.append(first)
    plot_df = pd.DataFrame(points)
    rings = []
    for radius in range(1, 6):
        for i, dim in enumerate(dims + dims[:1]):
            angle = 2 * math.pi * (i % n) / n
            rings.append({"ring": radius, "维度": dim, "x": radius * math.cos(angle), "y": radius * math.sin(angle), "order": i})
    axes = []
    labels = []
    for i, dim in enumerate(dims):
        angle = 2 * math.pi * i / n
        axes.extend(
            [
                {"维度": dim, "x": 0.0, "y": 0.0, "order": 0},
                {"维度": dim, "x": 5 * math.cos(angle), "y": 5 * math.sin(angle), "order": 1},
            ]
        )
        labels.append({"维度": dim, "x": 5.55 * math.cos(angle), "y": 5.55 * math.sin(angle)})

    ring_chart = (
        alt.Chart(pd.DataFrame(rings))
        .mark_line(color="#d7dce2", strokeWidth=1)
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), order="order:Q", detail="ring:N")
    )
    axis_chart = (
        alt.Chart(pd.DataFrame(axes))
        .mark_line(color="#e5e7eb", strokeWidth=1)
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), order="order:Q", detail="维度:N")
    )
    line_chart = (
        alt.Chart(plot_df)
        .mark_line(point=True, color="#2563eb", strokeWidth=3)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            order="order:Q",
            tooltip=[alt.Tooltip("维度:N"), alt.Tooltip("平均分:Q", format=".2f")],
        )
    )
    label_chart = (
        alt.Chart(pd.DataFrame(labels))
        .mark_text(fontSize=12, color="#374151")
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), text="维度:N")
    )
    return alt.layer(ring_chart, axis_chart, line_chart, label_chart).properties(height=360, title="整体细分雷达")


def segment_tables(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if results_df is None or results_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    data = results_df.copy()
    if "分群" not in data.columns:
        data["分群"] = "未分类"
    data["分群"] = data["分群"].fillna("未分类").astype(str).str.strip().replace("", "未分类")
    for dim in SEGMENT_DIMENSIONS:
        data[dim] = pd.to_numeric(data.get(dim), errors="coerce")
    total = max(1, len(data))
    summary = data.groupby("分群", as_index=False).agg(人数=("被试ID", "count"))
    summary["占比"] = summary["人数"] / total
    grouped = data.groupby("分群", as_index=False)[SEGMENT_DIMENSIONS].mean()
    dominant = grouped.set_index("分群")[SEGMENT_DIMENSIONS].fillna(0).idxmax(axis=1).to_dict()
    summary["主导维度"] = summary["分群"].map(dominant).fillna("暂无")
    summary = summary.sort_values(["人数", "分群"], ascending=[False, True])
    dim_avg = data[SEGMENT_DIMENSIONS].mean().reset_index()
    dim_avg.columns = ["维度", "平均分"]
    segment_dim = grouped.melt(id_vars="分群", value_vars=SEGMENT_DIMENSIONS, var_name="维度", value_name="平均分")
    return data, summary, dim_avg, segment_dim


def segment_overview_chart(summary: pd.DataFrame, dim_avg: pd.DataFrame) -> alt.Chart:
    share = (
        alt.Chart(summary)
        .mark_arc(innerRadius=68, outerRadius=120)
        .encode(
            theta=alt.Theta("人数:Q", stack=True),
            color=alt.Color("分群:N", title="分群"),
            tooltip=[alt.Tooltip("分群:N"), alt.Tooltip("人数:Q"), alt.Tooltip("占比:Q", format=".1%"), alt.Tooltip("主导维度:N")],
        )
        .properties(height=300, title="分群占比")
    )
    dimensions = (
        alt.Chart(dim_avg)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("维度:N", title=None, sort=SEGMENT_DIMENSIONS),
            y=alt.Y("平均分:Q", title="平均分", scale=alt.Scale(domain=[1, 5])),
            color=alt.Color("维度:N", legend=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=[alt.Tooltip("维度:N"), alt.Tooltip("平均分:Q", format=".2f")],
        )
        .properties(height=300, title="整体维度均值")
    )
    return alt.hconcat(share, dimensions).resolve_scale(color="independent")


def segment_heatmap_chart(segment_dim: pd.DataFrame) -> alt.Chart:
    heat = (
        alt.Chart(segment_dim)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("维度:N", title=None, sort=SEGMENT_DIMENSIONS),
            y=alt.Y("分群:N", title=None),
            color=alt.Color("平均分:Q", title="平均分", scale=alt.Scale(domain=[1, 5], scheme="tealblues")),
            tooltip=[alt.Tooltip("分群:N"), alt.Tooltip("维度:N"), alt.Tooltip("平均分:Q", format=".2f")],
        )
        .properties(height=320, title="分群 × 维度热力图")
    )
    labels = (
        alt.Chart(segment_dim)
        .mark_text(fontSize=12)
        .encode(
            x=alt.X("维度:N", sort=SEGMENT_DIMENSIONS),
            y="分群:N",
            text=alt.Text("平均分:Q", format=".1f"),
            color=alt.condition(alt.datum["平均分"] >= 3.5, alt.value("white"), alt.value("#111827")),
        )
    )
    return heat + labels


def segment_bubble_chart(segment_dim: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(segment_dim)
        .mark_circle(opacity=0.78)
        .encode(
            x=alt.X("维度:N", title=None, sort=SEGMENT_DIMENSIONS),
            y=alt.Y("分群:N", title=None),
            size=alt.Size("平均分:Q", title="平均分", scale=alt.Scale(range=[70, 900])),
            color=alt.Color("平均分:Q", title="平均分", scale=alt.Scale(domain=[1, 5], scheme="goldgreen")),
            tooltip=[alt.Tooltip("分群:N"), alt.Tooltip("维度:N"), alt.Tooltip("平均分:Q", format=".2f")],
        )
        .properties(height=320, title="分群画像气泡图")
    )


def segment_scatter_chart(data: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_circle(size=72, opacity=0.72)
        .encode(
            x=alt.X("价格敏感:Q", title="价格敏感", scale=alt.Scale(domain=[1, 5])),
            y=alt.Y("健康导向:Q", title="健康导向", scale=alt.Scale(domain=[1, 5])),
            color=alt.Color("分群:N", title="分群"),
            size=alt.Size("环保导向:Q", title="环保导向", scale=alt.Scale(range=[45, 360])),
            tooltip=[alt.Tooltip("被试ID:N"), alt.Tooltip("分群:N")] + [alt.Tooltip(f"{dim}:Q", format=".0f") for dim in SEGMENT_DIMENSIONS] + [alt.Tooltip("理由:N")],
        )
        .properties(height=360, title="个体散点：价格敏感 × 健康导向")
    )


def segment_box_chart(data: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data)
        .transform_fold(SEGMENT_DIMENSIONS, as_=["维度", "得分"])
        .mark_boxplot(extent="min-max", size=38)
        .encode(
            x=alt.X("维度:N", title=None, sort=SEGMENT_DIMENSIONS),
            y=alt.Y("得分:Q", title="得分分布", scale=alt.Scale(domain=[1, 5])),
            color=alt.Color("维度:N", legend=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=[alt.Tooltip("维度:N"), alt.Tooltip("得分:Q", format=".0f")],
        )
        .properties(height=320, title="六个维度的个体分布")
    )


def answer_position(question: dict | None):
    if not question:
        return None
    pos = question.get("Answers", {}).get("SelectedByPosition")
    if isinstance(pos, list):
        return pos[0] if len(pos) == 1 else None
    return pos if isinstance(pos, int) else None


def single_item_evaluation_for_language(
    rows,
    qid: str,
    question_text: str,
    options: list[str],
    selected_variant: str,
    *,
    batch_size: int = 8,
    workers: int = 4,
    api_sleep: float = 0.0,
    progress_callback=None,
    **kwargs,
) -> pd.DataFrame:
    records = []
    lo, hi = 1, len(options)
    prepared = []
    use_chinese = st.session_state["use_chinese_text"]
    for order, row in enumerate(rows):
        gt_q = find_question_by_id(row["ground_truth"], qid)
        rt_q = find_question_by_id(row["retest"], qid)
        prepared.append(
            {
                "order": order,
                "pid": row["pid"],
                "truth": answer_position(gt_q),
                "retest": answer_position(rt_q),
                "persona_text": zh_text(get_persona_variant(row, selected_variant), context=f"persona:{row['pid']}:{selected_variant}"),
            }
        )
    total = max(1, len(prepared))
    completed = 0
    for batch_start in range(0, len(prepared), max(1, batch_size)):
        batch = prepared[batch_start : batch_start + max(1, batch_size)]
        future_to_item = {}
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(batch)))) as executor:
            for item in batch:
                future_to_item[
                    executor.submit(
                        simulate_single_choice_answer,
                        item["persona_text"],
                        question_text,
                        options,
                        use_chinese,
                        kwargs,
                    )
                ] = item
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    twin, raw = future.result()
                except Exception as exc:
                    twin, raw = None, f"调用失败：{safe_error_message(exc)}"
                records.append(
                    {
                        "order": item["order"],
                        "pid": item["pid"],
                        "truth": item["truth"],
                        "retest": item["retest"],
                        "twin": twin,
                        "twin_label": options[twin - 1] if twin else None,
                        "raw_twin": raw,
                        "acc_twin": accuracy(twin, item["truth"], lo, hi),
                        "acc_retest": accuracy(item["retest"], item["truth"], lo, hi),
                    }
                )
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, records)
        if api_sleep and kwargs.get("api_key", "").strip():
            time.sleep(api_sleep)
    return pd.DataFrame(records).sort_values("order").drop(columns=["order"], errors="ignore").reset_index(drop=True)


def single_open_evaluation_for_language(
    rows,
    qid: str,
    question_text: str,
    selected_variant: str,
    *,
    batch_size: int = 8,
    workers: int = 4,
    api_sleep: float = 0.0,
    progress_callback=None,
    **kwargs,
) -> pd.DataFrame:
    records = []
    prepared = []
    use_chinese = st.session_state["use_chinese_text"]
    for order, row in enumerate(rows):
        gt_q = find_question_by_id(row["ground_truth"], qid) if qid != "CUSTOM_OPEN" else None
        rt_q = find_question_by_id(row["retest"], qid) if qid != "CUSTOM_OPEN" else None
        truth = answer_text(gt_q)
        retest = answer_text(rt_q)
        prepared.append(
            {
                "order": order,
                "pid": row["pid"],
                "truth_text": zh_text(truth, context=f"open-truth:{qid}:{row['pid']}") if truth else None,
                "retest_text": zh_text(retest, context=f"open-retest:{qid}:{row['pid']}") if retest else None,
                "persona_text": zh_text(get_persona_variant(row, selected_variant), context=f"open-persona:{row['pid']}:{selected_variant}"),
            }
        )
    total = max(1, len(prepared))
    completed = 0
    for batch_start in range(0, len(prepared), max(1, batch_size)):
        batch = prepared[batch_start : batch_start + max(1, batch_size)]
        future_to_item = {}
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(batch)))) as executor:
            for item in batch:
                future_to_item[
                    executor.submit(
                        simulate_single_open_answer,
                        item["persona_text"],
                        question_text,
                        use_chinese,
                        kwargs,
                    )
                ] = item
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    raw = future.result()
                except Exception as exc:
                    raw = f"调用失败：{safe_error_message(exc)}"
                records.append(
                    {
                        "order": item["order"],
                        "pid": item["pid"],
                        "qid": qid,
                        "truth_text": item["truth_text"],
                        "retest_text": item["retest_text"],
                        "twin_answer": visible_model_output(raw),
                    }
                )
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, records)
        if api_sleep and kwargs.get("api_key", "").strip():
            time.sleep(api_sleep)
    return pd.DataFrame(records).sort_values("order").drop(columns=["order"], errors="ignore").reset_index(drop=True)


def evaluate_items_for_language(rows, items, selected_variant: str, *, random_seed: int, api_sleep: float, **kwargs) -> pd.DataFrame:
    rng = random.Random(random_seed)
    records = []
    for item in items:
        options = item["options"]
        for row in rows:
            gt_q = find_question_by_id(row["ground_truth"], item["qid"])
            rt_q = find_question_by_id(row["retest"], item["qid"])
            truth = answer_position(gt_q)
            retest = answer_position(rt_q)
            persona_text = zh_text(get_persona_variant(row, selected_variant), context=f"batch-persona:{row['pid']}:{selected_variant}")
            twin, raw = simulate_answer_for_language(persona_text, item["text"], options, **kwargs)
            records.append(
                {
                    "variant": selected_variant,
                    "pid": row["pid"],
                    "qid": item["qid"],
                    "lo": item["lo"],
                    "hi": item["hi"],
                    "truth": truth,
                    "twin": twin,
                    "retest": retest,
                    "random": rng.randint(1, len(options)),
                    "raw_twin": raw,
                }
            )
            if api_sleep and kwargs.get("api_key", "").strip():
                time.sleep(api_sleep)
    return pd.DataFrame(records)


with st.sidebar:
    st.header("运行配置")
    num_personas = st.slider("载入画像数", min_value=8, max_value=100, value=100, step=2)
    use_hub = st.toggle("本地数据缺失时尝试 HuggingFace", value=False)
    text_language = st.radio("文本语言", ["中文", "原文"], horizontal=True, index=0)
    api_key = st.text_input("DeepSeek API Key", type="password", value=default_api_key())
    base_url = st.text_input("Base URL", value="https://api.deepseek.com")
    model = st.text_input("模型", value="deepseek-v4-flash")
    max_tokens = st.slider("回答 token 上限", min_value=3, max_value=600, value=30, step=1)
    thinking_enabled = st.toggle("DeepSeek thinking 模式", value=False)
    auto_translate_uncached = st.toggle("用 API 补翻未缓存文本", value=False)
    variant = st.selectbox("画像版本", PROFILE_VARIANTS, index=0, format_func=zh_variant_label)
    random_seed = st.number_input("随机种子", min_value=0, max_value=9999, value=0, step=1)
    api_sleep = st.slider("真实 API 间隔秒数", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

st.session_state["use_chinese_text"] = text_language == "中文"
st.session_state["api_key"] = api_key
st.session_state["base_url"] = base_url
st.session_state["model"] = model
st.session_state["thinking_mode"] = "enabled" if thinking_enabled else "disabled"
st.session_state["auto_translate_uncached"] = auto_translate_uncached

df, source = cached_data(num_personas, use_hub)
rows = df.to_dict("records")
mode = "真实 API" if api_key.strip() else "Mock"
if st.session_state["use_chinese_text"]:
    st.caption("中文模式已开启；已缓存文本会直接显示中文，未缓存文本默认使用本地兜底，不会阻塞页面加载。")
if st.session_state.get("translation_error"):
    st.warning(f"部分文本未能自动翻译，已使用本地兜底：{st.session_state['translation_error']}")

st.title("Twin-2K-500 交互实验台")
top = st.columns([1, 1, 1, 1, 1])
top[0].metric("数据源", source)
top[1].metric("画像数", f"{len(df)}")
top[2].metric("运行模式", mode)
top[3].metric("默认画像", zh_variant_label(variant) if st.session_state["use_chinese_text"] else variant)
top[4].metric("Thinking", "开" if st.session_state["thinking_mode"] == "enabled" else "关")

tabs = st.tabs(["画像", "画像构建", "单题模拟", "批量评估", "需求曲线", "管理沙盘", "细分雷达", "市场调研", "提示词"])

with tabs[0]:
    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        pid = st.selectbox("被试 ID", df["pid"].astype(str).tolist())
        selected = df[df["pid"].astype(str) == pid].iloc[0]
        max_blocks = st.slider("区块数", 1, 8, 3)
        max_q = st.slider("每区块题数", 1, 8, 3)
    with right:
        st.subheader(f"画像 {pid}")
        profile_rows = inspect_persona_rows(persona_blocks(selected), max_blocks=max_blocks, max_q=max_q)
        if st.session_state["use_chinese_text"]:
            profile_rows = profile_rows.copy()
            for col in ["block", "question", "answer"]:
                profile_rows[col] = profile_rows[col].map(lambda value: zh_text(str(value), context=f"profile-table:{col}"))
            profile_rows["type"] = profile_rows["type"].map(zh_question_types)
            profile_rows = profile_rows.rename(columns={"block": "区块", "question": "题目", "type": "题型", "answer": "作答"})
        st.dataframe(profile_rows, width="stretch", hide_index=True)
        with st.expander("LLM 画像文本", expanded=False):
            persona_text_display = zh_text(get_persona_variant(selected, variant), context=f"profile:{pid}:{variant}")
            st.text_area("persona_text", persona_text_display, height=360, label_visibility="collapsed")

with tabs[1]:
    st.subheader("如何建立一个可用画像")
    st.markdown(
        """
        数字孪生画像不是模型随口编出来的人设，而是把同一个人在历史问卷中的作答整理成可检索、可提示、可评估的个人上下文。
        这个 notebook 使用的是 `wave1_3_persona_json`：它包含被试在 W1-W3 的非留出题作答；预测时再拿 W4 留出题作为新问题。
        """
    )

    steps = pd.DataFrame(
        [
            {
                "步骤": "1. 收集历史作答",
                "输入": "人口统计、价值观、偏好、人格、行为选择等题目",
                "输出": "结构化 JSON 区块",
            },
            {
                "步骤": "2. 清理与标准化",
                "输入": "HTML、空白、选项编号、单选/多选答案",
                "输出": "题干、选项、已选答案",
            },
            {
                "步骤": "3. 选择画像粒度",
                "输入": "完整问卷、摘要画像、人口统计、空画像",
                "输出": "不同评测条件",
            },
            {
                "步骤": "4. 转成提示词上下文",
                "输入": "画像文本 + 新题 + 选项 + 输出格式",
                "输出": "大模型可回答的人物画像提示词",
            },
            {
                "步骤": "5. 用留出题评估",
                "输入": "孪生预测、历史真值、人类重测",
                "输出": "准确率、相关性、均值偏差、方差比例",
            },
        ]
    )
    st.dataframe(steps, width="stretch", hide_index=True)

    build_left, build_right = st.columns([0.34, 0.66], gap="large")
    with build_left:
        build_pid = st.selectbox("示例被试", df["pid"].astype(str).tolist(), key="build_pid")
        build_row = df[df["pid"].astype(str) == build_pid].iloc[0]
        build_blocks = persona_blocks(build_row)
        build_variant = st.selectbox(
            "画像构建方式",
            ["full_or_summary", "full", "demographics_only", "empty"],
            format_func=zh_variant_label,
        )
        st.metric("原始区块数", len(build_blocks))
        st.metric("原始问题数", sum(len(block.get("Questions", [])) for block in build_blocks))
    with build_right:
        st.markdown("**原始画像 JSON 的区块结构**")
        block_summary = persona_block_summary(build_blocks)
        if st.session_state["use_chinese_text"]:
            block_summary = block_summary.copy()
            block_summary["block"] = block_summary["block"].map(lambda value: zh_text(str(value), context="block-name"))
            block_summary["sample_question"] = block_summary["sample_question"].map(lambda value: zh_text(str(value), context="block-sample"))
            block_summary["types"] = block_summary["types"].map(zh_question_types)
            block_summary = block_summary.rename(columns={"block": "区块", "questions": "题目数", "types": "题型", "sample_question": "示例题干"})
        st.dataframe(block_summary, width="stretch", hide_index=True)

    st.markdown("**画像版本对比**")
    size_table = profile_size_table(build_row)
    if st.session_state["use_chinese_text"]:
        size_table = size_table.copy()
        size_table["preview"] = size_table["版本" if "版本" in size_table.columns else "variant"].map(
            lambda name: zh_text(get_persona_variant(build_row, str(name)), context=f"profile-preview:{build_pid}:{name}").replace("\n", " ")[:180]
        )
        size_table["variant"] = size_table["variant"].map(zh_variant_label)
        size_table = size_table.rename(columns={"variant": "版本", "label": "说明", "characters": "字符数", "lines": "行数", "preview": "预览"})
    st.dataframe(size_table, width="stretch", hide_index=True)

    profile_text = zh_text(get_persona_variant(build_row, build_variant), context=f"build-profile:{build_pid}:{build_variant}")
    preview_col, prompt_col = st.columns(2, gap="large")
    with preview_col:
        st.markdown("**生成后的画像文本**")
        st.text_area("profile_text", profile_text, height=430, label_visibility="collapsed")
    with prompt_col:
        qid_for_prompt, question_for_prompt, options_for_prompt = choose_eval_question([build_row.to_dict()])
        question_for_prompt = zh_text(question_for_prompt, context=f"build-question:{qid_for_prompt}")
        options_for_prompt = zh_options(options_for_prompt, context=f"build-options:{qid_for_prompt}")
        prompt = build_user_prompt_for_language(profile_text, question_for_prompt, options_for_prompt)
        st.markdown(f"**接入新问题后的提示词** `{qid_for_prompt}`")
        st.text_area("persona_prompt", prompt, height=430, label_visibility="collapsed")

    st.info(
        "画像构建的关键边界：只能使用预测发生前已经知道的信息；留出题答案不能放进画像，否则评估会变成泄漏答案。"
    )

with tabs[2]:
    pid_options = df["pid"].astype(str).tolist()
    pid_to_row = {str(row["pid"]): row for row in rows}
    default_pids = pid_options[: min(8, len(pid_options))]

    persona_col, count_col = st.columns([0.42, 0.58], gap="large")
    with persona_col:
        persona_mode = st.radio("画像选择", ["前N个画像", "手动选择画像"], horizontal=True)
    with count_col:
        if persona_mode == "前N个画像":
            show_n = st.slider("模拟人数", 1, min(30, len(rows)), min(8, len(rows)))
            selected_pids = pid_options[:show_n]
        else:
            selected_pids = st.multiselect("选择被试画像", pid_options, default=default_pids)
            if not selected_pids:
                st.warning("至少选择一个画像；已临时使用第一个画像。")
                selected_pids = pid_options[:1]
            if len(selected_pids) > 30:
                st.warning("单题模拟最多同时展示30个画像，已使用前30个。")
                selected_pids = selected_pids[:30]

    shown_rows = [pid_to_row[pid] for pid in selected_pids if pid in pid_to_row]
    st.caption(f"当前将模拟 {len(shown_rows)} 个画像：{', '.join(selected_pids[:8])}{' ...' if len(selected_pids) > 8 else ''}")

    default_qid, default_question, default_options = choose_eval_question(shown_rows)
    choice_items = candidate_eval_items(shown_rows, max_items=80, include_product_items=True)
    choice_items = choice_items or [{"qid": default_qid, "text": default_question, "options": default_options}]
    open_items = candidate_open_answer_items(shown_rows, max_items=80)
    if "single_choice_idx" not in st.session_state:
        st.session_state["single_choice_idx"] = 0
    if "single_open_idx" not in st.session_state:
        st.session_state["single_open_idx"] = 0
    if st.session_state["single_choice_idx"] >= len(choice_items):
        st.session_state["single_choice_idx"] = 0
    if open_items and st.session_state["single_open_idx"] >= len(open_items):
        st.session_state["single_open_idx"] = 0

    mode_col, question_col = st.columns([0.28, 0.72], gap="large")
    with mode_col:
        question_kind = st.radio("题型", ["选择题", "问答题"], horizontal=False)
        question_mode = st.radio("题目来源", ["数据集留出题", "自定义题目"], horizontal=False)
        if question_mode == "数据集留出题":
            if st.button("随机切换题干", width="stretch"):
                active_items = open_items if question_kind == "问答题" else choice_items
                active_key = "single_open_idx" if question_kind == "问答题" else "single_choice_idx"
                if len(active_items) > 1:
                    choices = [i for i in range(len(active_items)) if i != st.session_state[active_key]]
                    st.session_state[active_key] = random.choice(choices)
                else:
                    st.session_state[active_key] = 0
            if question_kind == "问答题":
                if open_items:
                    selected_item = open_items[st.session_state["single_open_idx"]]
                    qid = selected_item["qid"]
                    options = []
                    st.caption(f"{qid} · {zh_question_types(selected_item.get('question_type', 'TE'))}")
                else:
                    qid = "CUSTOM_OPEN"
                    options = []
                    st.caption("当前画像组合没有共同开放题，已切换为自定义开放题。")
            else:
                selected_item = choice_items[st.session_state["single_choice_idx"]]
                qid = selected_item["qid"]
                options = zh_options(selected_item["options"], context=f"single-options:{qid}")
                st.caption(qid)
        else:
            if question_kind == "问答题":
                qid = "CUSTOM_OPEN"
                options = []
            else:
                qid = "CUSTOM"
                options = option_editor(["否", "是"] if st.session_state["use_chinese_text"] else ["No", "Yes"])

    with question_col:
        if question_mode == "数据集留出题" and not (question_kind == "问答题" and not open_items):
            selected_item = open_items[st.session_state["single_open_idx"]] if question_kind == "问答题" else choice_items[st.session_state["single_choice_idx"]]
            display_question = zh_text(selected_item["text"], context=f"single-question:{qid}")
            question_text = st.text_area("题干", display_question, height=140, key=f"single_question_{question_kind}_{qid}_{text_language}")
        else:
            if question_kind == "问答题":
                default_custom_question = (
                    "请用一两句话说明：你在购买一种新饮料时，最看重哪些因素？"
                    if st.session_state["use_chinese_text"]
                    else "In one or two sentences, what factors matter most to you when buying a new beverage?"
                )
            else:
                default_custom_question = (
                    "价格为 $8.26 时，你会购买以下商品吗？商品：可口可乐经典汽水，12液量盎司，12罐装。"
                    if st.session_state["use_chinese_text"]
                    else "Would you buy the following product at $8.26? Product: Coca-Cola Classic Soda, 12 fl oz, 12-pack."
                )
            question_text = st.text_area("题干", default_custom_question, height=140, key=f"custom_question_text_{question_kind}_{text_language}")

    run_col1, run_col2 = st.columns(2)
    single_batch_size = run_col1.slider("单题批大小", min_value=1, max_value=max(1, min(30, len(shown_rows))), value=max(1, min(8, len(shown_rows))), step=1)
    single_workers = run_col2.slider("单题并发请求数", min_value=1, max_value=max(1, min(20, len(shown_rows))), value=max(1, min(8, len(shown_rows))), step=1)
    if question_kind == "选择题":
        token_note = (
            f"thinking 开启时使用侧栏 token 上限，不启用严格数字题上限；只要求返回数字。"
            if st.session_state["thinking_mode"] == "enabled"
            else f"输出 token 严格上限={CHOICE_MAX_TOKENS}；只要求返回数字。"
        )
        st.caption(
            f"选择题调用：thinking={'开' if st.session_state['thinking_mode'] == 'enabled' else '关'}；"
            f"{token_note}"
        )

    if st.button("运行单题模拟", type="primary", width="stretch"):
        if not shown_rows:
            st.error("至少需要选择一个画像。")
        elif not str(question_text or "").strip():
            st.error("题干不能为空。")
        elif question_kind == "选择题" and not options:
            st.error("至少需要一个选项。")
        else:
            progress = st.progress(0)
            status = st.empty()
            live_preview = st.empty()

            def update_single_progress(done: int, total: int, records: list[dict]) -> None:
                progress.progress(done / max(1, total))
                status.write(f"正在并发生成：已完成 {done}/{total}")
                preview = pd.DataFrame(records).sort_values("order").tail(8) if records else pd.DataFrame()
                if not preview.empty:
                    preview = preview.drop(columns=["order"], errors="ignore")
                    if question_kind == "问答题":
                        live_preview.dataframe(zh_table_columns(preview[["pid", "twin_answer"]]), width="stretch", hide_index=True)
                    else:
                        live_preview.dataframe(zh_table_columns(preview[["pid", "twin", "twin_label", "raw_twin"]]), width="stretch", hide_index=True)

            with st.spinner("正在并发模拟回答..."):
                if question_kind == "问答题":
                    result = single_open_evaluation_for_language(
                        shown_rows,
                        qid,
                        question_text,
                        variant,
                        batch_size=single_batch_size,
                        workers=single_workers,
                        api_sleep=api_sleep,
                        progress_callback=update_single_progress,
                        **llm_kwargs(api_key, base_url, model, max_tokens),
                    )
                else:
                    result = single_item_evaluation_for_language(
                        shown_rows,
                        qid,
                        question_text,
                        options,
                        variant,
                        batch_size=single_batch_size,
                        workers=single_workers,
                        api_sleep=api_sleep,
                        progress_callback=update_single_progress,
                        **llm_kwargs(api_key, base_url, model, max_tokens),
                    )
            status.success(f"单题模拟完成：{len(result)} 条回答")
            st.session_state["single_result"] = result
            st.session_state["single_options"] = options
            st.session_state["single_result_kind"] = question_kind

    result = st.session_state.get("single_result")
    if result is not None:
        if result.empty:
            st.warning("本次没有生成任何记录，请增加模拟人数或换一个题目。")
        elif st.session_state.get("single_result_kind") == "问答题":
            answer_series = result.get("twin_answer", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
            empty_count = int(answer_series.eq("").sum())
            c1, c2 = st.columns(2)
            c1.metric("生成回答数", len(result))
            c2.metric("空回复数", empty_count)
            if empty_count:
                st.warning("有模型空回复，已在表格中标出；可以提高 token 上限或关闭 thinking 后重试。")
            st.markdown("**数字孪生回答预览**")
            for _, row in result.head(8).iterrows():
                st.markdown(f"**被试 {row.get('pid', 'NA')}**")
                st.write(visible_model_output(row.get("twin_answer")))
            st.dataframe(zh_table_columns(result), width="stretch", hide_index=True)
        else:
            c1, c2 = st.columns(2)
            c1.metric("孪生平均准确率", f"{result['acc_twin'].mean():.3f}" if result["acc_twin"].notna().any() else "NA")
            c2.metric("人类重测平均准确率", f"{result['acc_retest'].mean():.3f}" if result["acc_retest"].notna().any() else "NA")
            if result["twin"].isna().all():
                st.warning("模型输出未能解析成选项编号，请展开“原始模型输出”查看原因。")
            st.dataframe(zh_table_columns(result.drop(columns=["raw_twin"])), width="stretch", hide_index=True)
            with st.expander("原始模型输出", expanded=False):
                st.dataframe(zh_table_columns(result[["pid", "raw_twin"]]), width="stretch", hide_index=True)

with tabs[3]:
    batch_people = st.slider("批量画像数", 2, min(50, len(rows)), min(8, len(rows)))
    batch_items_n = st.slider("题目数", 1, 6, 2)
    run_variants = st.multiselect("画像版本组合", PROFILE_VARIANTS, default=[variant])
    include_price = st.toggle("包含商品价格题", value=False)
    batch_rows = rows[:batch_people]
    items = candidate_eval_items(batch_rows, max_items=batch_items_n, include_product_items=include_price)
    run_items = []
    preview_records = []
    for item in items:
        item_text = zh_text(item["text"], context=f"batch-question:{item['qid']}")
        item_options = zh_options(item["options"], context=f"batch-options:{item['qid']}")
        run_item = dict(item)
        run_item["text"] = item_text
        run_item["options"] = item_options
        run_items.append(run_item)
        preview_records.append({"qid": item["qid"], "options": len(item_options), "question": item_text})
    item_preview = pd.DataFrame(preview_records)
    st.dataframe(item_preview, width="stretch", hide_index=True)

    if st.button("运行批量评估", type="primary", width="stretch"):
        if not run_variants:
            st.error("至少选择一个画像版本。")
        else:
            wide_tables = []
            metric_tables = []
            progress = st.progress(0)
            for i, selected_variant in enumerate(run_variants, start=1):
                wide = evaluate_items_for_language(
                    batch_rows,
                    run_items,
                    selected_variant,
                    random_seed=int(random_seed),
                    api_sleep=api_sleep,
                    **llm_kwargs(api_key, base_url, model, max_tokens),
                )
                wide_tables.append(wide)
                metric_tables.append(summarize_prediction_table(wide))
                progress.progress(i / len(run_variants))
            st.session_state["batch_predictions"] = pd.concat(wide_tables, ignore_index=True)
            st.session_state["metric_table"] = pd.concat(metric_tables, ignore_index=True)

    metric_table = st.session_state.get("metric_table")
    batch_predictions = st.session_state.get("batch_predictions")
    if metric_table is not None and batch_predictions is not None:
        rounded = metric_table.copy()
        numeric_cols = ["accuracy", "correlation", "mean_diff_sd", "variance_ratio"]
        rounded[numeric_cols] = rounded[numeric_cols].round(3)
        st.dataframe(rounded, width="stretch", hide_index=True)
        twin_metrics = metric_table[metric_table["predictor"] == "twin"]
        if not twin_metrics.empty:
            chart_data = twin_metrics.melt(
                id_vars=["variant"],
                value_vars=[c for c in ["accuracy", "correlation", "variance_ratio"] if c in twin_metrics.columns],
                var_name="metric",
                value_name="value",
            )
            chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("variant:N", title=None),
                    y=alt.Y("value:Q", title=None),
                    color=alt.Color("metric:N", title=None),
                    column=alt.Column("metric:N", title=None),
                    tooltip=["variant", "metric", alt.Tooltip("value:Q", format=".3f")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width="stretch")
        with st.expander("预测明细", expanded=False):
            st.dataframe(batch_predictions.drop(columns=["raw_twin"]), width="stretch", hide_index=True)

with tabs[4]:
    c1, c2, c3 = st.columns([0.5, 0.25, 0.25])
    default_product = "可口可乐经典汽水，12液量盎司，12罐装" if st.session_state["use_chinese_text"] else "Coca-Cola Classic Soda, 12 fl oz, 12-pack"
    product = c1.text_input("商品", value=default_product, key=f"demand_product_{text_language}")
    regular_price = c2.number_input("常规价格", min_value=0.01, value=8.26, step=0.25)
    demand_people = c3.slider("画像数", 2, min(150, len(rows)), min(100, len(rows)))
    b1, b2 = st.columns(2)
    demand_batch_size = b1.slider("批大小", min_value=1, max_value=50, value=min(20, max(1, len(rows))), step=1)
    demand_workers = b2.slider("并发请求数", min_value=1, max_value=20, value=8, step=1)
    ratios_text = st.text_input("价格比例", value="0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    try:
        ratios = [float(item.strip()) for item in ratios_text.split(",") if item.strip()]
    except ValueError:
        ratios = []
        st.error("价格比例需要是逗号分隔数字。")

    if st.button("生成需求曲线", type="primary", width="stretch"):
        if ratios:
            selected_rows = rows[:demand_people]
            progress = st.progress(0)
            status = st.empty()
            live_chart = st.empty()
            live_summary = st.empty()
            live_detail = st.empty()
            process_records = []
            curve_records = []
            total_steps = max(1, len(selected_rows))
            completed = 0
            kwargs = llm_kwargs(api_key, base_url, model, max_tokens)
            use_chinese = st.session_state["use_chinese_text"]

            for batch_start in range(0, len(selected_rows), demand_batch_size):
                batch = selected_rows[batch_start : batch_start + demand_batch_size]
                future_to_row = {}
                with ThreadPoolExecutor(max_workers=max(1, min(demand_workers, len(batch)))) as executor:
                    for row in batch:
                        persona_text = zh_text(get_persona_variant(row, variant), context=f"demand-persona:{row['pid']}:{variant}")
                        future_to_row[
                            executor.submit(
                                simulate_demand_curve_for_persona,
                                persona_text,
                                product,
                                float(regular_price),
                                ratios,
                                use_chinese,
                                kwargs,
                            )
                        ] = row
                    for future in as_completed(future_to_row):
                        row = future_to_row[future]
                        try:
                            persona_records = future.result()
                        except Exception as exc:
                            persona_records = simulate_demand_curve_for_persona(
                                zh_text(get_persona_variant(row, variant), context=f"demand-persona-fallback:{row['pid']}:{variant}"),
                                product,
                                float(regular_price),
                                ratios,
                                use_chinese,
                                {**kwargs, "api_key": ""},
                            )
                            for item in persona_records:
                                item["raw"] = f"调用失败后使用备用规则：{safe_error_message(exc)}"
                                item["来源"] = "备用规则"
                        completed += 1
                        for item in persona_records:
                            process_records.append(
                                {
                                    **item,
                                    "pid": row["pid"],
                                    "batch": batch_start // demand_batch_size + 1,
                                }
                            )
                        process_df = pd.DataFrame(process_records)
                        demand_df = (
                            process_df.groupby(["price_ratio", "price_$"], as_index=False)["buy"]
                            .mean()
                            .rename(columns={"buy": "P(buy)"})
                            .sort_values("price_ratio")
                        )
                        progress.progress(completed / total_steps)
                        status.write(
                            f"正在生成：批次 {batch_start // demand_batch_size + 1}，已完成 {completed}/{total_steps} 个画像；"
                            f"每个画像一次性返回 {len(ratios)} 个价格点"
                        )
                        live_detail.dataframe(process_df.drop(columns=["buy"], errors="ignore").tail(24), width="stretch", hide_index=True)
                        live_summary.dataframe(demand_df, width="stretch", hide_index=True)
                        live_chart.altair_chart(demand_line_chart(demand_df), width="stretch")
                if api_key.strip() and api_sleep:
                    time.sleep(api_sleep)

            process_df = pd.DataFrame(process_records)
            if not process_df.empty:
                curve_records = (
                    process_df.groupby(["price_ratio", "price_$"], as_index=False)["buy"]
                    .mean()
                    .rename(columns={"buy": "P(buy)"})
                    .sort_values("price_ratio")
                    .to_dict("records")
                )

            status.success("需求曲线生成完成。")
            st.session_state["demand_df"] = pd.DataFrame(curve_records)
            st.session_state["demand_process"] = pd.DataFrame(process_records).drop(columns=["buy"], errors="ignore")

    demand_df = st.session_state.get("demand_df")
    if demand_df is not None:
        st.altair_chart(demand_line_chart(demand_df), width="stretch")
        st.dataframe(demand_df, width="stretch", hide_index=True)
        demand_process = st.session_state.get("demand_process")
        if demand_process is not None:
            with st.expander("生成过程明细", expanded=True):
                st.dataframe(demand_process, width="stretch", hide_index=True)

with tabs[5]:
    st.subheader("管理策略沙盘")
    st.markdown(
        "用同一批数字孪生顾客预演多种管理或营销策略，比较平均接受度、高意向比例和个体差异。"
    )
    s1, s2, s3 = st.columns([0.45, 0.25, 0.3])
    management_context = s1.text_input(
        "管理情境",
        value="一家饮料公司准备推出低糖气泡饮",
        key="strategy_context",
    )
    strategy_people = s2.slider("模拟画像数", 2, min(150, len(rows)), min(100, len(rows)), key="strategy_people")
    strategy_variant = s3.selectbox("沙盘画像版本", PROFILE_VARIANTS, index=0, format_func=zh_variant_label, key="strategy_variant")
    strategy_model = st.text_input("沙盘模型", value="deepseek-v4-pro", key="strategy_model")

    default_strategies = pd.DataFrame(
        [
            {"策略": "价格优惠", "方案文案": "首购立减20%，并提供一张下次购买可用的优惠券。"},
            {"策略": "健康收益", "方案文案": "强调低糖、低热量，同时保留清爽口感，适合日常饮用。"},
            {"策略": "环保叙事", "方案文案": "使用可回收包装，并承诺每售出一箱就支持社区回收项目。"},
            {"策略": "社会证明", "方案文案": "展示同龄消费者的真实评价，并突出复购率和口碑推荐。"},
        ]
    )
    strategy_table = st.data_editor(
        default_strategies,
        hide_index=True,
        width="stretch",
        num_rows="dynamic",
        key="strategy_editor",
    )
    strategy_rows = []
    for _, row in strategy_table.iterrows():
        name = "" if pd.isna(row.get("策略")) else str(row.get("策略")).strip()
        message = "" if pd.isna(row.get("方案文案")) else str(row.get("方案文案")).strip()
        if name and message:
            strategy_rows.append({"策略": name, "方案文案": message})

    p1, p2, p3 = st.columns(3)
    strategy_batch_size = p1.slider("策略沙盘批大小", min_value=1, max_value=50, value=20, step=1, key="strategy_batch_size")
    strategy_workers = p2.slider("策略沙盘并发请求数", min_value=1, max_value=20, value=8, step=1, key="strategy_workers")
    strategy_intent_threshold = p3.slider("高意向阈值（含）", min_value=2, max_value=5, value=3, step=1, key="strategy_intent_threshold")

    if st.button("运行策略沙盘", type="primary", width="stretch"):
        if not strategy_rows:
            st.error("至少需要一个策略。")
        else:
            selected_rows = rows[:strategy_people]
            progress = st.progress(0)
            status = st.empty()
            live_decision = st.empty()
            live_summary = st.empty()
            live_chart = st.empty()
            live_detail = st.empty()
            records = []
            completed = 0
            total_steps = max(1, len(strategy_rows) * len(selected_rows))
            kwargs = llm_kwargs(api_key, base_url, strategy_model, max(220, max_tokens))
            use_chinese = st.session_state["use_chinese_text"]

            for strategy in strategy_rows:
                strategy_name = strategy["策略"]
                strategy_message = strategy["方案文案"]
                for batch_start in range(0, len(selected_rows), strategy_batch_size):
                    batch = selected_rows[batch_start : batch_start + strategy_batch_size]
                    future_to_row = {}
                    with ThreadPoolExecutor(max_workers=max(1, min(strategy_workers, len(batch)))) as executor:
                        for row in batch:
                            persona_text = zh_text(get_persona_variant(row, strategy_variant), context=f"strategy-persona:{row['pid']}:{strategy_variant}")
                            future_to_row[
                                executor.submit(
                                    simulate_strategy_answer,
                                    persona_text,
                                    management_context,
                                    strategy_name,
                                    strategy_message,
                                    use_chinese,
                                    kwargs,
                                )
                            ] = row
                        for future in as_completed(future_to_row):
                            row = future_to_row[future]
                            try:
                                decision = future.result()
                            except Exception as exc:
                                decision = {
                                    "score": None,
                                    "reason": "",
                                    "concern": "",
                                    "implication": "",
                                    "raw": f"{type(exc).__name__}: {exc}",
                                }
                            score = decision["score"]
                            records.append(
                                {
                                    "策略": strategy_name,
                                    "被试ID": row["pid"],
                                    "接受度": score,
                                    "高意向": 1 if score is not None and score >= strategy_intent_threshold else (0 if score is not None else None),
                                    "主要理由": decision["reason"],
                                    "主要顾虑": decision["concern"],
                                    "管理启示": decision["implication"],
                                    "批次": batch_start // strategy_batch_size + 1,
                                    "模型原始输出": decision["raw"],
                                }
                            )
                            completed += 1
                            progress.progress(completed / total_steps)
                            live_decision.info(
                                f"最新决策：策略「{strategy_name}」 · 被试 {row['pid']} · 接受度 {score if score else 'NA'} · "
                                f"理由：{decision['reason'] or 'NA'} · 顾虑：{decision['concern'] or 'NA'}"
                            )

                        result_df = pd.DataFrame(records)
                        summary = summarize_strategy_results(result_df, strategy_intent_threshold)
                        status.write(f"正在生成：{strategy_name}，批次 {batch_start // strategy_batch_size + 1}，已完成 {completed}/{total_steps}")
                        live_summary.dataframe(summary.round({"平均接受度": 3, "高意向比例": 3}), width="stretch", hide_index=True)
                    live_detail.dataframe(result_df.tail(24), width="stretch", hide_index=True)
                    dist_df = (
                        result_df.dropna(subset=["接受度"])
                        .assign(接受度=lambda frame: frame["接受度"].astype(int).astype(str))
                        .groupby(["策略", "接受度"], as_index=False)
                        .size()
                    )
                    high_chart = (
                        alt.Chart(summary)
                        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                        .encode(
                            x=alt.X("策略:N", title=None, sort="-y"),
                            y=alt.Y("高意向比例:Q", title=f"高意向比例（≥{strategy_intent_threshold}）", scale=alt.Scale(domain=[0, 1])),
                            tooltip=[alt.Tooltip("策略:N"), alt.Tooltip("平均接受度:Q", format=".3f"), alt.Tooltip("高意向比例:Q", format=".3f"), alt.Tooltip("样本数:Q"), alt.Tooltip("有效评分数:Q")],
                        )
                        .properties(height=260, title=f"高意向比例（≥{strategy_intent_threshold}）")
                    )
                    dist_chart = (
                        alt.Chart(dist_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("策略:N", title=None),
                            y=alt.Y("size:Q", title="人数"),
                            color=alt.Color("接受度:N", title="接受度", scale=alt.Scale(scheme="redyellowgreen")),
                            tooltip=[alt.Tooltip("策略:N"), alt.Tooltip("接受度:N"), alt.Tooltip("size:Q", title="人数")],
                        )
                        .properties(height=260, title="接受度分布")
                    )
                    live_chart.altair_chart(alt.vconcat(high_chart, dist_chart).resolve_scale(color="independent"), width="stretch")
                    if api_key.strip() and api_sleep:
                        time.sleep(api_sleep)

            status.success("策略沙盘生成完成。")
            st.session_state["strategy_results"] = pd.DataFrame(records)
            st.session_state["strategy_summary"] = summarize_strategy_results(pd.DataFrame(records), strategy_intent_threshold)
            st.session_state["strategy_narrative"] = generate_strategy_summary(
                st.session_state["strategy_summary"],
                pd.DataFrame(records),
                management_context,
                api_key,
                base_url,
                strategy_model,
                use_chinese,
            )

    strategy_summary = st.session_state.get("strategy_summary")
    strategy_results = st.session_state.get("strategy_results")
    strategy_narrative = st.session_state.get("strategy_narrative")
    if strategy_summary is not None and strategy_results is not None:
        strategy_summary = summarize_strategy_results(strategy_results, strategy_intent_threshold)
        st.session_state["strategy_summary"] = strategy_summary
        if st.button("重新生成整体总结", width="stretch"):
            st.session_state["strategy_narrative"] = generate_strategy_summary(
                strategy_summary,
                strategy_results,
                management_context,
                api_key,
                base_url,
                strategy_model,
                st.session_state["use_chinese_text"],
            )
            strategy_narrative = st.session_state["strategy_narrative"]
        if strategy_narrative:
            st.markdown("**整体总结**")
            st.info(strategy_narrative)
        st.markdown("**策略对比结果**")
        st.dataframe(strategy_summary.round({"平均接受度": 3, "高意向比例": 3}), width="stretch", hide_index=True)
        high_chart = (
            alt.Chart(strategy_summary)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("策略:N", title=None, sort="-y"),
                y=alt.Y("高意向比例:Q", title=f"高意向比例（≥{strategy_intent_threshold}）", scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip("策略:N"), alt.Tooltip("平均接受度:Q", format=".3f"), alt.Tooltip("高意向比例:Q", format=".3f"), alt.Tooltip("样本数:Q"), alt.Tooltip("有效评分数:Q")],
            )
            .properties(height=280, title=f"高意向比例（≥{strategy_intent_threshold}）")
        )
        mean_chart = (
            alt.Chart(strategy_summary)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("策略:N", title=None, sort="-y"),
                y=alt.Y("平均接受度:Q", title="平均接受度", scale=alt.Scale(domain=[1, 5])),
                tooltip=[alt.Tooltip("策略:N"), alt.Tooltip("平均接受度:Q", format=".3f"), alt.Tooltip("高意向比例:Q", format=".3f"), alt.Tooltip("样本数:Q"), alt.Tooltip("有效评分数:Q")],
            )
            .properties(height=280, title="平均接受度")
        )
        dist_df = (
            strategy_results.dropna(subset=["接受度"])
            .assign(接受度=lambda frame: frame["接受度"].astype(int).astype(str))
            .groupby(["策略", "接受度"], as_index=False)
            .size()
        )
        dist_chart = (
            alt.Chart(dist_df)
            .mark_bar()
            .encode(
                x=alt.X("策略:N", title=None),
                y=alt.Y("size:Q", title="人数"),
                color=alt.Color("接受度:N", title="接受度", scale=alt.Scale(scheme="redyellowgreen")),
                tooltip=[alt.Tooltip("策略:N"), alt.Tooltip("接受度:N"), alt.Tooltip("size:Q", title="人数")],
            )
            .properties(height=280, title="接受度分布")
        )
        position_chart = (
            alt.Chart(strategy_summary)
            .mark_circle(opacity=0.78)
            .encode(
                x=alt.X("平均接受度:Q", title="平均接受度", scale=alt.Scale(domain=[1, 5])),
                y=alt.Y("高意向比例:Q", title="高意向比例", scale=alt.Scale(domain=[0, 1])),
                size=alt.Size("样本数:Q", title="样本数", scale=alt.Scale(range=[180, 900])),
                color=alt.Color("策略:N", title="策略"),
                tooltip=[alt.Tooltip("策略:N"), alt.Tooltip("平均接受度:Q", format=".3f"), alt.Tooltip("高意向比例:Q", format=".3f"), alt.Tooltip("样本数:Q"), alt.Tooltip("有效评分数:Q")],
            )
            .properties(height=300, title="策略定位图")
        )
        strategy_heat = (
            alt.Chart(dist_df)
            .mark_rect(cornerRadius=2)
            .encode(
                x=alt.X("接受度:N", title="接受度"),
                y=alt.Y("策略:N", title=None),
                color=alt.Color("size:Q", title="人数", scale=alt.Scale(scheme="yellowgreenblue")),
                tooltip=[alt.Tooltip("策略:N"), alt.Tooltip("接受度:N"), alt.Tooltip("size:Q", title="人数")],
            )
            .properties(height=300, title="策略 × 接受度热力图")
        )
        st.altair_chart(alt.hconcat(high_chart, mean_chart).resolve_scale(color="independent"), width="stretch")
        st.altair_chart(alt.hconcat(position_chart, strategy_heat).resolve_scale(color="independent"), width="stretch")
        st.altair_chart(dist_chart, width="stretch")
        st.markdown("**理由样例**")
        reason_cols = ["策略", "被试ID", "接受度", "主要理由", "主要顾虑", "管理启示"]
        missing_reason_cols = [col for col in reason_cols if col not in strategy_results.columns]
        if missing_reason_cols:
            st.info("当前沙盘结果来自旧版结构，重新运行一次策略沙盘后会显示完整理由、顾虑和管理启示。")
        sample_reasons = strategy_results.reindex(columns=reason_cols).fillna("").tail(12)
        st.dataframe(sample_reasons, width="stretch", hide_index=True)
        with st.expander("个体级模拟明细", expanded=False):
            st.dataframe(strategy_results, width="stretch", hide_index=True)

with tabs[6]:
    st.subheader("市场细分雷达")
    g1, g2, g3 = st.columns([0.46, 0.24, 0.3])
    segment_context = g1.text_input(
        "细分情境",
        value="低糖气泡饮进入校园便利店渠道",
        key="segment_context",
    )
    segment_people = g2.slider("细分画像数", 2, min(150, len(rows)), min(100, len(rows)), key="segment_people")
    segment_variant = g3.selectbox("细分画像版本", PROFILE_VARIANTS, index=0, format_func=zh_variant_label, key="segment_variant")
    segment_model = st.text_input("细分模型", value="deepseek-v4-pro", key="segment_model")
    q1, q2 = st.columns(2)
    segment_batch_size = q1.slider("细分批大小", min_value=1, max_value=50, value=20, step=1, key="segment_batch_size")
    segment_workers = q2.slider("细分并发请求数", min_value=1, max_value=20, value=8, step=1, key="segment_workers")

    if st.button("运行细分雷达", type="primary", width="stretch"):
        selected_rows = rows[:segment_people]
        progress = st.progress(0)
        status = st.empty()
        live_decision = st.empty()
        live_summary = st.empty()
        live_chart = st.empty()
        live_detail = st.empty()
        records = []
        completed = 0
        total_steps = max(1, len(selected_rows))
        kwargs = llm_kwargs(api_key, base_url, segment_model, max(260, max_tokens))
        use_chinese = st.session_state["use_chinese_text"]

        for batch_start in range(0, len(selected_rows), segment_batch_size):
            batch = selected_rows[batch_start : batch_start + segment_batch_size]
            future_to_row = {}
            with ThreadPoolExecutor(max_workers=max(1, min(segment_workers, len(batch)))) as executor:
                for row in batch:
                    persona_text = zh_text(get_persona_variant(row, segment_variant), context=f"segment-persona:{row['pid']}:{segment_variant}")
                    future_to_row[
                        executor.submit(
                            simulate_segment_answer,
                            persona_text,
                            segment_context,
                            use_chinese,
                            kwargs,
                        )
                    ] = row
                for future in as_completed(future_to_row):
                    row = future_to_row[future]
                    try:
                        segment = future.result()
                    except Exception as exc:
                        segment = {
                            **{dim: None for dim in SEGMENT_DIMENSIONS},
                            "分群": "调用失败",
                            "理由": f"{type(exc).__name__}: {exc}",
                            "模型原始输出": f"{type(exc).__name__}: {exc}",
                        }
                    record = {
                        "被试ID": row["pid"],
                        "批次": batch_start // segment_batch_size + 1,
                        "分群": segment.get("分群") or "未分类",
                        "理由": segment.get("理由") or "",
                        "模型原始输出": segment.get("模型原始输出") or "",
                    }
                    for dim in SEGMENT_DIMENSIONS:
                        record[dim] = segment.get(dim)
                    records.append(record)
                    completed += 1
                    progress.progress(completed / total_steps)
                    dim_scores = " / ".join(f"{dim}{record[dim] if record[dim] is not None else 'NA'}" for dim in SEGMENT_DIMENSIONS)
                    live_decision.info(f"最新细分：被试 {row['pid']} · {record['分群']} · {dim_scores} · 理由：{record['理由'] or 'NA'}")

            result_df = pd.DataFrame(records)
            segment_data, summary, dim_avg, segment_dim = segment_tables(result_df)
            status.write(f"正在生成：批次 {batch_start // segment_batch_size + 1}，已完成 {completed}/{total_steps}")
            if not summary.empty:
                live_summary.dataframe(summary.round({"占比": 3}), width="stretch", hide_index=True)
                live_chart.altair_chart(segment_overview_chart(summary, dim_avg), width="stretch")
            live_detail.dataframe(result_df.tail(24), width="stretch", hide_index=True)
            if api_key.strip() and api_sleep:
                time.sleep(api_sleep)

        segment_data, summary, dim_avg, segment_dim = segment_tables(pd.DataFrame(records))
        status.success("细分雷达生成完成。")
        st.session_state["segment_results"] = segment_data
        st.session_state["segment_summary"] = summary
        st.session_state["segment_avg"] = dim_avg
        st.session_state["segment_dim"] = segment_dim

    segment_results = st.session_state.get("segment_results")
    segment_summary = st.session_state.get("segment_summary")
    segment_avg = st.session_state.get("segment_avg")
    segment_dim = st.session_state.get("segment_dim")
    if segment_results is not None and segment_summary is not None and segment_avg is not None and segment_dim is not None and not segment_results.empty:
        dominant_segment = segment_summary.iloc[0]["分群"] if not segment_summary.empty else "暂无"
        strongest_dim = segment_avg.sort_values("平均分", ascending=False).iloc[0]["维度"] if not segment_avg.empty else "暂无"
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("样本数", len(segment_results))
        m2.metric("分群数", segment_results["分群"].nunique())
        m3.metric("最大分群", dominant_segment)
        m4.metric("最强维度", strongest_dim)

        st.markdown("**分群汇总**")
        st.dataframe(segment_summary.round({"占比": 3}), width="stretch", hide_index=True)
        st.altair_chart(segment_overview_chart(segment_summary, segment_avg), width="stretch")
        st.altair_chart(alt.hconcat(segment_heatmap_chart(segment_dim), segment_bubble_chart(segment_dim)).resolve_scale(color="independent"), width="stretch")
        st.altair_chart(alt.hconcat(segment_scatter_chart(segment_results), segment_box_chart(segment_results)).resolve_scale(color="independent"), width="stretch")
        st.altair_chart(radar_chart(segment_avg), width="stretch")

        st.markdown("**理由样例**")
        reason_cols = ["被试ID", "分群"] + SEGMENT_DIMENSIONS + ["理由"]
        st.dataframe(segment_results.reindex(columns=reason_cols).fillna("").tail(16), width="stretch", hide_index=True)
        with st.expander("个体级细分明细", expanded=False):
            st.dataframe(segment_results, width="stretch", hide_index=True)


with tabs[7]:
    st.subheader("市场调研")
    st.markdown("围绕一个产品概念或营销假设，批量访谈数字孪生画像，汇总购买意向、支付意愿、核心卖点、主要顾虑和推荐渠道。")
    r1, r2, r3 = st.columns([0.44, 0.24, 0.32])
    research_goal = r1.text_input(
        "调研目标",
        value="评估低糖气泡饮进入校园便利店渠道的早期市场反馈",
        key="market_research_goal",
    )
    market_people = r2.slider("调研画像数", 2, min(150, len(rows)), min(60, len(rows)), key="market_people")
    market_variant = r3.selectbox("调研画像版本", PROFILE_VARIANTS, index=0, format_func=zh_variant_label, key="market_variant")
    product_concept = st.text_area(
        "产品/概念",
        value="一款低糖气泡饮，主打清爽口感、低热量、适合学习和工作间隙饮用，计划先在校园便利店和电商渠道试销。",
        height=110,
        key="market_product_concept",
    )
    question_table = st.data_editor(
        pd.DataFrame({"访谈问题": MARKET_RESEARCH_DEFAULT_QUESTIONS}),
        hide_index=True,
        width="stretch",
        num_rows="dynamic",
        key="market_question_editor",
    )
    research_questions = []
    for _, row in question_table.iterrows():
        question = "" if pd.isna(row.get("访谈问题")) else str(row.get("访谈问题")).strip()
        if question:
            research_questions.append(question)

    m1, m2, m3, m4 = st.columns(4)
    market_model = m1.text_input("调研模型", value="deepseek-v4-pro", key="market_model")
    market_batch_size = m2.slider("调研批大小", min_value=1, max_value=50, value=20, step=1, key="market_batch_size")
    market_workers = m3.slider("调研并发请求数", min_value=1, max_value=20, value=8, step=1, key="market_workers")
    market_intent_threshold = m4.slider("高意向阈值（含）", min_value=2, max_value=5, value=4, step=1, key="market_intent_threshold")

    if st.button("运行市场调研", type="primary", width="stretch"):
        if not str(research_goal or "").strip():
            st.error("调研目标不能为空。")
        elif not str(product_concept or "").strip():
            st.error("产品/概念不能为空。")
        elif not research_questions:
            st.error("至少需要一个访谈问题。")
        else:
            selected_rows = rows[:market_people]
            progress = st.progress(0)
            status = st.empty()
            live_decision = st.empty()
            live_summary = st.empty()
            live_detail = st.empty()
            records = []
            completed = 0
            total_steps = max(1, len(selected_rows))
            kwargs = llm_kwargs(api_key, base_url, market_model, max(380, max_tokens))
            use_chinese = st.session_state["use_chinese_text"]

            for batch_start in range(0, len(selected_rows), market_batch_size):
                batch = selected_rows[batch_start : batch_start + market_batch_size]
                future_to_row = {}
                with ThreadPoolExecutor(max_workers=max(1, min(market_workers, len(batch)))) as executor:
                    for row in batch:
                        persona_text = zh_text(get_persona_variant(row, market_variant), context=f"market-persona:{row['pid']}:{market_variant}")
                        future_to_row[
                            executor.submit(
                                simulate_market_research_answer,
                                persona_text,
                                research_goal,
                                product_concept,
                                research_questions,
                                use_chinese,
                                kwargs,
                            )
                        ] = row
                    for future in as_completed(future_to_row):
                        row = future_to_row[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = parse_market_research_response(
                                fallback_market_research_answer(
                                    zh_text(get_persona_variant(row, market_variant), context=f"market-persona-fallback:{row['pid']}:{market_variant}"),
                                    research_goal,
                                    product_concept,
                                    research_questions,
                                )
                            )
                            result["来源"] = "备用规则"
                            result["模型原始输出"] = f"调用失败后使用备用规则：{safe_error_message(exc)}"
                        intent = result.get("购买意向")
                        records.append(
                            {
                                "被试ID": row["pid"],
                                "批次": batch_start // market_batch_size + 1,
                                "购买意向": intent,
                                "高意向": 1 if intent is not None and intent >= market_intent_threshold else (0 if intent is not None else None),
                                "支付意愿": result.get("支付意愿"),
                                "核心卖点": result.get("核心卖点"),
                                "主要顾虑": result.get("主要顾虑"),
                                "推荐渠道": result.get("推荐渠道"),
                                "文案角度": result.get("文案角度"),
                                "用户原话": result.get("用户原话"),
                                "访谈回答": result.get("访谈回答"),
                                "来源": result.get("来源"),
                                "模型原始输出": result.get("模型原始输出"),
                            }
                        )
                        completed += 1
                        progress.progress(completed / total_steps)
                        live_decision.info(
                            f"最新访谈：被试 {row['pid']} · 购买意向 {intent if intent else 'NA'} · "
                            f"卖点：{result.get('核心卖点') or 'NA'} · 顾虑：{result.get('主要顾虑') or 'NA'}"
                        )

                result_df = pd.DataFrame(records)
                status.write(f"正在生成：批次 {batch_start // market_batch_size + 1}，已完成 {completed}/{total_steps}")
                if not result_df.empty:
                    live_summary.dataframe(
                        result_df.reindex(columns=["被试ID", "购买意向", "支付意愿", "核心卖点", "主要顾虑", "推荐渠道"]).tail(24),
                        width="stretch",
                        hide_index=True,
                    )
                    live_detail.dataframe(result_df.tail(24), width="stretch", hide_index=True)
                if api_key.strip() and api_sleep:
                    time.sleep(api_sleep)

            market_results = pd.DataFrame(records)
            status.success("市场调研生成完成。")
            st.session_state["market_research_results"] = market_results
            st.session_state["market_research_narrative"] = generate_market_research_summary(
                market_results,
                product_concept,
                research_goal,
                api_key,
                base_url,
                market_model,
                use_chinese,
                market_intent_threshold,
            )

    market_results = st.session_state.get("market_research_results")
    market_narrative = st.session_state.get("market_research_narrative")
    if market_results is not None and not market_results.empty:
        market_results = market_results.copy()
        market_results["购买意向"] = pd.to_numeric(market_results.get("购买意向"), errors="coerce")
        market_results["支付意愿"] = pd.to_numeric(market_results.get("支付意愿"), errors="coerce")
        valid_intent = market_results["购买意向"].dropna()
        high_share = valid_intent.ge(market_intent_threshold).mean() if not valid_intent.empty else float("nan")
        barrier_counts = top_label_counts(market_results.get("主要顾虑", pd.Series(dtype=str)), "主要顾虑")
        channel_counts = top_label_counts(market_results.get("推荐渠道", pd.Series(dtype=str)), "推荐渠道")
        need_counts = top_label_counts(market_results.get("核心卖点", pd.Series(dtype=str)), "核心卖点")
        top_barrier = barrier_counts.iloc[0]["主要顾虑"] if not barrier_counts.empty else "暂无"
        top_channel = channel_counts.iloc[0]["推荐渠道"] if not channel_counts.empty else "暂无"
        top_need = need_counts.iloc[0]["核心卖点"] if not need_counts.empty else "暂无"

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("样本数", len(market_results))
        k2.metric("平均购买意向", f"{valid_intent.mean():.2f}" if not valid_intent.empty else "NA")
        k3.metric("高意向比例", f"{high_share:.1%}" if not pd.isna(high_share) else "NA")
        k4.metric("最常见卖点", top_need)
        k5.metric("主要顾虑", top_barrier)

        if st.button("重新生成调研总结", width="stretch"):
            st.session_state["market_research_narrative"] = generate_market_research_summary(
                market_results,
                product_concept,
                research_goal,
                api_key,
                base_url,
                market_model,
                st.session_state["use_chinese_text"],
                market_intent_threshold,
            )
            market_narrative = st.session_state["market_research_narrative"]
        if market_narrative:
            st.markdown("**调研总结**")
            st.info(market_narrative)

        intent_dist = (
            market_results.dropna(subset=["购买意向"])
            .assign(购买意向=lambda frame: frame["购买意向"].astype(int).astype(str))
            .groupby(["购买意向"], as_index=False)
            .size()
        )
        if not intent_dist.empty:
            intent_chart = (
                alt.Chart(intent_dist)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("购买意向:N", title="购买意向"),
                    y=alt.Y("size:Q", title="人数"),
                    color=alt.Color("购买意向:N", title="购买意向", scale=alt.Scale(scheme="redyellowgreen")),
                    tooltip=[alt.Tooltip("购买意向:N"), alt.Tooltip("size:Q", title="人数")],
                )
                .properties(height=280, title="购买意向分布")
            )
            st.altair_chart(intent_chart, width="stretch")

        chart_left, chart_right = st.columns(2)
        if not barrier_counts.empty:
            barrier_chart = (
                alt.Chart(barrier_counts.head(8))
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("人数:Q", title="人数"),
                    y=alt.Y("主要顾虑:N", title=None, sort="-x"),
                    tooltip=[alt.Tooltip("主要顾虑:N"), alt.Tooltip("人数:Q"), alt.Tooltip("占比:Q", format=".1%")],
                )
                .properties(height=280, title="主要顾虑")
            )
            chart_left.altair_chart(barrier_chart, width="stretch")
        if not channel_counts.empty:
            channel_chart = (
                alt.Chart(channel_counts.head(8))
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("人数:Q", title="人数"),
                    y=alt.Y("推荐渠道:N", title=None, sort="-x"),
                    tooltip=[alt.Tooltip("推荐渠道:N"), alt.Tooltip("人数:Q"), alt.Tooltip("占比:Q", format=".1%")],
                )
                .properties(height=280, title=f"推荐渠道（当前首选：{top_channel}）")
            )
            chart_right.altair_chart(channel_chart, width="stretch")

        st.markdown("**主题汇总**")
        summary_cols = st.columns(3)
        summary_cols[0].dataframe(need_counts.head(8), width="stretch", hide_index=True)
        summary_cols[1].dataframe(barrier_counts.head(8), width="stretch", hide_index=True)
        summary_cols[2].dataframe(channel_counts.head(8), width="stretch", hide_index=True)
        st.markdown("**访谈明细**")
        detail_cols = ["被试ID", "购买意向", "支付意愿", "核心卖点", "主要顾虑", "推荐渠道", "文案角度", "用户原话"]
        st.dataframe(market_results.reindex(columns=detail_cols), width="stretch", hide_index=True)
        with st.expander("完整访谈回答与模型输出", expanded=False):
            st.dataframe(market_results, width="stretch", hide_index=True)


with tabs[8]:
    prompt_row = df.iloc[0]
    prompt_persona = zh_text(get_persona_text(prompt_row), context=f"prompt-persona:{prompt_row['pid']}")
    prompt_question = zh_text(LINDA_TEXT, context="prompt-linda")
    prompt_options = zh_options(LINDA_OPTS, context="prompt-linda-options")
    prompt = build_user_prompt_for_language(prompt_persona, prompt_question, prompt_options)
    st.subheader("System")
    system_text = ZH_SYSTEM_MESSAGE if st.session_state["use_chinese_text"] else SYSTEM_MESSAGE
    st.markdown(f"<div class='wrapped-prompt'>{html.escape(system_text)}</div>", unsafe_allow_html=True)
    st.subheader("User")
    st.text_area("prompt", prompt, height=460, label_visibility="collapsed")
    if st.button("用该提示词模拟一次", width="stretch"):
        value, raw = simulate_answer_for_language(prompt_persona, prompt_question, prompt_options, **llm_kwargs(api_key, base_url, model, max_tokens))
        st.success(f"选项 {value}: {prompt_options[value - 1] if value else 'NA'}")
        st.code(raw, language="text")

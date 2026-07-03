from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATASET_REPO = "LLM-Digital-Twin/Twin-2K-500"
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


def disable_proxy_env() -> None:
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)


disable_proxy_env()
SYSTEM_MESSAGE = (
    "You are an AI expert at predicting how a specific person would answer survey questions. "
    "You are given a persona profile (the person's past survey answers) and a new question with "
    "format instructions. Answer exactly as this persona would, staying consistent with their "
    "profile, and follow the format instructions precisely."
)
LINDA_TEXT = (
    "Linda is 31, single, outspoken and very bright; she majored in philosophy and was "
    "concerned with discrimination and social justice. It is ___ that Linda is a bank teller "
    "and is active in the feminist movement."
)
LINDA_OPTS = [
    "Extremely improbable",
    "Very improbable",
    "Somewhat probable",
    "Moderately probable",
    "Very probable",
    "Extremely probable",
]
PROFILE_VARIANTS = ["full_or_summary", "demographics_only", "empty"]


def _strip(value) -> str:
    if value is None:
        return ""
    text = re.sub(r"<[^>]*>", " ", str(value)).replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def _q_mc(qid: str, text: str, options: list[str], selected_pos: int) -> dict:
    return {
        "QuestionID": qid,
        "QuestionText": text,
        "QuestionType": "MC",
        "Options": options,
        "Settings": {"Selector": "SAVR"},
        "Answers": {
            "SelectedByPosition": selected_pos,
            "SelectedText": options[selected_pos - 1],
        },
    }


def _make_synthetic(n: int = 8) -> pd.DataFrame:
    rng = random.Random(42)
    regions = ["Northeast US", "Midwest US", "South US", "West US"]
    ages = ["18-29", "30-49", "50-64", "65+"]
    incomes = ["<$30k", "$30k-$60k", "$60k-$100k", "$100k-$150k", ">$150k"]
    politics = ["Very liberal", "Liberal", "Moderate", "Conservative", "Very conservative"]
    rows = []
    for i in range(n):
        pid = f"SYN{i:03d}"
        persona = [
            {
                "BlockName": "Demographics",
                "Questions": [
                    _q_mc("QID_region", "Which part of the US do you live in?", regions, rng.randint(1, 4)),
                    _q_mc("QID_age", "How old are you?", ages, rng.randint(1, 4)),
                    _q_mc("QID_income", "What is your household income?", incomes, rng.randint(1, 5)),
                    _q_mc("QID_politics", "Politically, how do you describe yourself?", politics, rng.randint(1, 5)),
                ],
            },
            {
                "BlockName": "Personality (Need for Cognition, 1 item)",
                "Questions": [
                    _q_mc(
                        "QID_nfc",
                        "I really enjoy a task that involves coming up with new solutions.",
                        ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                        rng.randint(1, 5),
                    )
                ],
            },
        ]
        linda_true = rng.choice([1, 2, 2, 3])
        ground_truth = [
            {"BlockName": "Heuristics & Biases (holdout)", "Questions": [_q_mc("QID_linda", LINDA_TEXT, LINDA_OPTS, linda_true)]}
        ]
        linda_retest = min(6, max(1, linda_true + rng.choice([-1, 0, 0, 1])))
        retest = [
            {"BlockName": "Heuristics & Biases (holdout)", "Questions": [_q_mc("QID_linda", LINDA_TEXT, LINDA_OPTS, linda_retest)]}
        ]
        rows.append(
            {
                "pid": pid,
                "persona_json": json.dumps(persona),
                "persona_text": None,
                "ground_truth": json.dumps(ground_truth),
                "retest": json.dumps(retest),
                "summary": None,
            }
        )
    return pd.DataFrame(rows)


def _read_local_parquet_config(root: Path, config_name: str) -> pd.DataFrame:
    combined = root / "data" / "twin_2k_500_local" / f"{config_name}.parquet"
    if combined.exists():
        return pd.read_parquet(combined)
    chunk_dir = root / "data" / "Twin-2K-500" / config_name / "chunks"
    files = sorted(chunk_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet chunks found under {chunk_dir}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def _standardize_wave_split(ws: pd.DataFrame, num_personas: int) -> pd.DataFrame:
    ws = ws.head(num_personas).copy()
    persona_text = ws["wave1_3_persona_text"].tolist() if "wave1_3_persona_text" in ws.columns else [None] * len(ws)
    return pd.DataFrame(
        {
            "pid": ws["pid"].tolist(),
            "persona_json": ws["wave1_3_persona_json"].tolist(),
            "persona_text": persona_text,
            "ground_truth": ws["wave4_Q_wave1_3_A"].tolist(),
            "retest": ws["wave4_Q_wave4_A"].tolist(),
        }
    )


def load_data(num_personas: int = 30, root: Path | None = None, use_hub: bool = True) -> tuple[pd.DataFrame, str]:
    root = root or Path.cwd()
    try:
        ws = _read_local_parquet_config(root, "wave_split")
        df = _standardize_wave_split(ws, num_personas)
        try:
            fp = _read_local_parquet_config(root, "full_persona")
            summaries = dict(zip(fp["pid"].astype(str), fp["persona_summary"]))
            df["summary"] = df["pid"].astype(str).map(lambda pid: summaries.get(pid))
        except Exception:
            df["summary"] = None
        return df, "local parquet"
    except Exception:
        pass

    if use_hub:
        try:
            from datasets import load_dataset

            ws = load_dataset(DATASET_REPO, "wave_split", split="data")
            df = _standardize_wave_split(ws.to_pandas(), num_personas)
            try:
                fp = load_dataset(DATASET_REPO, "full_persona", split="data").to_pandas()
                summaries = dict(zip(fp["pid"].astype(str), fp["persona_summary"]))
                df["summary"] = df["pid"].astype(str).map(lambda pid: summaries.get(pid))
            except Exception:
                df["summary"] = None
            return df, "huggingface"
        except Exception:
            pass

    return _make_synthetic(max(num_personas, 8)).head(num_personas), "synthetic"


def persona_json_to_text(blocks: list[dict]) -> str:
    lines: list[str] = []
    for block in blocks:
        lines.append(f"## {block.get('BlockName', 'Section')}")
        for question in block.get("Questions", []):
            lines.append(f"Q: {_strip(question.get('QuestionText'))}")
            options = question.get("Options")
            if options:
                lines.append("Options: " + "; ".join(f"{i + 1}={_strip(option)}" for i, option in enumerate(options)))
            selected = question.get("Answers", {}).get("SelectedText")
            if isinstance(selected, list):
                selected = ", ".join(map(_strip, selected))
            lines.append(f"Answer: {_strip(selected) if selected else '[no answer]'}")
            lines.append("")
    return "\n".join(lines)


def get_persona_text(row) -> str:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    summary = row.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary
    raw_text = row.get("persona_text")
    if isinstance(raw_text, str) and raw_text.strip():
        return raw_text
    return persona_json_to_text(json.loads(row["persona_json"]))


def get_demographics_text(row) -> str:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    blocks = json.loads(row["persona_json"])
    keywords = ("demographic", "gender", "age", "education", "income", "region")
    keep = [block for block in blocks if any(keyword in str(block.get("BlockName", "")).lower() for keyword in keywords)]
    return persona_json_to_text(keep or blocks[:1])


def get_persona_variant(row, variant: str = "full_or_summary") -> str:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    if variant in {"full", "full_text"}:
        raw_text = row.get("persona_text")
        return raw_text if isinstance(raw_text, str) and raw_text.strip() else persona_json_to_text(json.loads(row["persona_json"]))
    if variant in {"summary", "full_or_summary"}:
        return get_persona_text(row)
    if variant == "demographics_only":
        return get_demographics_text(row)
    if variant == "empty":
        return "No individual persona information is provided. Predict the answer using only the question and options."
    raise ValueError(f"Unknown persona variant: {variant}")


def persona_blocks(row) -> list[dict]:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    return json.loads(row["persona_json"])


def inspect_persona_rows(blocks: list[dict], max_blocks: int = 3, max_q: int = 3) -> pd.DataFrame:
    rows = []
    for block in blocks[:max_blocks]:
        for question in block.get("Questions", [])[:max_q]:
            selected = question.get("Answers", {}).get("SelectedText", "[no answer]")
            if isinstance(selected, list):
                selected = ", ".join(map(str, selected))
            rows.append(
                {
                    "block": block.get("BlockName", "?"),
                    "question": _strip(question.get("QuestionText")),
                    "type": question.get("QuestionType"),
                    "answer": _strip(selected),
                }
            )
    return pd.DataFrame(rows)


def build_user_prompt(persona_text: str, question_text: str, options: list[str], fmt: str = "Only return the option number, no other text.") -> str:
    opt_block = ""
    if options:
        opt_block = "Options:\n" + "\n".join(f"  {i + 1} = {option}" for i, option in enumerate(options)) + "\n"
    return (
        f"PERSONA PROFILE:\n{persona_text}\n\n"
        f"QUESTION: {question_text}\n\n"
        f"{opt_block}\n"
        f"FORMAT INSTRUCTIONS: {fmt}"
    )


def call_llm(
    system: str,
    user: str,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int = 5,
    thinking: str = "disabled",
    strict_max_tokens: bool = False,
) -> str:
    from openai import OpenAI
    import httpx

    disable_proxy_env()
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(timeout=60.0, trust_env=False),
    )
    request = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if model.startswith("deepseek-v4") and thinking in {"enabled", "disabled"}:
        request["extra_body"] = {"thinking": {"type": thinking}}
        if thinking == "enabled" and not strict_max_tokens:
            request["max_tokens"] = max(int(max_tokens), 600)
    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content or ""
    if str(content).strip():
        return str(content).strip()
    if model.startswith("deepseek-v4") and thinking == "enabled":
        retry_request = dict(request)
        retry_request["extra_body"] = {"thinking": {"type": "disabled"}}
        retry_request["max_tokens"] = int(max_tokens) if strict_max_tokens else max(int(max_tokens), 128)
        retry = client.chat.completions.create(**retry_request)
        retry_content = retry.choices[0].message.content or ""
        return str(retry_content).strip()
    return ""


def _stable_rng(text: str) -> random.Random:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def mock_llm(system: str, user: str, n_options: int | None = None) -> str:
    del system
    user_lower = user.lower()
    persona_part = user.split("QUESTION:")[0]
    question_part = user.split("QUESTION:")[-1]
    persona_rng = _stable_rng(persona_part)
    question_rng = _stable_rng(persona_part + "\n" + question_part)
    if "feminist movement" in user_lower or "女权" in user or "女性主义" in user:
        return str(question_rng.choice([2, 2, 3]))
    if "would you buy" in user_lower or "会购买" in user or "是否购买" in user:
        match = re.search(r"at \$([0-9]+(?:\.[0-9]+)?)", user)
        if not match:
            match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)", user)
        price = float(match.group(1)) if match else 5.0
        willingness_to_pay = 3.0 + 10.0 * (persona_rng.randint(0, 999) / 999.0)
        return "2" if price <= willingness_to_pay else "1"
    if n_options:
        middle = (n_options + 1) // 2
        jitter = question_rng.choice([-1, 0, 0, 1])
        return str(min(max(1, middle + jitter), n_options))
    return "1"


def parse_option_number(raw: str, n_options: int | None = None) -> int | None:
    match = re.search(r"-?\d+", raw or "")
    if not match:
        return None
    value = int(match.group())
    if n_options:
        value = min(max(value, 1), n_options)
    return value


def simulate_answer(
    persona_text: str,
    question_text: str,
    options: list[str],
    *,
    api_key: str = "",
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-v4-flash",
    max_tokens: int = 5,
    thinking: str = "disabled",
    strict_max_tokens: bool = False,
) -> tuple[int | None, str]:
    user = build_user_prompt(persona_text, question_text, options)
    if api_key.strip():
        raw = call_llm(
            SYSTEM_MESSAGE,
            user,
            api_key.strip(),
            base_url,
            model,
            max_tokens=max_tokens,
            thinking=thinking,
            strict_max_tokens=strict_max_tokens,
        )
    else:
        raw = mock_llm(SYSTEM_MESSAGE, user, n_options=len(options) if options else None)
    return parse_option_number(raw, len(options) if options else None), raw


def simulate_answer_for_row(row, question_text: str, options: list[str], variant: str = "full_or_summary", **kwargs) -> tuple[int | None, str]:
    return simulate_answer(get_persona_variant(row, variant), question_text, options, **kwargs)


def _answer_position(question: dict | None) -> int | None:
    if not question:
        return None
    pos = question.get("Answers", {}).get("SelectedByPosition")
    if isinstance(pos, list):
        return pos[0] if len(pos) == 1 else None
    return pos if isinstance(pos, int) else None


def iter_answer_questions(answer_block_json: str) -> Iterable[dict]:
    for block in json.loads(answer_block_json):
        for question in block.get("Questions", []):
            yield question


def find_question_by_id(answer_block_json: str, qid: str) -> dict | None:
    for question in iter_answer_questions(answer_block_json):
        if question.get("QuestionID") == qid:
            return question
    return None


def choose_eval_question(rows: list[dict], preferred_qids: tuple[str, ...] = ("QID291", "QID196")) -> tuple[str, str, list[str]]:
    common = None
    examples = {}
    for row in rows:
        gt_ids, rt_ids = set(), set()
        for source, target in [(row["ground_truth"], gt_ids), (row["retest"], rt_ids)]:
            for question in iter_answer_questions(source):
                options = question.get("Options") or []
                if question.get("QuestionType") == "MC" and 2 <= len(options) <= 7 and _answer_position(question) is not None:
                    target.add(question.get("QuestionID"))
                    examples.setdefault(question.get("QuestionID"), question)
        ids = gt_ids & rt_ids
        common = ids if common is None else common & ids
    common = common or set()
    for qid in preferred_qids:
        if qid in common:
            question = examples[qid]
            return qid, _strip(question.get("QuestionText")), question.get("Options")
    for qid in sorted(common):
        if not str(qid).startswith("QID9_"):
            question = examples[qid]
            return qid, _strip(question.get("QuestionText")), question.get("Options")
    if common:
        qid = sorted(common)[0]
        question = examples[qid]
        return qid, _strip(question.get("QuestionText")), question.get("Options")
    return "SYN_LINDA", LINDA_TEXT, LINDA_OPTS


def accuracy(pred: int | None, truth: int | None, lo: int, hi: int) -> float:
    if pred is None or truth is None or hi <= lo:
        return np.nan
    return 1.0 - abs(pred - truth) / (hi - lo)


def single_item_evaluation(rows: list[dict], qid: str, question_text: str, options: list[str], variant: str = "full_or_summary", **kwargs) -> pd.DataFrame:
    lo, hi = 1, len(options)
    records = []
    for row in rows:
        gt_q = find_question_by_id(row["ground_truth"], qid)
        rt_q = find_question_by_id(row["retest"], qid)
        truth = _answer_position(gt_q) if gt_q else None
        retest = _answer_position(rt_q) if rt_q else None
        twin, raw = simulate_answer_for_row(row, question_text, options, variant=variant, **kwargs)
        records.append(
            {
                "pid": row["pid"],
                "truth": truth,
                "retest": retest,
                "twin": twin,
                "twin_label": options[twin - 1] if twin else None,
                "raw_twin": raw,
                "acc_twin": accuracy(twin, truth, lo, hi),
                "acc_retest": accuracy(retest, truth, lo, hi),
            }
        )
    return pd.DataFrame(records)


def candidate_eval_items(rows: list[dict], max_items: int = 2, include_product_items: bool = False) -> list[dict]:
    common = None
    examples = {}
    for row in rows:
        gt_ids, rt_ids = set(), set()
        for source, target in [(row["ground_truth"], gt_ids), (row["retest"], rt_ids)]:
            for question in iter_answer_questions(source):
                options = question.get("Options") or []
                qid = question.get("QuestionID")
                if question.get("QuestionType") == "MC" and 2 <= len(options) <= 7 and _answer_position(question) is not None:
                    if include_product_items or not str(qid).startswith("QID9_"):
                        target.add(qid)
                        examples.setdefault(qid, question)
        ids = gt_ids & rt_ids
        common = ids if common is None else common & ids
    items = []
    for qid in sorted(common or set())[:max_items]:
        question = examples[qid]
        options = question.get("Options") or []
        items.append({"qid": qid, "text": _strip(question.get("QuestionText")), "options": options, "lo": 1, "hi": len(options)})
    return items or [{"qid": "SYN_LINDA", "text": LINDA_TEXT, "options": LINDA_OPTS, "lo": 1, "hi": len(LINDA_OPTS)}]


def scale_value(x, lo: int, hi: int) -> float:
    if x is None or pd.isna(x) or hi <= lo:
        return np.nan
    return (float(x) - lo) / (hi - lo)


def safe_corr(a, b) -> float:
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if len(a) < 2 or a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_items(
    rows: list[dict],
    items: list[dict],
    variant: str = "full_or_summary",
    *,
    random_seed: int = 0,
    api_sleep: float = 0.0,
    **kwargs,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    records = []
    for item in items:
        options = item["options"]
        for row in rows:
            gt_q = find_question_by_id(row["ground_truth"], item["qid"])
            rt_q = find_question_by_id(row["retest"], item["qid"])
            truth = _answer_position(gt_q) if gt_q else None
            retest = _answer_position(rt_q) if rt_q else None
            twin, raw = simulate_answer_for_row(row, item["text"], options, variant=variant, **kwargs)
            records.append(
                {
                    "variant": variant,
                    "pid": row["pid"],
                    "qid": item["qid"],
                    "lo": item["lo"],
                    "hi": item["hi"],
                    "truth": truth,
                    "twin": twin,
                    "retest": retest,
                    "random": int(rng.integers(1, len(options) + 1)),
                    "raw_twin": raw,
                }
            )
            if api_sleep and kwargs.get("api_key", "").strip():
                time.sleep(api_sleep)
    return pd.DataFrame(records)


def summarize_prediction_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor in ["twin", "retest", "random"]:
        sub = wide_df.dropna(subset=["truth", predictor]).copy()
        if sub.empty:
            continue
        truth_s = sub.apply(lambda row: scale_value(row["truth"], row["lo"], row["hi"]), axis=1)
        pred_s = sub.apply(lambda row: scale_value(row[predictor], row["lo"], row["hi"]), axis=1)
        acc = 1 - (pred_s - truth_s).abs()
        truth_sd = truth_s.std(ddof=0)
        truth_var = truth_s.var(ddof=0)
        pred_var = pred_s.var(ddof=0)
        rows.append(
            {
                "variant": sub["variant"].iloc[0],
                "predictor": predictor,
                "n_obs": len(sub),
                "n_items": sub["qid"].nunique(),
                "accuracy": acc.mean(),
                "correlation": safe_corr(pred_s, truth_s),
                "mean_diff_sd": (pred_s.mean() - truth_s.mean()) / truth_sd if truth_sd > 0 else np.nan,
                "variance_ratio": pred_var / truth_var if truth_var > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def simulate_purchase(persona_text: str, product: str, price: float, **kwargs) -> int:
    question = f"Would you buy the following product at ${price:.2f}? Product: {product}."
    value, _ = simulate_answer(persona_text, question, ["No", "Yes"], **kwargs)
    return 1 if value == 2 else 0


def demand_curve(
    rows: list[dict],
    product: str,
    regular_price: float,
    price_ratios: list[float],
    *,
    variant: str = "full_or_summary",
    api_sleep: float = 0.0,
    **kwargs,
) -> pd.DataFrame:
    persona_texts = [get_persona_variant(row, variant) for row in rows]
    records = []
    for ratio in price_ratios:
        price = ratio * regular_price
        buys = [simulate_purchase(text, product, price, **kwargs) for text in persona_texts]
        records.append({"price_ratio": ratio, "price_$": round(price, 2), "P(buy)": float(np.mean(buys))})
        if api_sleep and kwargs.get("api_key", "").strip():
            time.sleep(api_sleep)
    return pd.DataFrame(records)

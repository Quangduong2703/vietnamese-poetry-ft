from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path
from urllib.request import urlretrieve

from poetry_pipeline.settings import DATA_DIR, ensure_runtime_dirs


DATASET_URL = (
    "https://huggingface.co/datasets/phamson02/vietnamese-poetry-corpus/"
    "resolve/main/poems_dataset.csv?download=true"
)
CSV_PATH = DATA_DIR / "poems_dataset.csv"
JSONL_PATH = DATA_DIR / "poems_dataset_normalized.jsonl"
MINIMAL_JSONL_PATH = DATA_DIR / "train_minimal.jsonl"
CREATIVE_JSONL_PATH = DATA_DIR / "train_creative.jsonl"
CREATIVE_TRAIN_PATH = DATA_DIR / "train_creative_train.jsonl"
CREATIVE_VALID_PATH = DATA_DIR / "train_creative_valid.jsonl"
LUC_BAT_CREATIVE_PATH = DATA_DIR / "train_creative_luc_bat.jsonl"
LUC_BAT_TRAIN_PATH = DATA_DIR / "train_creative_luc_bat_train.jsonl"
LUC_BAT_VALID_PATH = DATA_DIR / "train_creative_luc_bat_valid.jsonl"
TRAIN_TEXT_PATH = DATA_DIR / "poems_dataset_train.txt"

SPECIAL_INSTRUCTION = "<|instruction|>"
SPECIAL_POEM = "<|poem|>"
SPECIAL_END = "<|end|>"

INLINE_SPACE_RE = re.compile(r"[ \t\u00A0]+")
LINE_BREAK_MARKER_RE = re.compile(r"\s*<\s*\n\s*>\s*")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")


def download_dataset() -> None:
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        print(f"Using existing CSV: {CSV_PATH}")
        return

    print(f"Downloading dataset to {CSV_PATH} ...")
    urlretrieve(DATASET_URL, CSV_PATH)
    print("Download complete.")


def normalize_unicode(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    return text


def normalize_inline_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""

    text = normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", " ")
    text = INLINE_SPACE_RE.sub(" ", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return text.strip()


def normalize_poem_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""

    text = normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = LINE_BREAK_MARKER_RE.sub("\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = MULTI_BLANK_LINES_RE.sub("\n\n", text)

    normalized_lines: list[str] = []
    blank_pending = False

    for raw_line in text.split("\n"):
        line = INLINE_SPACE_RE.sub(" ", raw_line)
        line = SPACE_BEFORE_PUNCT_RE.sub(r"\1", line)
        line = line.strip()

        if not line:
            if normalized_lines and not blank_pending:
                normalized_lines.append("")
                blank_pending = True
            continue

        normalized_lines.append(line)
        blank_pending = False

    while normalized_lines and normalized_lines[0] == "":
        normalized_lines.pop(0)
    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines)


def build_form_phrase(form: str) -> str:
    form = normalize_inline_text(form)
    if not form:
        return "một bài thơ"
    if form.lower().startswith("thơ "):
        return f"một bài {form}"
    return f"một bài thơ {form}"


def build_instruction(title: str, genre: str, specific_genre: str) -> str:
    form = normalize_inline_text(specific_genre) or normalize_inline_text(genre)
    form_phrase = build_form_phrase(form)
    title = normalize_inline_text(title)

    if title:
        return f'Hãy viết {form_phrase} có tiêu đề "{title}".'
    return f"Hãy viết {form_phrase}."


def build_train_text(instruction: str, poem: str) -> str:
    return (
        f"{SPECIAL_INSTRUCTION}\n"
        f"{instruction}\n"
        f"{SPECIAL_POEM}\n"
        f"{poem}\n"
        f"{SPECIAL_END}\n"
    )


def normalize_for_match(text: str) -> str:
    text = normalize_inline_text(text).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return INLINE_SPACE_RE.sub(" ", text).strip()


def infer_topic_from_title(title: str) -> str:
    title = normalize_inline_text(title)
    if not title:
        return ""

    normalized_title = normalize_for_match(title)
    token_set = set(normalized_title.split())

    def contains_all(*tokens: str) -> bool:
        return all(token in token_set for token in tokens)

    phrase_rules = [
        (("nhớ", "quê"), "nỗi nhớ quê hương"),
        (("nhớ", "mẹ"), "nỗi nhớ mẹ"),
        (("nhớ", "em"), "nỗi nhớ trong tình yêu"),
        (("nhớ", "anh"), "nỗi nhớ trong tình yêu"),
        (("quê", "hương"), "quê hương"),
        (("mùa", "thu"), "mùa thu"),
        (("mùa", "xuân"), "mùa xuân"),
        (("mùa", "đông"), "mùa đông"),
        (("mùa", "hạ"), "mùa hè"),
        (("hoa", "cúc"), "hoa cúc và thiên nhiên"),
        (("hoa", "sen"), "hoa sen và thiên nhiên"),
        (("mưa", "đêm"), "đêm mưa và tâm trạng"),
    ]
    for tokens, topic in phrase_rules:
        if contains_all(*tokens):
            return topic

    keyword_rules = [
        ("mẹ và tình mẫu tử", {"mẹ", "má", "u"}),
        ("cha và gia đình", {"cha", "bố", "ba"}),
        ("tình yêu", {"yêu", "thương", "duyên", "tương", "tư"}),
        ("quê hương", {"quê", "làng", "xóm", "thôn"}),
        ("mùa xuân", {"xuân", "tết"}),
        ("mùa thu", {"thu"}),
        ("mùa đông", {"đông"}),
        ("mùa hè", {"hạ", "hè"}),
        ("đêm và trăng", {"đêm", "trăng", "nguyệt", "khuya"}),
        ("mưa gió và tâm trạng", {"mưa", "gió", "bão"}),
        ("sông nước và biển", {"sông", "biển", "hồ", "nước", "thuyền", "bến", "suối"}),
        ("hoa cỏ thiên nhiên", {"hoa", "lá", "đào", "mai", "cúc", "sen", "cỏ"}),
        ("nỗi nhớ và hoài niệm", {"nhớ", "hoài", "niệm", "xưa"}),
        ("buồn và cô đơn", {"buồn", "sầu", "lệ", "khóc"}),
        ("đời sống và nhân sinh", {"đời", "người", "thân", "phận", "kiếp"}),
    ]
    for topic, keywords in keyword_rules:
        if token_set.intersection(keywords):
            return topic

    return ""


def pick_template(seed_text: str, templates: list[str]) -> str:
    digest = hashlib.sha1(seed_text.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % len(templates)
    return templates[index]


def build_creative_instruction(
    title: str,
    genre: str,
    specific_genre: str,
    period: str,
    row_id: int,
    topic: str,
) -> str:
    form = normalize_inline_text(specific_genre) or normalize_inline_text(genre)
    form_phrase = build_form_phrase(form)
    title = normalize_inline_text(title)
    period = normalize_inline_text(period)
    topic = normalize_inline_text(topic)

    seed_parts = [str(row_id), title, form, period, topic]
    seed_text = "|".join(seed_parts)

    if topic and title and period:
        templates = [
            'Hãy sáng tác {form_phrase} về {topic}, lấy cảm hứng từ nhan đề "{title}" và gợi không khí {period}.',
            'Viết {form_phrase} xoay quanh {topic}, dựa trên cảm hứng từ "{title}" và phảng phất sắc thái {period}.',
            'Làm {form_phrase} về {topic}, với điểm tựa là nhan đề "{title}" và dư vị của thời {period}.',
            'Sáng tác {form_phrase} mang chủ đề {topic}, gợi từ "{title}" và hơi thở {period}.',
        ]
        return pick_template(seed_text, templates).format(
            form_phrase=form_phrase,
            title=title,
            period=period,
            topic=topic,
        )

    if topic and title:
        templates = [
            'Hãy sáng tác {form_phrase} về {topic}, lấy cảm hứng từ nhan đề "{title}".',
            'Viết {form_phrase} xoay quanh {topic}, dựa trên cảm hứng từ "{title}".',
            'Làm {form_phrase} về {topic}, với điểm tựa cảm xúc là "{title}".',
            'Sáng tác {form_phrase} mang chủ đề {topic}, gợi từ nhan đề "{title}".',
        ]
        return pick_template(seed_text, templates).format(
            form_phrase=form_phrase,
            title=title,
            topic=topic,
        )

    if topic and period:
        templates = [
            'Hãy sáng tác {form_phrase} về {topic}, gợi không khí {period}.',
            'Viết {form_phrase} xoay quanh {topic}, với cảm hứng phảng phất từ thời {period}.',
            'Làm {form_phrase} về {topic}, giữ dư vị thơ ca của giai đoạn {period}.',
            'Sáng tác {form_phrase} mang chủ đề {topic}, với sắc thái gợi nhớ thời {period}.',
        ]
        return pick_template(seed_text, templates).format(
            form_phrase=form_phrase,
            period=period,
            topic=topic,
        )

    if topic:
        templates = [
            'Hãy sáng tác {form_phrase} về {topic}.',
            'Viết {form_phrase} xoay quanh chủ đề {topic}.',
            'Làm {form_phrase} lấy cảm hứng từ {topic}.',
            'Sáng tác {form_phrase} mang mạch cảm xúc về {topic}.',
            'Hãy viết {form_phrase} với hình ảnh gợi về {topic}.',
        ]
        return pick_template(seed_text, templates).format(
            form_phrase=form_phrase,
            topic=topic,
        )

    if title and period:
        templates = [
            'Hãy sáng tác {form_phrase} lấy cảm hứng từ nhan đề "{title}", gợi không khí {period}.',
            'Viết {form_phrase} xoay quanh "{title}", với cảm hứng phảng phất từ thời {period}.',
            'Làm {form_phrase} dựa trên nhan đề "{title}", giữ dư vị thơ ca của giai đoạn {period}.',
            'Sáng tác {form_phrase} từ cảm hứng "{title}", mang sắc thái gợi nhớ thời {period}.',
        ]
        return pick_template(seed_text, templates).format(
            form_phrase=form_phrase,
            title=title,
            period=period,
        )

    if title:
        templates = [
            'Hãy sáng tác {form_phrase} lấy cảm hứng từ nhan đề "{title}".',
            'Viết {form_phrase} xoay quanh cảm hứng "{title}".',
            'Làm {form_phrase} gợi ra thế giới của nhan đề "{title}".',
            'Sáng tác {form_phrase} dựa trên cảm hứng từ "{title}".',
            'Hãy viết {form_phrase} với điểm tựa cảm xúc là "{title}".',
            'Viết {form_phrase} theo hướng giàu hình ảnh, lấy "{title}" làm mạch cảm hứng.',
        ]
        return pick_template(seed_text, templates).format(
            form_phrase=form_phrase,
            title=title,
        )

    if period:
        templates = [
            'Hãy sáng tác {form_phrase} với cảm hứng gợi nhớ giai đoạn {period}.',
            'Viết {form_phrase} mang không khí thơ ca của thời {period}.',
            'Làm {form_phrase} theo mạch cảm xúc phảng phất dấu ấn {period}.',
            'Sáng tác {form_phrase} với sắc thái gợi nhắc đến thời {period}.',
        ]
        return pick_template(seed_text, templates).format(
            form_phrase=form_phrase,
            period=period,
        )

    templates = [
        'Hãy sáng tác {form_phrase} với hình ảnh và liên tưởng giàu chất thơ.',
        'Viết {form_phrase} theo hướng giàu cảm xúc và hình ảnh.',
        'Làm {form_phrase} với giọng điệu tự nhiên và giàu liên tưởng.',
        'Sáng tác {form_phrase} sao cho mạch thơ mềm và giàu hình ảnh.',
    ]
    return pick_template(seed_text, templates).format(form_phrase=form_phrase)


def assign_split(text: str, valid_percent: int = 5) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "valid" if bucket < valid_percent else "train"


def process_dataset() -> tuple[int, int]:
    processed = 0
    skipped = 0

    with CSV_PATH.open("r", encoding="utf-8", newline="") as csv_file, JSONL_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as jsonl_file, MINIMAL_JSONL_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as minimal_jsonl_file, CREATIVE_JSONL_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as creative_jsonl_file, CREATIVE_TRAIN_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as creative_train_file, CREATIVE_VALID_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as creative_valid_file, LUC_BAT_CREATIVE_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as luc_bat_creative_file, LUC_BAT_TRAIN_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as luc_bat_train_file, LUC_BAT_VALID_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as luc_bat_valid_file, TRAIN_TEXT_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as train_file:
        reader = csv.DictReader(csv_file)

        for row_id, row in enumerate(reader, start=1):
            poem = normalize_poem_text(row.get("content"))
            if not poem:
                skipped += 1
                continue

            title = normalize_inline_text(row.get("title"))
            genre = normalize_inline_text(row.get("genre"))
            specific_genre = normalize_inline_text(row.get("specific_genre"))
            period = normalize_inline_text(row.get("period"))
            form = specific_genre or genre
            topic = infer_topic_from_title(title)
            instruction = build_instruction(title, genre, specific_genre)
            creative_instruction = build_creative_instruction(
                title=title,
                genre=genre,
                specific_genre=specific_genre,
                period=period,
                row_id=row_id,
                topic=topic,
            )

            record = {
                "id": row_id,
                "instruction": instruction,
                "poem": poem,
                "title": title,
                "genre": genre,
                "specific_genre": specific_genre,
                "period": period,
                "author": normalize_inline_text(row.get("author")),
                "source_url": normalize_inline_text(row.get("url")),
            }
            minimal_record = {
                "instruction": instruction,
                "poem": poem,
                "form": form,
                "title": title,
                "genre": genre,
                "specific_genre": specific_genre,
                "period": period,
            }
            creative_record = {
                "instruction": creative_instruction,
                "poem": poem,
                "form": form,
                "topic": topic,
                "title": title,
                "genre": genre,
                "specific_genre": specific_genre,
                "period": period,
            }
            split_name = assign_split(poem)

            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            minimal_jsonl_file.write(json.dumps(minimal_record, ensure_ascii=False) + "\n")
            creative_jsonl_file.write(json.dumps(creative_record, ensure_ascii=False) + "\n")
            if split_name == "valid":
                creative_valid_file.write(json.dumps(creative_record, ensure_ascii=False) + "\n")
            else:
                creative_train_file.write(json.dumps(creative_record, ensure_ascii=False) + "\n")
            if form.lower() == "lục bát":
                luc_bat_creative_file.write(json.dumps(creative_record, ensure_ascii=False) + "\n")
                if split_name == "valid":
                    luc_bat_valid_file.write(json.dumps(creative_record, ensure_ascii=False) + "\n")
                else:
                    luc_bat_train_file.write(json.dumps(creative_record, ensure_ascii=False) + "\n")
            train_file.write(build_train_text(instruction, poem))
            processed += 1

    return processed, skipped


def main() -> None:
    ensure_runtime_dirs()
    download_dataset()
    processed, skipped = process_dataset()
    print(f"Processed records: {processed}")
    print(f"Skipped records:   {skipped}")
    print(f"JSONL output:      {JSONL_PATH}")
    print(f"Minimal JSONL:     {MINIMAL_JSONL_PATH}")
    print(f"Creative JSONL:    {CREATIVE_JSONL_PATH}")
    print(f"Creative train:    {CREATIVE_TRAIN_PATH}")
    print(f"Creative valid:    {CREATIVE_VALID_PATH}")
    print(f"Luc bat creative:  {LUC_BAT_CREATIVE_PATH}")
    print(f"Luc bat train:     {LUC_BAT_TRAIN_PATH}")
    print(f"Luc bat valid:     {LUC_BAT_VALID_PATH}")
    print(f"Train text output: {TRAIN_TEXT_PATH}")


if __name__ == "__main__":
    main()

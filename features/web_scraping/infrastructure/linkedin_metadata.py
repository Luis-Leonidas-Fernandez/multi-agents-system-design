"""Metadata inference helpers for LinkedIn job detail bodies."""
from __future__ import annotations

import re


def _sanitize_description(value: str, limit: int = 1000) -> str:
    normalized = re.sub(r"\s+", " ", value or "").strip()
    normalized = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[redacted-email]",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"https?://\S+", "[redacted-url]", normalized)
    normalized = re.sub(
        r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
        "[redacted-phone]",
        normalized,
    )
    return normalized[:limit].strip()


def _truncate_at_word(value: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", value or "").strip()
    if len(normalized) <= limit:
        return normalized
    cutoff = normalized[:limit].rsplit(" ", 1)[0].strip()
    return cutoff or normalized[:limit].strip()


def _sanitize_structured_item(value: str, limit: int = 180) -> str:
    sanitized = _sanitize_description(value, limit=max(limit * 3, limit))
    return _truncate_at_word(sanitized, limit)

_SECTION_PATTERNS = {
    "candidate_expectations": (
        r"requirements?",
        r"qualifications?",
        r"required qualifications?",
        r"minimum qualifications?",
        r"required skills?(?: and experience)?",
        r"what you(?:'|’)ll need(?: to succeed)?",
        r"what you need to succeed",
        r"what we need to see",
        r"who you are",
        r"who are we looking for\??",
        r"we are looking for",
        r"what you bring",
        r"応募資格",
        r"必須要件",
        r"歓迎要件",
        r"求める人物像",
        r"必要な経験",
        r"자격요건",
        r"지원자격",
        r"필수요건",
        r"우대사항",
    ),
    "responsibilities": (
        r"responsibilities",
        r"key responsibilities",
        r"role responsibilities",
        r"duties(?: and responsibilities)?",
        r"tasks",
        r"what you(?:'|’)ll do",
        r"what you will do",
        r"what you(?:'|’)ll be doing",
        r"in this role,? you(?:'|’)ll get to",
        r"in this role",
        r"業務内容",
        r"仕事内容",
        r"従事すべき業務の内容",
        r"担当業務",
        r"담당업무",
        r"주요업무",
        r"주요 업무",
    ),
}
_OTHER_SECTION_PATTERN = re.compile(
    r"^(?:about(?: us| the role)?|about .+|benefits?|preferred qualifications?|"
    r"nice to have|ways to stand out(?: from the crowd)?|company|company information|"
    r"compensation(?: & benefits)?|salary|location|working conditions?|work hours?|"
    r"holidays?(?: and leave)?|probation period|insurance|additional information|"
    r"discover more.*|equal opportunity employer|disclaimer|job type|"
    r"待遇|福利厚生|会社概要|勤務地|勤務時間|休日|給与|雇用形態|회사소개|혜택|복리후생)\s*:?\s*$",
    re.IGNORECASE,
)


def _clean_section_heading_candidate(line: str) -> str:
    cleaned = re.sub(r"^[#>*•●▪◦\-\s]+", "", line or "").strip()
    cleaned = cleaned.strip("[](){}【】「」『』〈〉《》")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _section_heading(line: str) -> tuple[str, str] | None:
    cleaned = _clean_section_heading_candidate(line)
    for kind, patterns in _SECTION_PATTERNS.items():
        for pattern in patterns:
            match = re.match(
                rf"^(?:{pattern})\s*:?\s*(.*)$",
                cleaned,
                re.IGNORECASE,
            )
            if match:
                return kind, match.group(1).strip()
    return None


def _bounded_unique_items(
    values: list[str],
    *,
    limit: int = 6,
    item_limit: int = 180,
) -> list[str]:
    bounded: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = re.sub(r"^[#>*•●▪◦\-\s]+", "", value or "").strip()
        item = re.sub(r"^\d+[.)]\s+", "", item).strip()
        item = _sanitize_structured_item(item, limit=item_limit)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        bounded.append(item)
        if len(bounded) >= limit:
            break
    return bounded


def _extract_structured_sections(text: str) -> tuple[list[str], list[str]]:
    sections: dict[str, list[str]] = {
        "candidate_expectations": [],
        "responsibilities": [],
    }
    active = ""
    for line in (text or "").splitlines():
        heading = _section_heading(line)
        if heading:
            active, inline_value = heading
            if inline_value:
                sections[active].append(inline_value)
            continue
        if _OTHER_SECTION_PATTERN.match(line.strip()):
            active = ""
            continue
        if active and line.strip():
            sections[active].append(line)
    return (
        _bounded_unique_items(sections["candidate_expectations"]),
        _bounded_unique_items(sections["responsibilities"]),
    )


_HARD_SKILL_PATTERNS = (
    ("Python", r"\bpython\b"),
    ("SQL", r"\bsql\b"),
    ("R", r"(?<![\w.])R(?![\w.])"),
    ("Java", r"\bjava\b"),
    ("C++", r"(?<!\w)c\+\+(?!\w)"),
    ("PyTorch", r"\bpytorch\b"),
    ("TensorFlow", r"\btensorflow\b"),
    ("scikit-learn", r"\b(?:scikit-learn|sklearn)\b"),
    ("Spark", r"\b(?:apache\s+)?spark\b"),
    ("Databricks", r"\bdatabricks\b"),
    ("AWS", r"\b(?:aws|amazon web services)\b"),
    ("GCP", r"\b(?:gcp|google cloud(?: platform)?)\b"),
    ("Azure", r"\b(?:microsoft\s+)?azure\b"),
    ("Docker", r"\bdocker\b"),
    ("Kubernetes", r"\b(?:kubernetes|k8s)\b"),
    ("Git", r"\bgit\b"),
    ("MLOps", r"\bmlops\b"),
    ("LLM", r"\b(?:llms?|large language models?)\b"),
    ("RAG", r"\b(?:rag|retrieval[- ]augmented generation)\b"),
    ("NLP", r"\b(?:nlp|natural language processing)\b"),
    ("Computer Vision", r"\bcomputer vision\b|コンピュータビジョン|컴퓨터 비전"),
    ("Data pipelines", r"\bdata pipelines?\b|データパイプライン|데이터 파이프라인"),
    (
        "Vector DB",
        r"\b(?:vector (?:db|database)s?|pinecone|weaviate|milvus|pgvector)\b",
    ),
    ("Power BI", r"\bpower\s*bi\b"),
    ("Tableau", r"\btableau\b"),
    ("Looker", r"\blooker\b"),
    ("BI", r"\bbusiness intelligence\b"),
    ("pandas", r"\bpandas\b"),
    ("NumPy", r"\bnumpy\b"),
    ("Hugging Face", r"\bhugging\s*face\b"),
    ("LangChain", r"\blangchain\b"),
    ("MLflow", r"\bmlflow\b"),
    ("Airflow", r"\b(?:apache\s+)?airflow\b"),
    ("Kafka", r"\b(?:apache\s+)?kafka\b"),
    ("Snowflake", r"\bsnowflake\b"),
)


def _infer_hard_skills(text: str) -> list[str]:
    return [
        label
        for label, pattern in _HARD_SKILL_PATTERNS
        if re.search(pattern, text or "", re.IGNORECASE)
    ][:40]


_SOFT_SKILL_PATTERNS = (
    ("Communication", r"\bcommunication skills?\b|コミュニケーション(?:能力|スキル)|의사소통(?: 능력)?"),
    (
        "Collaboration",
        r"\b(?:collaboration|collaborative|teamwork)\b|"
        r"協調性|チームワーク|협업|팀워크",
    ),
    (
        "Problem solving",
        r"\bproblem[- ]solving\b|問題解決(?:能力)?|문제 해결(?: 능력)?",
    ),
    ("Ownership", r"\bownership\b|オーナーシップ|주인의식"),
    ("Leadership", r"\bleadership\b|リーダーシップ|리더십"),
    ("Adaptability", r"\badaptability\b|\badaptable\b|適応力|적응력"),
    (
        "Critical thinking",
        r"\bcritical thinking\b|批判的思考|クリティカルシンキング|비판적 사고",
    ),
    (
        "Stakeholder management",
        r"\bstakeholder(?: management)?\b|ステークホルダー|이해관계자",
    ),
    (
        "Cross-functional collaboration",
        r"\bcross[- ]functional\b|部門横断|クロスファンクショナル|부서 간 협업",
    ),
)


def _infer_soft_skills(text: str) -> list[str]:
    return [
        label
        for label, pattern in _SOFT_SKILL_PATTERNS
        if re.search(pattern, text or "", re.IGNORECASE)
    ][:20]


def _infer_explicit_status(
    text: str,
    *,
    positive_patterns: tuple[str, ...],
    negative_patterns: tuple[str, ...],
    mention_pattern: str,
    positive_value: str,
    negative_value: str,
) -> str:
    positive = any(re.search(pattern, text, re.IGNORECASE) for pattern in positive_patterns)
    negative = any(re.search(pattern, text, re.IGNORECASE) for pattern in negative_patterns)
    if positive and negative:
        return "ambiguous"
    if positive:
        return positive_value
    if negative:
        return negative_value
    if re.search(mention_pattern, text, re.IGNORECASE):
        return "ambiguous"
    return "unknown"


def _infer_visa_status(text: str) -> str:
    return _infer_explicit_status(
        text,
        positive_patterns=(
            r"\b(?:full\s+)?visa sponsorship\b.{0,80}\b(?:available|provided|offered)\b",
            r"\bvisa sponsorship (?:is )?(?:available|provided|offered)\b",
            r"\b(?:offer|provide) visa sponsorship\b",
            r"\b(?:we|company|employer) (?:can |will )?sponsor\b.{0,30}\bvisa\b",
            r"\bvisa (?:support|sponsorship) (?:available|provided|offered)\b",
            r"ビザ(?:サポート|支援)(?:あり|可能|提供)",
            r"(?:就労|勤務)ビザ.{0,20}(?:支援|サポート|取得可能)",
            r"ビザスポンサー(?:可能|あり|提供)",
            r"비자\s*(?:지원|스폰서십)(?:\s*(?:가능|제공))",
            r"(?:취업|근무)\s*비자.{0,20}(?:지원|발급 가능)",
        ),
        negative_patterns=(
            r"\bno visa sponsorship\b",
            r"\b(?:unable|cannot|can't|do not|does not|won't) sponsor\b",
            r"\bsponsorship (?:is )?not available\b",
            r"\bwithout (?:visa )?sponsorship\b",
            r"ビザ(?:サポート|支援)(?:不可|なし)",
            r"ビザスポンサー(?:不可|なし)",
            r"(?:就労|勤務)ビザ.{0,20}自己負担",
            r"비자\s*(?:지원|스폰서십)\s*(?:불가|없음)",
            r"(?:취업|근무)\s*비자.{0,20}(?:지원 불가|본인 부담)",
        ),
        mention_pattern=r"\bvisa\b|sponsorship|ビザ|비자",
        positive_value="sponsorship",
        negative_value="no_sponsorship",
    )


def _infer_relocation_support(text: str) -> str:
    return _infer_explicit_status(
        text,
        positive_patterns=(
            r"\brelocation (?:assistance|support|package)\b.{0,80}\b(?:available|provided|offered)\b",
            r"\brelocation (?:assistance|support|package) "
            r"(?:is )?(?:available|provided|offered)\b",
            r"\bwe (?:can |will )?(?:support|assist with) relocation\b",
            r"転居(?:支援|サポート)(?:あり|可能|提供)",
            r"引越し(?:支援|サポート|費用)(?:あり|可能|提供)",
            r"이주\s*(?:지원|패키지)(?:\s*(?:가능|제공))",
            r"이전\s*(?:지원|비용)(?:\s*(?:가능|제공))",
        ),
        negative_patterns=(
            r"\bno relocation (?:assistance|support|package)?\b",
            r"\brelocation (?:is )?not (?:available|provided|offered)\b",
            r"転居(?:支援|サポート)(?:不可|なし)",
            r"引越し(?:支援|サポート)(?:不可|なし)",
            r"이주\s*(?:지원|패키지)\s*(?:불가|없음)",
            r"이전\s*(?:지원|비용)\s*(?:불가|없음)",
        ),
        mention_pattern=r"\brelocation\b|転居|이주",
        positive_value="yes",
        negative_value="no",
    )


def _infer_foreigner_acceptance(text: str) -> str:
    return _infer_explicit_status(
        text,
        positive_patterns=(
            r"\b(?:foreign|overseas|international) "
            r"(?:applicants?|candidates?) (?:are )?(?:welcome|accepted|eligible)\b",
            r"\binternational applications?\b.{0,80}\b(?:welcome|accepted|eligible)\b",
            r"\bwe welcome both local and international applications?\b",
            r"\bapplications? from abroad (?:are )?(?:welcome|accepted)\b",
            r"外国人(?:応募者|候補者)?(?:歓迎|応募可)",
            r"海外(?:在住者|応募者)(?:歓迎|応募可)",
            r"国籍不問",
            r"외국인\s*(?:지원자)?\s*(?:환영|지원 가능)",
            r"해외\s*(?:거주자|지원자)\s*(?:환영|지원 가능)",
            r"국적\s*무관",
        ),
        negative_patterns=(
            r"\b(?:foreign|overseas|international) "
            r"(?:applicants?|candidates?) (?:are )?not (?:accepted|eligible)\b",
            r"\bdomestic applicants? only\b",
            r"\bmust already (?:reside|be based) in\b",
            r"外国人(?:応募者|候補者)?(?:不可|対象外)",
            r"海外(?:在住者|応募者)(?:不可|対象外)",
            r"日本国内在住者(?:のみ|限定)",
            r"외국인\s*(?:지원자)?\s*(?:불가|대상 아님)",
            r"해외\s*(?:거주자|지원자)\s*(?:불가|대상 아님)",
            r"국내\s*거주자만",
        ),
        mention_pattern=(
            r"\b(?:foreign|overseas|international) (?:applicants?|candidates?)\b"
            r"|外国人|海外(?:在住者|応募者)|외국인|해외\s*(?:거주자|지원자)"
        ),
        positive_value="yes",
        negative_value="no",
    )


def _infer_language_requirements(text: str) -> list[str]:
    requirements: list[str] = []

    def add(value: str) -> None:
        if value not in requirements:
            requirements.append(value)

    japanese_level = re.search(
        r"(?:JLPT\s*)?(N[1-5])(?:\s*(?:以上|レベル))?",
        text,
        re.IGNORECASE,
    )
    if japanese_level and re.search(r"JLPT|日本語|Japanese", text, re.IGNORECASE):
        add(f"Japanese (JLPT {japanese_level.group(1).upper()})")
    korean_level = re.search(
        r"\bTOPIK\s*(?:level|급)?\s*([1-6])(?:급)?\b",
        text,
        re.IGNORECASE,
    )
    if korean_level:
        add(f"Korean (TOPIK {korean_level.group(1)})")

    language_specs = (
        ("Japanese", r"Japanese|日本語|일본어"),
        ("Korean", r"Korean|韓国語|한국어"),
        ("English", r"English|英語|영어"),
    )
    qualifier_specs = (
        ("business", r"business(?:[- ]level)?|ビジネスレベル|비즈니스\s*수준"),
        ("native", r"native(?:[- ]level)?|ネイティブ|원어민"),
        ("fluent", r"fluent|流暢|유창"),
        (
            "conversational",
            r"conversational(?:[- ]level)?|日常会話|회화\s*(?:수준|가능)",
        ),
        (
            "professional",
            r"professional(?:[- ]level)?|業務レベル|업무\s*수준",
        ),
    )
    requirement_cue = (
        r"required|must|preferred|proficiency|必須|歓迎|必要|"
        r"필수|우대|능통|가능"
    )
    for language, alias in language_specs:
        occurrences = list(re.finditer(alias, text, re.IGNORECASE))
        if not occurrences:
            continue
        qualifier = ""
        explicit = False
        for occurrence in occurrences:
            line_start = text.rfind("\n", 0, occurrence.start()) + 1
            line_end = text.find("\n", occurrence.end())
            if line_end < 0:
                line_end = len(text)
            start = max(line_start, occurrence.start() - 80)
            end = min(line_end, occurrence.end() + 80)
            context = text[start:end]
            explicit = explicit or bool(
                re.search(requirement_cue, context, re.IGNORECASE)
            )
            for label, pattern in qualifier_specs:
                if re.search(pattern, context, re.IGNORECASE):
                    qualifier = label
                    explicit = True
                    break
            if qualifier:
                break
        if qualifier:
            add(f"{language} ({qualifier})")
        elif explicit:
            add(language)
    return requirements


def _infer_experience_requirements(text: str) -> list[str]:
    patterns = (
        r"[^.;\n]{0,70}\b(?:at least|minimum(?: of)?|min\.?)?\s*"
        r"\d+\+?(?:\s*[-–]\s*\d+)?\s+years?"
        r"(?:\s+of)?\s+experience[^.;\n]{0,80}",
        r"[^.;\n]{0,70}\b\d+\+?\s+años?(?:\s+de)?\s+experiencia[^.;\n]{0,80}",
        r"[^。；;\n]{0,70}(?:経験)?\s*\d+\s*年以上[^。；;\n]{0,80}",
        r"[^。；;\n]{0,70}\d+\s*年(?:程度|以上)の経験[^。；;\n]{0,80}",
        r"[^.;\n]{0,70}(?:경력\s*)?\d+\s*년\s*이상[^.;\n]{0,80}",
        r"[^.;\n]{0,70}\d+\s*년(?:의)?\s*경력[^.;\n]{0,80}",
    )
    line_patterns = (
        r"\b(?:at least|minimum(?: of)?|min\.?)?\s*\d+\+?(?:\s*[-–]\s*\d+)?\s+years?\b",
        r"\b\d+\+?\s+años?\b",
        r"\d+\s*年以上|\d+\s*年(?:程度|以上)の経験",
        r"\d+\s*년\s*이상|\d+\s*년(?:의)?\s*경력",
    )
    experience_cue = re.compile(
        r"experience|hands[- ]on|industry|academic|relevant work|"
        r"professional background|proven expertise|実務経験|経験|경력|경험|experiencia",
        re.IGNORECASE,
    )
    exclusion_cue = re.compile(
        r"salary|compensation|payout|annual salary|fixed overtime|"
        r"勤務時間|休日|給与|年収|月収|보험|급여",
        re.IGNORECASE,
    )
    requirements: list[str] = []

    def add(fragment: str) -> None:
        cleaned = _truncate_at_word(
            re.sub(r"\s+", " ", fragment or "").strip(" ,:-•-"),
            160,
        )
        if cleaned and cleaned not in requirements:
            requirements.append(cleaned)

    for line in (text or "").splitlines():
        normalized_line = re.sub(r"\s+", " ", line).strip(" ,:-•-")
        if not normalized_line or exclusion_cue.search(normalized_line):
            continue
        if experience_cue.search(normalized_line) and any(
            re.search(pattern, normalized_line, re.IGNORECASE)
            for pattern in line_patterns
        ):
            add(normalized_line)
            if len(requirements) >= 3:
                return requirements

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            fragment = match.group(0)
            if exclusion_cue.search(fragment):
                continue
            add(fragment)
            if len(requirements) >= 3:
                return requirements
    return requirements


__all__ = [
    "_extract_structured_sections",
    "_infer_experience_requirements",
    "_infer_foreigner_acceptance",
    "_infer_hard_skills",
    "_infer_language_requirements",
    "_infer_relocation_support",
    "_infer_soft_skills",
    "_infer_visa_status",
    "_sanitize_description",
]

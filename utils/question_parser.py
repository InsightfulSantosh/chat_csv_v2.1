import re
import difflib
import pandas as pd
from typing import ClassVar, List, Set, Tuple, Dict
from pydantic import PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config_manager import get_config_manager
from utils.llm_factory import LLMFactory
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Constants
FORBIDDEN_REFERENTS: Set[str] = {
    "there", "those", "them", "these", "that", "above", "their",
    "dataset", "entire data", "whole data", "full data"
}

FORBIDDEN_PATTERNS = [
    r'\btop\s+\d+\b', r'\bbottom\s+\d+\b', r'\bfirst\s+\d+\b', r'\blast\s+\d+\b',
    r'\bhighest\s+\d+\b', r'\blowest\s+\d+\b', r'\btop\s+\w+\b', r'\bbest\s+\d+\b',
    r'\bworst\s+\d+\b', r'\bmaximum\s+\d+\b', r'\bminimum\s+\d+\b',
]

# Helper functions
def extract_forbidden_terms(text: str) -> List[str]:
    return [word for word in FORBIDDEN_REFERENTS if word in text.lower()]

def detect_forbidden_patterns(text: str) -> List[str]:
    return [match for pattern in FORBIDDEN_PATTERNS for match in re.findall(pattern, text, re.IGNORECASE)]

def get_fuzzy_matches(user_input: str, candidates: List[str], case_map: Dict[str, str], cutoff=0.90) -> Dict[str, str]:
    tokens = re.findall(r"\b[\w()]+(?: [\w()]+)*\b", user_input.lower())
    return {
        word: case_map.get(close[0], close[0])
        for word in tokens
        if (close := difflib.get_close_matches(word, candidates, n=1, cutoff=cutoff))
    }

class RewriteQuestionTool(BaseTool):
    """Tool for strictly rewriting user questions to align with dataset schema."""

    name: ClassVar[str] = "StrictQuestionRewriter"
    description: ClassVar[str] = (
        "Strictly rewrites user questions to align with the dataset schema and known values. "
        "Fixes typos using fuzzy matching, but does not alter referents or structure."
    )

    _df: pd.DataFrame = PrivateAttr()
    _llm: ChatGoogleGenerativeAI = PrivateAttr()
    _model_name: str = PrivateAttr()

    def __init__(self, df: pd.DataFrame, provider: str = "google", model_name: str = "gemini-1.5-pro"):
        super().__init__()
        self._df = df
        self._model_name = model_name
        self._llm = LLMFactory.create_llm(
            provider=provider,
            model=model_name,
            temperature=0
        )

    def extract_values(self, date_threshold=0.8, freq_threshold=0.2, topn=50) -> Tuple[List[str], List[str]]:
        obj_df = self._df.select_dtypes(include="object")
        date_cols = [
            col for col in obj_df.columns
            if pd.to_datetime(obj_df[col], errors="coerce", format="mixed").notna().mean() >= date_threshold
        ]

        filtered_df = obj_df.drop(columns=date_cols, errors="ignore")
        values = {
            str(v).strip()
            for col in filtered_df.columns
            if filtered_df[col].nunique() / len(filtered_df) <= freq_threshold
            for v in filtered_df[col].dropna().value_counts().head(topn).index
            if v and len(str(v).strip()) < 100
        }

        return list(self._df.columns), sorted(values)

    def _run(self, user_input: str) -> str:
        columns, values = self.extract_values()
        combined_map = {**{c.lower(): c for c in columns}, **{v.lower(): v for v in values}}
        fuzzy_replacements = get_fuzzy_matches(user_input, list(combined_map.keys()), combined_map)

        corrected_input = user_input
        for old, new in fuzzy_replacements.items():
            corrected_input = re.sub(rf"\b{re.escape(old)}\b", new, corrected_input, flags=re.IGNORECASE)

        forbidden_terms = extract_forbidden_terms(user_input)
        forbidden_matches = detect_forbidden_patterns(user_input)

        escape = lambda lst: [s.replace('{', '{{').replace('}', '}}') for s in lst]
        escaped_columns, escaped_values = escape(columns), escape(values)

        prompt_template = f"""INSTRUCTIONS:
You are a typo-fixing assistant for a production-grade data pipeline. Your role is strictly limited to correcting column/value typos.

✅ YOUR TASK:
- Fix typos ONLY in column names and values
- Convert all words to lowercase
- Convert plurals to singular (e.g., "cities" → "city")
- Return output exactly as received with typo fixes applied

🚫 YOU MUST NOT:
- Expand or infer numeric references ("top N", "first 5", etc.)
- Interpret referents like: "these", "that", "there", "those"
- Change structure, grammar, or word order
- Add, remove, or rephrase any word
- Hallucinate or guess values/columns outside schema

🎯 VALID REFERENCES:
- Column names: {escaped_columns}
- Allowed values: {escaped_values}
- Patterns to preserve: {forbidden_matches or 'None'}
- Referents to preserve: {forbidden_terms or 'None'}

📌 RULES:
- Use only terms from the `columns` and `values_map` lists
- Leave unmatched terms unchanged unless typo is obvious
- Always apply `lowercase` and `singular` rules
- Keep original sentence structure intact

🔎 USER INPUT:
{corrected_input}

➡️ OUTPUT (REWRITE ONLY):
REWRITE:
"""

        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self._llm | StrOutputParser()

        rewritten = chain.invoke({
            "corrected_input": corrected_input,
            "columns": ", ".join(escaped_columns),
            "unique_values": ", ".join(escaped_values),
            "forbidden_patterns": ", ".join(forbidden_matches) if forbidden_matches else "None",
            "present_forbidden": ", ".join(forbidden_terms) if forbidden_terms else "None"
        }).strip()
        # Remove 'REWRITE:' label if LLM adds it anyway
        if rewritten.lower().startswith("rewrite:"):
            rewritten = rewritten[len("rewrite:"):].strip()

        # Sanity check: ensure no original words were dropped
        original_tokens = set(re.findall(r'\b\w+\b', corrected_input.lower()))
        rewritten_tokens = set(re.findall(r'\b\w+\b', rewritten.lower()))
        missing = original_tokens - rewritten_tokens
        if missing:
            logger.warning(f"⚠️ WARNING: These words were removed during rewriting: {', '.join(sorted(missing))}")

        return rewritten


class QuestionParser:
    """Main class for parsing and rewriting user questions."""

    def __init__(self, df: pd.DataFrame, provider: str = "google", model_name: str = "gemini-1.5-pro"):
        self.rewrite_tool = RewriteQuestionTool(df, provider, model_name)

    def rewrite_question(self, question: str) -> str:
        return self.rewrite_tool._run(question)

    def extract_schema_info(self) -> Dict[str, List[str]]:
        columns, values = self.rewrite_tool.extract_values()
        return {"columns": columns, "values": values}


def rewrite_user_question(question: str, df: pd.DataFrame, provider: str = "google", model_name: str = "gemini-1.5-pro") -> str:
    return QuestionParser(df, provider, model_name).rewrite_question(question) 
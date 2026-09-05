"""
AutoGrade Writing Pro
----------------------
A Streamlit application for English teachers to evaluate student
Analytical Exposition essays using the Google Gemini API.

Requires: streamlit, pandas, google-genai, pydantic
"""

import io
import re
import time
import html
import json
from typing import List, Literal, Optional

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

APP_TITLE = "AutoGrade Writing Pro"

# Current, live Gemini models (as of the app's release). Update this list
# if Google retires or adds models — check https://ai.google.dev/gemini-api/docs/models
MODEL_OPTIONS = {
    "Gemini 3.5 Flash (fast, recommended default)": "gemini-3.5-flash",
    "Gemini 3.1 Pro (highest quality, slower/costlier)": "gemini-3.1-pro-preview",
    "Gemini 3.5 Flash-Lite (cheapest, lighter reasoning)": "gemini-3.5-flash-lite",
}

RUBRIC_CRITERIA = ["Content", "Organization", "Grammar", "Vocabulary", "Mechanics"]
MAX_POINTS_PER_CRITERION = 20
MAX_TOTAL_SCORE = MAX_POINTS_PER_CRITERION * len(RUBRIC_CRITERIA)

CATEGORY_EMOJI = {
    "Content": "💡",
    "Organization": "🧩",
    "Grammar": "✏️",
    "Vocabulary": "📖",
    "Mechanics": "🔎",
}

CATEGORY_COLOR = {
    "Content": "#FFF3CD",       # soft yellow
    "Organization": "#D6E4FF",  # soft blue
    "Grammar": "#FFD6D6",       # soft red
    "Vocabulary": "#E2D6FF",    # soft purple
    "Mechanics": "#D6FFE2",     # soft green
}

RUBRIC_DESCRIPTION = """
You are an expert, strict but fair English teacher grading a student's
ANALYTICAL EXPOSITION TEXT. An Analytical Exposition must have:
  1. Thesis (introduces the topic and states the writer's position),
  2. Arguments (a series of points, each with supporting evidence/elaboration),
  3. Reiteration (restates the position / conclusion).

Grade strictly against this 5-criteria rubric. Each criterion is worth a
MAXIMUM of 20 points (total maximum = 100 points):

- Content (0-20): Relevance and depth of ideas, quality of thesis and
  arguments, strength of supporting evidence, logical reasoning.
- Organization (0-20): Presence and clarity of thesis / arguments /
  reiteration structure, paragraphing, cohesion and coherence, use of
  transition/connective devices.
- Grammar (0-20): Sentence structure accuracy, tense consistency,
  subject-verb agreement, correct use of complex/compound sentences.
- Vocabulary (0-20): Range and precision of word choice, appropriate
  register/formality, use of topic-specific and academic vocabulary.
- Mechanics (0-20): Spelling, punctuation, capitalization, formatting.

Scoring bands per criterion (use as a guide, then assign a precise integer):
  18-20 excellent | 14-17 good | 10-13 adequate | 5-9 weak | 0-4 very poor
"""


# ----------------------------------------------------------------------
# Structured output schema (Gemini will be constrained to return this)
# ----------------------------------------------------------------------

class ScoreBreakdown(BaseModel):
    content: int = Field(ge=0, le=MAX_POINTS_PER_CRITERION)
    organization: int = Field(ge=0, le=MAX_POINTS_PER_CRITERION)
    grammar: int = Field(ge=0, le=MAX_POINTS_PER_CRITERION)
    vocabulary: int = Field(ge=0, le=MAX_POINTS_PER_CRITERION)
    mechanics: int = Field(ge=0, le=MAX_POINTS_PER_CRITERION)


class Annotation(BaseModel):
    quoted_text: str = Field(
        description="A short verbatim snippet (max ~12 words) copied exactly "
        "from the student's essay that this comment refers to."
    )
    category: Literal["Content", "Organization", "Grammar", "Vocabulary", "Mechanics"]
    severity: Literal["minor", "major"]
    comment: str = Field(description="A brief, constructive explanation of the issue or highlight.")


class EssayEvaluation(BaseModel):
    scores: ScoreBreakdown
    total_score: int = Field(ge=0, le=MAX_TOTAL_SCORE)
    strengths: List[str] = Field(description="2-4 concrete strengths of the essay.")
    improvements: List[str] = Field(description="2-4 concrete, actionable suggestions for improvement.")
    overall_feedback: str = Field(
        description="A constructive, encouraging paragraph (4-8 sentences) summarizing "
        "the evaluation, written directly to the student."
    )
    annotations: List[Annotation] = Field(
        description="6-15 inline annotations pointing to specific parts of the essay."
    )


# ----------------------------------------------------------------------
# Gemini call
# ----------------------------------------------------------------------

def build_prompt(student_name: str, student_class: str, essay_text: str) -> str:
    return f"""{RUBRIC_DESCRIPTION}

Student name: {student_name or "N/A"}
Class: {student_class or "N/A"}

Evaluate the following student essay. Quote snippets EXACTLY as they
appear in the essay (so they can be located via string matching) for
every annotation.

--- ESSAY START ---
{essay_text}
--- ESSAY END ---
"""


def evaluate_essay(
    client: genai.Client,
    model_name: str,
    student_name: str,
    student_class: str,
    essay_text: str,
) -> EssayEvaluation:
    """Calls the Gemini API and returns a validated EssayEvaluation."""
    prompt = build_prompt(student_name, student_class, essay_text)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=EssayEvaluation,
        ),
    )

    # The SDK parses response.text into the pydantic model automatically
    # when response_schema is a pydantic BaseModel. Fall back to manual
    # parsing if .parsed is unavailable for any reason.
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, EssayEvaluation):
        result = parsed
    else:
        result = EssayEvaluation.model_validate(json.loads(response.text))

    # Defensive recompute: trust the rubric sum over the model's own total,
    # in case of any arithmetic slip.
    computed_total = (
        result.scores.content
        + result.scores.organization
        + result.scores.grammar
        + result.scores.vocabulary
        + result.scores.mechanics
    )
    result.total_score = computed_total
    return result


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------

def render_annotated_text(essay_text: str, annotations: List[Annotation]) -> str:
    """Builds HTML with highlighted spans + emoji for each annotation that
    can be located verbatim in the essay. Longer snippets are applied first
    to reduce the chance of partial/overlapping matches."""
    escaped = html.escape(essay_text)

    # Sort longest-first so shorter substrings of an already-wrapped
    # snippet don't get double-wrapped.
    sorted_annotations = sorted(annotations, key=lambda a: len(a.quoted_text), reverse=True)

    used_spans = set()
    for idx, ann in enumerate(sorted_annotations):
        snippet = ann.quoted_text.strip()
        if not snippet:
            continue
        escaped_snippet = html.escape(snippet)
        if escaped_snippet in used_spans:
            continue
        # Only replace the first occurrence to avoid runaway repeated highlighting
        pattern = re.escape(escaped_snippet)
        match = re.search(pattern, escaped)
        if not match:
            continue
        used_spans.add(escaped_snippet)
        color = CATEGORY_COLOR.get(ann.category, "#EEEEEE")
        emoji = CATEGORY_EMOJI.get(ann.category, "🔸")
        tooltip = html.escape(f"[{ann.category} - {ann.severity}] {ann.comment}")
        replacement = (
            f'<span style="background-color:{color}; border-radius:3px; padding:1px 2px;" '
            f'title="{tooltip}">{escaped_snippet} {emoji}</span>'
        )
        escaped = escaped[: match.start()] + replacement + escaped[match.end():]

    return escaped.replace("\n", "<br>")


def render_score_summary(result: EssayEvaluation):
    total_pct = round((result.total_score / MAX_TOTAL_SCORE) * 100)
    st.metric("Total Score", f"{result.total_score} / {MAX_TOTAL_SCORE}", f"{total_pct}%")

    cols = st.columns(5)
    breakdown = {
        "Content": result.scores.content,
        "Organization": result.scores.organization,
        "Grammar": result.scores.grammar,
        "Vocabulary": result.scores.vocabulary,
        "Mechanics": result.scores.mechanics,
    }
    for col, (criterion, score) in zip(cols, breakdown.items()):
        with col:
            st.markdown(f"**{CATEGORY_EMOJI[criterion]} {criterion}**")
            st.progress(score / MAX_POINTS_PER_CRITERION)
            st.caption(f"{score} / {MAX_POINTS_PER_CRITERION}")


def render_result(essay_text: str, result: EssayEvaluation, key_prefix: str = ""):
    render_score_summary(result)

    st.subheader("📝 Annotated Essay")
    st.caption("Hover over a highlighted phrase to see the comment.")
    annotated_html = render_annotated_text(essay_text, result.annotations)
    st.markdown(
        f'<div style="line-height:1.9; padding:14px; border:1px solid #DDD; '
        f'border-radius:8px; background-color:#FAFAFA;">{annotated_html}</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("✅ Strengths")
        for s in result.strengths:
            st.markdown(f"- {s}")
    with col_b:
        st.subheader("🎯 Areas to Improve")
        for i in result.improvements:
            st.markdown(f"- {i}")

    st.subheader("🗒️ Overall Feedback")
    st.info(result.overall_feedback)

    with st.expander("View all annotation comments"):
        for ann in result.annotations:
            st.markdown(
                f"**{CATEGORY_EMOJI.get(ann.category, '🔸')} {ann.category} "
                f"({ann.severity})** — _{ann.quoted_text}_\n\n{ann.comment}"
            )


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

def get_client(api_key: str) -> Optional[genai.Client]:
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Could not initialize Gemini client: {e}")
        return None


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="✍️", layout="wide")
    st.title(f"✍️ {APP_TITLE}")
    st.caption("AI-assisted essay grading for Analytical Exposition texts, powered by Google Gemini.")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get a key from https://aistudio.google.com/apikey. "
            "It is only kept in this browser session and never saved to disk.",
        )
        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
        model_name = MODEL_OPTIONS[model_label]

        st.divider()
        mode = st.radio("Mode", ["Single Evaluation", "Batch Evaluation"])

        st.divider()
        st.caption(
            "Rubric: Content, Organization, Grammar, Vocabulary, Mechanics — "
            f"{MAX_POINTS_PER_CRITERION} pts each, {MAX_TOTAL_SCORE} pts total."
        )

    if not api_key:
        st.warning("👈 Enter your Google Gemini API key in the sidebar to get started.")
        st.stop()

    client = get_client(api_key)
    if client is None:
        st.stop()

    # ---------------- Single Evaluation ----------------
    if mode == "Single Evaluation":
        st.subheader("Single Essay Evaluation")

        col1, col2 = st.columns(2)
        with col1:
            student_name = st.text_input("Student Name")
        with col2:
            student_class = st.text_input("Class")

        essay_text = st.text_area("Essay Text (Analytical Exposition)", height=300)

        evaluate_clicked = st.button("🚀 Evaluate Essay", type="primary")

        if evaluate_clicked:
            if not essay_text.strip():
                st.error("Please paste the student's essay text before evaluating.")
            else:
                with st.spinner(f"Evaluating with {model_name}..."):
                    try:
                        result = evaluate_essay(
                            client, model_name, student_name, student_class, essay_text
                        )
                        st.session_state["single_result"] = result
                        st.session_state["single_essay_text"] = essay_text
                    except (ClientError, ServerError) as e:
                        st.error(f"Gemini API error: {e}")
                    except json.JSONDecodeError:
                        st.error("The model did not return valid JSON. Please try again.")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

        if "single_result" in st.session_state:
            st.divider()
            render_result(
                st.session_state["single_essay_text"],
                st.session_state["single_result"],
            )

    # ---------------- Batch Evaluation ----------------
    else:
        st.subheader("Batch Essay Evaluation")
        st.markdown(
            "Upload a CSV file with the columns: **`student_name`**, **`class`**, **`essay`**."
        )

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                return

            required_cols = {"student_name", "class", "essay"}
            missing = required_cols - set(df_input.columns)
            if missing:
                st.error(f"CSV is missing required column(s): {', '.join(missing)}")
                return

            st.dataframe(df_input.head(), use_container_width=True)
            st.caption(f"{len(df_input)} row(s) detected.")

            if st.button("🚀 Run Batch Evaluation", type="primary"):
                progress = st.progress(0, text="Starting...")
                results_rows = []
                total = len(df_input)

                for i, row in df_input.iterrows():
                    name = str(row.get("student_name", ""))
                    cls = str(row.get("class", ""))
                    essay = str(row.get("essay", ""))

                    progress.progress((i) / total, text=f"Evaluating {name or f'row {i+1}'}...")

                    if not essay.strip():
                        results_rows.append({
                            "student_name": name, "class": cls, "total_score": None,
                            "content": None, "organization": None, "grammar": None,
                            "vocabulary": None, "mechanics": None,
                            "overall_feedback": "SKIPPED: empty essay",
                        })
                        continue

                    try:
                        result = evaluate_essay(client, model_name, name, cls, essay)
                        results_rows.append({
                            "student_name": name,
                            "class": cls,
                            "total_score": result.total_score,
                            "content": result.scores.content,
                            "organization": result.scores.organization,
                            "grammar": result.scores.grammar,
                            "vocabulary": result.scores.vocabulary,
                            "mechanics": result.scores.mechanics,
                            "overall_feedback": result.overall_feedback,
                        })
                    except (ClientError, ServerError) as e:
                        results_rows.append({
                            "student_name": name, "class": cls, "total_score": None,
                            "content": None, "organization": None, "grammar": None,
                            "vocabulary": None, "mechanics": None,
                            "overall_feedback": f"ERROR: {e}",
                        })
                    except Exception as e:
                        results_rows.append({
                            "student_name": name, "class": cls, "total_score": None,
                            "content": None, "organization": None, "grammar": None,
                            "vocabulary": None, "mechanics": None,
                            "overall_feedback": f"ERROR: {e}",
                        })

                    # Small delay to be gentle on rate limits for large batches
                    time.sleep(0.3)

                progress.progress(1.0, text="Done!")
                df_results = pd.DataFrame(results_rows)

                st.success(f"Evaluated {len(df_results)} essay(s).")
                st.dataframe(df_results, use_container_width=True)

                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="autograde_batch_results.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()

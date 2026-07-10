# reducerGraph.py
from __future__ import annotations

import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from models import GlobalImagePlan, State

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3,
)

image_client = OpenAI()


def merge_content(state: State) -> dict:
    plan = state["plan"]

    sections = state.get("sections", [])
    ordered_sections = [md for _, md in sorted(sections, key=lambda x: x[0])]

    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"

    return {"merged_md": merged_md}


def build_image_context(markdown: str) -> str:
    sections = []

    current_heading = None
    current_lines = []

    for line in markdown.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            if current_heading:
                sections.append(
                    current_heading + "\n" + " ".join(current_lines[:3])
                )

            current_heading = line
            current_lines = []

        else:
            if len(current_lines) < 3:
                current_lines.append(line)

    if current_heading:
        sections.append(
            current_heading + "\n" + " ".join(current_lines[:3])
        )

    return "\n\n".join(sections)


DECIDE_IMAGES_SYSTEM = """
You are an expert technical editor.

Your task is ONLY to decide where images should appear.

You are given:
- Blog metadata
- Section headings
- Short summaries

Rules:

- Maximum 3 images.
- Only suggest diagrams that improve understanding.
- Prefer:
  * architecture diagrams
  * flowcharts
  * comparison tables
  * timelines
  * conceptual illustrations

Never create decorative images.

Insert placeholders exactly:

[[IMAGE_1]]
[[IMAGE_2]]
[[IMAGE_3]]

Return ONLY GlobalImagePlan.
"""


def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)

    plan = state["plan"]
    merged_md = state["merged_md"]

    compact_context = build_image_context(merged_md)

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=f"""
Title: {plan.blog_title}

Audience: {plan.audience}

Blog Kind: {plan.blog_kind}

Topic: {state["topic"]}

Article Summary

{compact_context}
"""
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _generate_image(prompt: str, out_path: Path):
    import base64

    result = image_client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        quality="high",
        size="1024x1024",
    )

    img = base64.b64decode(result.data[0].b64_json)

    out_path.write_bytes(img)


def generate_and_place_images(state: State):
    plan = state["plan"]

    md = state.get("md_with_placeholders") or state["merged_md"]
    specs = state.get("image_specs", [])

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in specs:
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                _generate_image(spec["prompt"], out_path)
            except Exception as e:
                md = md.replace(
                    spec["placeholder"],
                    f"> Image generation failed\n>\n> {e}"
                )
                continue

        md = md.replace(
            spec["placeholder"],
            f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        )

    state["fileName"] = re.sub(r'[<>:"/\\\\|?*]', "_", plan.blog_title)

    # Path(f"{state["fileName"]}.md").write_text(
    #     md,
    #     encoding="utf-8",
    # )

    return {"final": md}


def markdown_validator(state: State) -> str:
    MARKDOWN_VALIDATOR_SYSTEM=MARKDOWN_VALIDATOR_SYSTEM = """
    You are a senior Markdown validator.

    Your ONLY job is to fix formatting issues.

    DO NOT:
    - rewrite content
    - summarize
    - add new information
    - remove technical details
    - change explanations

    ONLY fix:

    1. Markdown syntax
    - headings
    - lists
    - tables
    - links
    - images
    - blockquotes

    2. Code blocks
    - close every fence
    - add missing language if obvious
    - never modify code logic

    3. Math

    Inline math:

    $...$

    Display math:

    $$
    ...
    $$

    Never output:

    \\[
    ...
    \\]

    Never output raw LaTeX outside math delimiters.

    4. Whitespace

    5. Invalid Markdown

    Return ONLY valid GitHub-Flavored Markdown.
    """
    validator = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
    )

    try:
        response = validator.invoke(
            [
                SystemMessage(content=MARKDOWN_VALIDATOR_SYSTEM),
                HumanMessage(content=state["final"]),
            ]
        )

        final_md = response.content

        # Gemini sometimes returns list instead of string
        if isinstance(final_md, list):
            text_parts = []

            for part in final_md:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    text_parts.append(part.get("text", ""))
                elif hasattr(part, "text"):
                    text_parts.append(part.text)

            final_md = "\n".join(text_parts)

        final_md = final_md.strip()

    except Exception as e:
        print(f"[Markdown Validator Error] {e}")
        print("Using original markdown.")

        # Fallback to original markdown
        final_md = state["final"]

    # Save the final markdown (validated or original)
    Path(f"{state["fileName"]}.md").write_text(
        final_md,
        encoding="utf-8"
    )

    return {
        "final": final_md
    }

reducer_graph = StateGraph(State)

reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_node("markdown_validator",markdown_validator)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", "markdown_validator")
reducer_graph.add_edge("markdown_validator",END)

reducer_subgraph = reducer_graph.compile()

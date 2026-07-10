from pydantic import Field,BaseModel
from typing import TypedDict, List, Optional, Literal, Annotated
from models import EvidenceItem,Plan,RouterDecision,EvidencePack,Task
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv
from datetime import date,timedelta
from reducerGraph1 import reducer_subgraph
from models import State
import os
load_dotenv()

llm=None
# if not os.environ.get("Groq_API_KEY"):
#     print("Error: Groq_API_KEY is still not detected in the environment.")
#     process.exit(1)
# else:
#     llm=ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0.5
#     )

# if not os.environ.get("GOOGLE_API_KEY"):
#     print("Error: GOOGLE_API_KEY is still not detected in the environment.")
#     process.exit(1)
# else:
#     llm=ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash",
#     temperature=0.5
#     )

if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY is still not detected in the environment.")
    process.exit(1)
else:
    llm=ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.5
    )



# -----------------------------
# 3) Router (decide upfront)
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state: State) -> dict:
    
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"



def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    
    tool = TavilySearch(max_results=max_results)
    results = tool.invoke({"query": query})["results"]

    # print(results["results"]," ",type(results)," ")
    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:

    # take the first 10 queries from state
    queries = (state.get("queries", []) or [])
    max_results = 6

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}



ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create ONLY 4–6 sections.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.

IMPORTANT:
This application has a strict LLM budget.
Every task results in one additional LLM request.
Optimize the outline to minimize the total number of model calls while maintaining quality.

Do NOT split content into multiple tiny sections.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    return {"plan": plan}

# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.

Formatting Rules (STRICT)

Markdown:
- Output valid GitHub-Flavored Markdown only.
- Never output malformed Markdown.
- Every heading must begin with ## or ###.
- Lists must use '-' or numbered lists.
- Always close every Markdown code fence.
- Never leave unfinished code blocks.

Mathematical expressions:
- Use LaTeX ONLY when mathematically necessary.
- Inline equations MUST use:
  $ ... $

Example:
The attention score is computed as
$QK^T$.

- Block equations MUST use:

$$
...
$$

Example:

$$
Attention(Q,K,V)=Softmax\left(\\frac{QK^T}{\sqrt{d_k}}\\right)V
$$

Never output:

\[
...
\]

Never output raw LaTeX commands outside $...$ or $$...$$.

Tables:
- Produce valid Markdown tables.

Code:
- Always specify the language.

Example:

```python
...
"""

def worker_node(payload: dict) -> dict:
    
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    # section_md = llm.invoke(
    #     [
    #         SystemMessage(content=WORKER_SYSTEM),
    #         HumanMessage(
    #             content=(
    #                 f"Blog title: {plan.blog_title}\n"
    #                 f"Audience: {plan.audience}\n"
    #                 f"Tone: {plan.tone}\n"
    #                 f"Blog kind: {plan.blog_kind}\n"
    #                 f"Constraints: {plan.constraints}\n"
    #                 f"Topic: {topic}\n"
    #                 f"Mode: {mode}\n\n"
    #                 f"Section title: {task.title}\n"
    #                 f"Goal: {task.goal}\n"
    #                 f"Target words: {task.target_words}\n"
    #                 f"Tags: {task.tags}\n"
    #                 f"requires_research: {task.requires_research}\n"
    #                 f"requires_citations: {task.requires_citations}\n"
    #                 f"requires_code: {task.requires_code}\n"
    #                 f"Bullets:{bullets_text}\n\n"
    #                 f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
    #             )
    #         ),
    #     ]
    # ).content.strip()


    response = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    )

    # ---------------- Debug ----------------
    print("\n" + "=" * 80)
    print("Response object type :", type(response))
    print("Content type         :", type(response.content))
    print("Content value        :")
    print(response.content)
    print("=" * 80 + "\n")
    # ---------------------------------------

    content = response.content

    # Case 1: Normal string response
    if isinstance(content, str):
        section_md = content.strip()

    # Case 2: Multi-part response
    elif isinstance(content, list):
        text_parts = []

        for part in content:
            print("PART TYPE:", type(part))
            print("PART:", part)

            if isinstance(part, str):
                text_parts.append(part)

            elif isinstance(part, dict):
                if "text" in part:
                    text_parts.append(part["text"])

            elif hasattr(part, "text"):
                text_parts.append(part.text)

        section_md = "\n".join(text_parts).strip()

    # Unknown response format
    else:
        raise TypeError(
            f"Unexpected response.content type: {type(content)}\n"
            f"Value: {content}"
        )

    print("\nGenerated Markdown:\n")
    print(section_md)
    print("\n" + "=" * 80)
    return {"sections": [(task.id, section_md)]}




globe=StateGraph(State)
globe.add_node("router",router_node)
globe.add_node("orchestrator",orchestrator_node)
globe.add_node("research",research_node)
globe.add_node("worker",worker_node)
globe.add_node("reducer",reducer_subgraph)

globe.add_edge(START,"router")
globe.add_conditional_edges("router",route_next,{"research":"research","orchestrator":"orchestrator"})
globe.add_edge("research","orchestrator")
globe.add_conditional_edges("orchestrator",fanout,["worker"])
globe.add_edge("worker","reducer")
globe.add_edge("reducer",END)

app=globe.compile()


def run(topic: str, as_of: Optional[str] = None):
    if as_of is None:
        as_of = date.today().isoformat()

    out = app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of,
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }
    )

    return out

if __name__=="__main__":
   run("Attention machanism in transformers") 
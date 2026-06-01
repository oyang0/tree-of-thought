import heapq
import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
import streamlit as st
from openai import OpenAI


# -----------------------------
# Data model
# -----------------------------
@dataclass
class ThoughtNode:
    node_id: int
    parent_id: Optional[int]
    thought: str
    depth: int
    path_probability: float
    cost: float
    is_terminal: bool = False
    final_answer: Optional[str] = None
    children: list[int] = field(default_factory=list)
    expanded: bool = False
    error: Optional[str] = None


# -----------------------------
# Utility functions
# -----------------------------
DEFAULT_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "DeepSeek-V4-Pro"
EPSILON = 1e-12


def initialize_state() -> None:
    if "nodes" not in st.session_state:
        root = ThoughtNode(
            node_id=0,
            parent_id=None,
            thought="ROOT: Problem statement",
            depth=0,
            path_probability=1.0,
            cost=0.0,
        )
        st.session_state.nodes = {0: root}
        st.session_state.next_node_id = 1
        st.session_state.frontier = [(0.0, 0, 0)]
        st.session_state.generated_thoughts = 0
        st.session_state.expanded_nodes = 0
        st.session_state.final_answer = None
        st.session_state.best_terminal_node_id = None
        st.session_state.last_error = None
        st.session_state.search_started = False
        st.session_state.search_finished = False


def reset_search(problem: str) -> None:
    root = ThoughtNode(
        node_id=0,
        parent_id=None,
        thought=f"ROOT: {problem[:500]}",
        depth=0,
        path_probability=1.0,
        cost=0.0,
    )
    st.session_state.nodes = {0: root}
    st.session_state.next_node_id = 1
    st.session_state.frontier = [(0.0, 0, 0)]
    st.session_state.generated_thoughts = 0
    st.session_state.expanded_nodes = 0
    st.session_state.final_answer = None
    st.session_state.best_terminal_node_id = None
    st.session_state.last_error = None
    st.session_state.search_started = True
    st.session_state.search_finished = False


def clamp_probability(value) -> float:
    try:
        p = float(value)
        if math.isnan(p) or math.isinf(p):
            return EPSILON
        return max(EPSILON, min(1.0, p))
    except Exception:
        return EPSILON


def normalize_candidates(candidates: list[dict]) -> list[dict]:
    """Clamp malformed probabilities and normalize if total mass is unreasonable."""
    cleaned = []

    for item in candidates:
        if not isinstance(item, dict):
            continue

        thought = str(item.get("thought", "")).strip()
        if not thought:
            continue

        cleaned.append(
            {
                "thought": thought,
                "probability": clamp_probability(item.get("probability", EPSILON)),
                "is_terminal": bool(item.get("is_terminal", False)),
                "final_answer": item.get("final_answer"),
            }
        )

    if not cleaned:
        return []

    total = sum(c["probability"] for c in cleaned)

    # If the model returns scores rather than probabilities, normalize them.
    # If the values already sum to <= 1, keep them as approximate policy mass.
    if total > 1.0 + 1e-6:
        for c in cleaned:
            c["probability"] = clamp_probability(c["probability"] / total)

    return cleaned


def extract_json_object(text: str) -> dict:
    """Parse strict JSON, with a small fallback for fenced JSON output."""
    if not text:
        raise ValueError("Empty model response.")

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def path_to_node(node_id: int) -> list[ThoughtNode]:
    nodes = st.session_state.nodes
    path = []
    current = nodes[node_id]

    while current is not None:
        path.append(current)
        if current.parent_id is None:
            break
        current = nodes.get(current.parent_id)

    return list(reversed(path))


def visible_path_context(node_id: int) -> str:
    path = path_to_node(node_id)
    non_root = [n for n in path if n.parent_id is not None]
    if not non_root:
        return "No prior visible thoughts/actions yet."
    return "\n".join(
        f"{i + 1}. {n.thought}" for i, n in enumerate(non_root)
    )


def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url.strip() or DEFAULT_BASE_URL)


def generate_candidate_thoughts(
    client: OpenAI,
    problem: str,
    node: ThoughtNode,
    branching_factor: int,
) -> tuple[list[dict], Optional[str]]:
    system_prompt = """
You are a policy model for a Tree-of-Thoughts search algorithm.

Return ONLY valid JSON. Do not use markdown.

Generate concise, visible reasoning steps or candidate actions.
Do NOT expose hidden chain-of-thought or long private reasoning traces.
Each thought should be short and useful for progressing toward the answer.

You must estimate a policy probability for each candidate thought.
The probabilities do not need to sum to 1, but each must be between 0 and 1.

If a candidate directly gives the final answer, set is_terminal=true and include final_answer.
Otherwise set is_terminal=false and final_answer=null.
""".strip()

    user_prompt = f"""
Problem:
{problem}

Current visible path:
{visible_path_context(node.node_id)}

Current depth: {node.depth}

Generate up to {branching_factor} next candidate thoughts/actions.

Return this exact JSON shape:
{{
  "candidates": [
    {{
      "thought": "concise visible step or action",
      "probability": 0.42,
      "is_terminal": false,
      "final_answer": null
    }}
  ]
}}
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content
        parsed = extract_json_object(content)
        candidates = parsed.get("candidates", [])
        if not isinstance(candidates, list):
            return [], "Model JSON did not contain a list field named 'candidates'."
        return normalize_candidates(candidates)[:branching_factor], None

    except Exception as exc:
        return [], f"API/JSON error while expanding node {node.node_id}: {exc}"


def expand_one_node(
    client: OpenAI,
    problem: str,
    thought_budget: int,
    branching_factor: int,
    max_depth: int,
) -> bool:
    """Expand one lowest-cost frontier node. Returns True if search should continue."""
    nodes = st.session_state.nodes

    while st.session_state.frontier:
        _, _, node_id = heapq.heappop(st.session_state.frontier)
        node = nodes[node_id]

        if node.expanded:
            continue
        if node.is_terminal:
            st.session_state.final_answer = node.final_answer or node.thought
            st.session_state.best_terminal_node_id = node.node_id
            st.session_state.search_finished = True
            return False
        if node.depth >= max_depth:
            node.expanded = True
            continue

        remaining_budget = thought_budget - st.session_state.generated_thoughts
        if remaining_budget <= 0:
            st.session_state.search_finished = True
            return False

        ask_for = min(branching_factor, remaining_budget)
        candidates, error = generate_candidate_thoughts(
            client=client,
            problem=problem,
            node=node,
            branching_factor=ask_for,
        )

        node.expanded = True
        st.session_state.expanded_nodes += 1

        if error:
            node.error = error
            st.session_state.last_error = error
            return bool(st.session_state.frontier)

        if not candidates:
            node.error = "No valid candidate thoughts were returned."
            st.session_state.last_error = node.error
            return bool(st.session_state.frontier)

        for candidate in candidates:
            child_id = st.session_state.next_node_id
            st.session_state.next_node_id += 1

            local_p = clamp_probability(candidate["probability"])
            path_p = clamp_probability(node.path_probability * local_p)
            depth = node.depth + 1
            cost = depth / max(path_p, EPSILON)

            child = ThoughtNode(
                node_id=child_id,
                parent_id=node.node_id,
                thought=candidate["thought"],
                depth=depth,
                path_probability=path_p,
                cost=cost,
                is_terminal=bool(candidate["is_terminal"]),
                final_answer=candidate.get("final_answer"),
            )

            nodes[child_id] = child
            node.children.append(child_id)
            st.session_state.generated_thoughts += 1

            if child.is_terminal:
                st.session_state.final_answer = child.final_answer or child.thought
                st.session_state.best_terminal_node_id = child.node_id
                st.session_state.search_finished = True
                return False

            heapq.heappush(st.session_state.frontier, (child.cost, child.node_id, child.node_id))

            if st.session_state.generated_thoughts >= thought_budget:
                st.session_state.search_finished = True
                return False

        return True

    st.session_state.search_finished = True
    return False


def run_search(
    api_key: str,
    base_url: str,
    problem: str,
    thought_budget: int,
    branching_factor: int,
    max_depth: int,
) -> None:
    client = make_client(api_key, base_url)

    with st.status("Running Levin Tree Search-style Tree-of-Thoughts...", expanded=True) as status:
        start = time.time()
        step = 0

        while not st.session_state.search_finished:
            step += 1
            st.write(
                f"Step {step}: generated "
                f"{st.session_state.generated_thoughts}/{thought_budget} thoughts..."
            )

            should_continue = expand_one_node(
                client=client,
                problem=problem,
                thought_budget=thought_budget,
                branching_factor=branching_factor,
                max_depth=max_depth,
            )

            if not should_continue:
                break

        elapsed = time.time() - start

        if st.session_state.final_answer:
            status.update(
                label=f"Search finished with a terminal answer in {elapsed:.2f}s.",
                state="complete",
            )
        elif st.session_state.last_error:
            status.update(
                label="Search stopped with an error. See details below.",
                state="error",
            )
        else:
            status.update(
                label=f"Budget/depth exhausted in {elapsed:.2f}s.",
                state="complete",
            )


def render_tree(node_id: int = 0) -> None:
    node = st.session_state.nodes[node_id]

    label = (
        f"Node {node.node_id} | depth={node.depth} | "
        f"p={node.path_probability:.4g} | cost={node.cost:.4g}"
    )
    if node.is_terminal:
        label += " | TERMINAL"
    if node.error:
        label += " | ERROR"

    with st.expander(label, expanded=(node.depth <= 1)):
        st.markdown(f"**Thought:** {node.thought}")
        st.caption(
            f"parent={node.parent_id} · children={node.children} · expanded={node.expanded}"
        )

        if node.final_answer:
            st.success(f"Final answer: {node.final_answer}")

        if node.error:
            st.error(node.error)

        for child_id in node.children:
            render_tree(child_id)


def nodes_dataframe() -> pd.DataFrame:
    rows = []
    for node in st.session_state.nodes.values():
        row = asdict(node)
        row["children"] = ", ".join(map(str, node.children))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("node_id")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Policy-Guided ToT Search", page_icon="🌳", layout="wide")
initialize_state()

st.title("🌳 Policy-Guided Tree-of-Thoughts Search")
st.write(
    "A compact educational Streamlit implementation inspired by "
    "**Policy-Guided Search on Tree-of-Thoughts for Efficient Problem Solving "
    "with Bounded Language Model Queries**. "
    "The app treats the model as a policy over concise next thoughts/actions and "
    "uses a Levin Tree Search-style priority: `cost = depth / path_probability`."
)

with st.sidebar:
    st.header("API settings")
    api_key = st.text_input("DeepSeek API key", type="password")
    base_url = st.text_input(
        "OpenAI-compatible base URL",
        value=DEFAULT_BASE_URL,
        help="Default placeholder endpoint for DeepSeek's OpenAI-compatible API.",
    )

    st.header("Search controls")
    thought_budget = st.number_input(
        "Max thought budget",
        min_value=1,
        max_value=500,
        value=30,
        step=1,
        help="Each generated candidate thought counts against this budget.",
    )
    branching_factor = st.number_input(
        "Branching factor",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )
    max_depth = st.number_input(
        "Max depth",
        min_value=1,
        max_value=50,
        value=6,
        step=1,
    )

    st.divider()
    reset_clicked = st.button("Reset search tree", use_container_width=True)
    run_clicked = st.button("Run / continue search", type="primary", use_container_width=True)

problem = st.text_area(
    "Problem statement",
    height=150,
    placeholder="Example: Solve 8x + 7 = -23. Return the value of x.",
)

if reset_clicked:
    if problem.strip():
        reset_search(problem.strip())
        st.success("Search tree reset.")
    else:
        st.warning("Enter a problem before resetting the search tree.")

if run_clicked:
    if not api_key:
        st.error("Please provide a DeepSeek API key.")
    elif not problem.strip():
        st.error("Please enter a problem statement.")
    else:
        if not st.session_state.search_started:
            reset_search(problem.strip())

        if st.session_state.generated_thoughts >= thought_budget:
            st.warning("The current tree has already reached the configured thought budget.")
        elif st.session_state.search_finished and st.session_state.final_answer:
            st.info("A terminal answer has already been found. Reset to start a new search.")
        else:
            run_search(
                api_key=api_key,
                base_url=base_url,
                problem=problem.strip(),
                thought_budget=int(thought_budget),
                branching_factor=int(branching_factor),
                max_depth=int(max_depth),
            )

st.subheader("Current best / final answer")
if st.session_state.final_answer:
    st.success(st.session_state.final_answer)

    if st.session_state.best_terminal_node_id is not None:
        st.markdown("**Answer path:**")
        answer_path = [
            n for n in path_to_node(st.session_state.best_terminal_node_id)
            if n.parent_id is not None
        ]
        for i, node in enumerate(answer_path, start=1):
            st.markdown(f"{i}. {node.thought}")
else:
    st.info("No terminal answer found yet.")

if st.session_state.last_error:
    st.error(st.session_state.last_error)

st.subheader("Search statistics")
frontier_count = sum(
    1
    for _, _, node_id in st.session_state.frontier
    if not st.session_state.nodes[node_id].expanded
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Generated thoughts", st.session_state.generated_thoughts)
col2.metric("Thought budget", int(thought_budget))
col3.metric("Expanded nodes", st.session_state.expanded_nodes)
col4.metric("Frontier nodes", frontier_count)
col5.metric("Total nodes", len(st.session_state.nodes))

tab_tree, tab_table = st.tabs(["Expandable tree view", "Explored nodes table"])

with tab_tree:
    render_tree(0)

with tab_table:
    df = nodes_dataframe()
    st.dataframe(
        df[
            [
                "node_id",
                "parent_id",
                "thought",
                "depth",
                "path_probability",
                "cost",
                "is_terminal",
                "final_answer",
                "children",
                "expanded",
                "error",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

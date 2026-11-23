import os
import shutil
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from state import OrganizerState, FileItem

load_dotenv()

TARGET_ROOT = "D:\\"
MAX_DEPTH = 2
EXCLUDED_DIRS = {"$recycle.bin", "system volume information", "venv", "__pycache__", "node_modules", ".git"}
SKIP_FILES = {"desktop.ini", "thumbs.db"}
SKIP_EXTENSIONS = {".lnk", ".url", ".tmp", ".crdownload", ".part"}
MAX_CANDIDATES = 50

limiter = InMemoryRateLimiter(requests_per_second=0.25)
llm = ChatOpenAI(model="nvidia/nemotron-3.5-super-vl:free",
                 base_url="https://openrouter.ai/api/v1",
                 api_key=os.getenv("OPENROUTER_API_KEY"),
                 temperature=0, rate_limiter=limiter)


class Decision(BaseModel):
    topic: str
    target_folder: str
    reason: str


classifier = llm.with_structured_output(Decision)

PROMPT_V1 = """Choose the best folder for this file, based on the file name and the folder names.
File name: {name}
Extension: {extension}
Available folders:
{folders}
If the file name is in Hebrew, first translate it to English in your mind and identify its professional topic.
Return target_folder exactly as written in the list, or "unsure" if no folder fits. Add a short reason in Hebrew."""


def load_folders_node(state: OrganizerState) -> dict:
    result = []
    for current, subdirs, _ in os.walk(TARGET_ROOT, onerror=lambda e: None):
        rel = os.path.relpath(current, TARGET_ROOT)
        depth = 0 if rel == "." else len(rel.split(os.sep))
        subdirs[:] = [d for d in subdirs if d.lower() not in EXCLUDED_DIRS and not d.startswith((".", "$"))]
        if depth >= MAX_DEPTH:
            subdirs[:] = []
        if depth > 0:
            result.append(current)
    return {"target_folders": result}


def scan_node(state: OrganizerState) -> dict:
    found: list[FileItem] = []
    for folder in state["folders"]:
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            extension = os.path.splitext(entry.name)[1].lower()
            if entry.name.lower() in SKIP_FILES or extension in SKIP_EXTENSIONS:
                continue
            found.append({"path": entry.path,
                          "size_kb": round(entry.stat().st_size / 1024, 1),
                          "extension": extension})
    return {"files": found[:MAX_CANDIDATES]}


def classify_file(item: FileItem, folders: list[str], prompt: str = PROMPT_V1) -> FileItem:
    name = os.path.basename(item["path"])
    result = classifier.invoke(prompt.format(name=name, folders="\n".join(folders), **item))
    target = result.target_folder if result.target_folder in folders else "unsure"
    return {**item, "target_folder": target, "reason": result.reason}


def classify_node(state: OrganizerState) -> dict:
    return {"files": [classify_file(f, state["target_folders"]) for f in state["files"]]}


def route_after_classify(state: OrganizerState) -> str:
    if any(f.get("target_folder", "unsure") != "unsure" for f in state["files"]):
        return "approve"
    return "report"


def target_path(item: FileItem) -> str:
    return os.path.join(item["target_folder"], os.path.basename(item["path"]))


def approve_node(state: OrganizerState) -> dict:
    to_move = [{**f, "target": target_path(f)} for f in state["files"]
               if f.get("target_folder", "unsure") != "unsure"]
    answer = interrupt({"files": to_move})
    approved = [to_move[i]["path"] for i in answer if 0 <= i < len(to_move)]
    return {"approved": approved}


def move_node(state: OrganizerState) -> dict:
    by_path = {f["path"]: f for f in state["files"]}
    moved = []
    for path in state["approved"]:
        target = target_path(by_path[path])
        if os.path.exists(target):
            continue
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.move(path, target)
            moved.append(path)
        except OSError:
            continue
    return {"moved": moved}


def report_node(state: OrganizerState) -> dict:
    unsure = sum(1 for f in state["files"] if f.get("target_folder", "unsure") == "unsure")
    return {"report": f"נסרקו {len(state['files'])} קבצים. לא נמצאה תיקייה ל-{unsure}. הועברו: {len(state.get('moved', []))}."}


builder = StateGraph(OrganizerState)
builder.add_node("load_folders", load_folders_node)
builder.add_node("scan", scan_node)
builder.add_node("classify", classify_node)
builder.add_node("approve", approve_node)
builder.add_node("move", move_node)
builder.add_node("report", report_node)
builder.add_edge(START, "load_folders")
builder.add_edge("load_folders", "scan")
builder.add_edge("scan", "classify")
builder.add_conditional_edges("classify", route_after_classify, {"approve": "approve", "report": "report"})
builder.add_edge("approve", "move")
builder.add_edge("move", "report")
builder.add_edge("report", END)
graph = builder.compile(checkpointer=MemorySaver())

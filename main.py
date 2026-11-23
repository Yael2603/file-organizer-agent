import os
import sys
from langgraph.types import Command
from graph import graph


def default_folders() -> list[str]:
    home = os.path.expanduser("~")
    candidates = [os.path.join(home, "Desktop"),
                  os.path.join(home, "OneDrive", "Desktop"),
                  os.path.join(home, "Downloads")]
    return [p for p in candidates if os.path.isdir(p)]


def main():
    folders = sys.argv[1:] or default_folders()
    config = {"configurable": {"thread_id": "run-1"}}
    result = graph.invoke({"folders": folders, "approved": [], "moved": []}, config)
    while "__interrupt__" in result:
        files = result["__interrupt__"][0].value["files"]
        for i, f in enumerate(files):
            print(f"[{i}] {os.path.basename(f['path'])} -> {f['target_folder']}  ({f['reason']})")
        raw = input("מספרי הקבצים להעברה, מופרדים בפסיק (Enter = כלום): ")
        chosen = [int(x) for x in raw.split(",") if x.strip().isdigit()]
        result = graph.invoke(Command(resume=chosen), config)
    print(result["report"])


if __name__ == "__main__":
    main()

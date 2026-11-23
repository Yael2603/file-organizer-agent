import os
from langsmith import Client
from graph import classify_file, PROMPT_V1

client = Client()
DATASET = "file-organizer-eval"
ALGO, OS_, CV, IMG, INST = (r"D:\לימודים\אלגוריתמים", r"D:\לימודים\מערכות הפעלה",
                            r"D:\עבודה\קורות חיים", r"D:\תמונות", r"D:\התקנות")
FOLDERS = [ALGO, OS_, CV, IMG, INST]
EXAMPLES = [
    ("מצגת חמדני.pptx", ALGO), ("Dijkstra summary.pdf", ALGO), ("תרגיל 3 - קוד האפמן.docx", ALGO),
    ("round_robin_scheduling.docx", OS_), ("deadlock.pptx", OS_), ("סיכום תהליכים וזיכרון.pdf", OS_),
    ("cv_2026.pdf", CV), ("קורות חיים מעודכן.docx", CV),
    ("IMG_2034.jpg", IMG), ("WhatsApp Image 2026-11-02.jpeg", IMG),
    ("python-3.12.msi", INST), ("קבלה חשמל.pdf", "unsure"),
]

PROMPT_V2 = PROMPT_V1 + """
Rules:
- Match by the subject of the file name: topic keywords (in Hebrew or English) that belong to a folder's subject area go to that folder, even if the folder name does not appear in the file name.
- Use the extension for media files and installers.
- If the name gives no clear clue, answer "unsure". Do not guess.
- Copy the folder path exactly, character by character."""


def make_target(prompt: str):
    def target(inputs: dict) -> dict:
        name = inputs["file_name"]
        item = {"path": name, "size_kb": 100.0, "extension": os.path.splitext(name)[1].lower()}
        return {"target_folder": classify_file(item, FOLDERS, prompt)["target_folder"]}
    return target


def correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["target_folder"] == reference_outputs["target_folder"]


if __name__ == "__main__":
    if not client.has_dataset(dataset_name=DATASET):
        ds = client.create_dataset(DATASET)
        client.create_examples(dataset_id=ds.id, examples=[
            {"inputs": {"file_name": n}, "outputs": {"target_folder": t}} for n, t in EXAMPLES])
    client.evaluate(make_target(PROMPT_V1), data=DATASET, evaluators=[correct], experiment_prefix="before")
    client.evaluate(make_target(PROMPT_V2), data=DATASET, evaluators=[correct], experiment_prefix="after")

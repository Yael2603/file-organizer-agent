from typing import TypedDict, NotRequired


class FileItem(TypedDict):
    path: str
    size_kb: float
    extension: str
    target_folder: NotRequired[str]
    reason: NotRequired[str]


class OrganizerState(TypedDict):
    folders: list[str]
    target_folders: list[str]
    files: list[FileItem]
    approved: list[str]
    moved: list[str]
    report: str

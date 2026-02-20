from typing import Any, Dict, List, Union

from rich.console import Console
from rich.panel import Panel

console = Console()


def _summarize_image_part(part: Dict[str, Any], index: int) -> str:
    mime_type = part.get("mime_type", "image/png")
    data = part.get("data") or part.get("base64") or ""
    size = len(data)
    preview = f"{data[:24]}..." if data else "N/A"
    return f"[Image {index} | mime={mime_type} | base64_len={size} | preview={preview}]"


def format_message_content(content: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content)
    segments: List[str] = []
    image_counter = 1
    for part in content or []:
        part_type = part.get("type")
        if part_type == "text":
            segments.append(part.get("text", ""))
        elif part_type == "image_base64":
            segments.append(_summarize_image_part(part, image_counter))
            image_counter += 1
        else:
            segments.append(str(part))
    return "\n".join(filter(None, segments))


def print_user_panel(content: Union[str, List[Dict[str, Any]], Dict[str, Any]], title: str = "User → Model") -> None:
    console.print(Panel(format_message_content(content), title=title, style="cyan"))


def print_assistant_panel(text: str, title: str = "Model → Assistant") -> None:
    # Escape Rich markup tags by doubling brackets to prevent parsing errors
    # This handles cases where LLM output contains bracket tags like [/Reasoning]
    # Rich uses [text] for markup, so [ becomes [[ and ] becomes ]] to escape
    try:
        # Escape brackets to prevent Rich from trying to parse them as markup
        escaped_text = (text.strip() if text else "").replace("[", "[[").replace("]", "]]")
        console.print(Panel(escaped_text, title=title, style="red"))
    except Exception:
        # Fallback: just print as plain text without panel if anything goes wrong
        console.print(f"{title}: {text.strip() if text else ''}")



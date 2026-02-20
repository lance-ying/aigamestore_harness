import base64
import copy
import io
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.llm_interface_utils.rich_logging import print_user_panel, print_assistant_panel
from PIL import Image

logger = logging.getLogger(__name__)

SUMMARY_PATTERN = re.compile(r"<summary>(.*?)</summary>", re.IGNORECASE | re.DOTALL)
SCRATCHPAD_PATTERN = re.compile(r"<scratchpad>(.*?)</scratchpad>", re.IGNORECASE | re.DOTALL)


def append_history_entry(entry: Dict[str, Any], history: List[Dict[str, Any]], full_history: List[Dict[str, Any]]) -> None:
    """Append an entry to both the active history and the archival history."""
    history.append(entry)
    full_history.append(copy.deepcopy(entry))


def extract_summary_text(text: str) -> Optional[str]:
    """Return the <summary>...</summary> block from the supplied text, if present."""
    if not text:
        return None
    match = SUMMARY_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_scratchpad_text(text: str) -> Optional[str]:
    """Return the <scratchpad>...</scratchpad> block from the supplied text, if present."""
    if not text:
        return None
    match = SCRATCHPAD_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def save_screenshot(results_dir: str, episode_count: int, step_count: int, screenshot_bytes: bytes, suffix: Optional[str] = None) -> str:
    """Persist a screenshot PNG and return its path."""
    try:
        suffix_part = f"_{suffix}" if suffix else ""
        filename = f"episode_{episode_count}_step_{step_count}{suffix_part}.png"
        filepath = os.path.join(results_dir, "screenshots", filename)
        with open(filepath, "wb") as file:
            file.write(screenshot_bytes)
        return filepath
    except Exception as exc:
        logger.error(f"Error saving screenshot: {exc}")
        return ""


def save_action_frame(gameplay_dir: str, frame_idx: int, screenshot_bytes: bytes) -> str:
    """Persist a post-action gameplay frame PNG and return its path."""
    try:
        filename = f"frame_step_{frame_idx:05d}.png"
        filepath = os.path.join(gameplay_dir, filename)
        with open(filepath, "wb") as file:
            file.write(screenshot_bytes)
        return filepath
    except Exception as exc:
        logger.error(f"Error saving action frame: {exc}")
        return ""


def build_frame_reference(path: Optional[str], image_bytes: Optional[bytes], mime_type: str = "image/png") -> Optional[Dict[str, Optional[str]]]:
    """Create a serializable descriptor containing the image path and base64 payload."""
    if not path and not image_bytes:
        return None
    base64_data = base64.b64encode(
        _prepare_image_for_base64(image_bytes, mime_type)
    ).decode("utf-8") if image_bytes else None
    return {
        "path": path,
        "base64": base64_data,
        "mime_type": mime_type,
    }


def _prepare_image_for_base64(image_bytes: bytes, mime_type: str) -> bytes:
    """Downscale the image so the smaller dimension is at most 256px before encoding."""
    if not image_bytes:
        return image_bytes
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            min_dim = min(width, height)
            if min_dim <= 256:
                return image_bytes
            scale = 256 / float(min_dim)
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            resized = img.resize(new_size, Image.LANCZOS)
            buffer = io.BytesIO()
            format_name = "JPEG" if mime_type.lower() in ("image/jpeg", "image/jpg") else "PNG"
            resized.save(buffer, format=format_name)
            return buffer.getvalue()
    except Exception as exc:
        logger.warning(f"Failed to resize image for base64 encoding: {exc}")
        return image_bytes


def save_prompt(results_dir: str, step_count: int, episode_count: int, content: List[Dict[str, Any]]) -> str:
    """
    Save the full prompt content before sending to the model.
    
    Args:
        results_dir: Directory to save the prompt
        step_count: Current step number
        episode_count: Current episode number
        content: The prompt content list (text and images)
    
    Returns:
        Path to the saved prompt file
    """
    prompts_dir = os.path.join(results_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    
    prompt_path = os.path.join(prompts_dir, f"episode_{episode_count}_step_{step_count}_prompt.json")
    
    # Serialize the prompt content
    serializable_content = []
    for item in content:
        item_type = item.get("type")
        if item_type == "text":
            serializable_content.append({
                "type": "text",
                "text": item.get("text", "")
            })
        elif item_type == "image_base64":
            # Base64 image - save metadata but truncate base64 data for readability
            base64_data = item.get("data", "")
            mime_type = item.get("mime_type", "image/png")
            serializable_content.append({
                "type": "image_base64",
                "mime_type": mime_type,
                "base64_length": len(base64_data) if base64_data else 0,
                "base64_preview": base64_data[:100] + "..." if base64_data and len(base64_data) > 100 else base64_data,
                "note": "Full base64 data truncated for readability"
            })
        elif item_type == "image_url" or item_type == "image":
            # For images, save the path or base64 info
            image_info = item.get("image_url", item.get("image", {}))
            if isinstance(image_info, dict):
                url = image_info.get("url", "")
                if url.startswith("data:image"):
                    # Base64 image - truncate for readability
                    serializable_content.append({
                        "type": "image_url",
                        "format": "base64_data_url",
                        "mime_type": image_info.get("mime_type", "unknown"),
                        "url_preview": url[:200] + "..." if len(url) > 200 else url,
                        "note": "Base64 image data (truncated in JSON)"
                    })
                else:
                    # File path
                    serializable_content.append({
                        "type": "image_url",
                        "path": url
                    })
            else:
                serializable_content.append({
                    "type": "image_url",
                    "data": str(image_info)[:100] + "..." if len(str(image_info)) > 100 else str(image_info)
                })
        else:
            # Unknown type, save as-is (with truncation for large data)
            serializable_content.append({
                "type": item.get("type", "unknown"),
                "data": str(item)[:500] + "..." if len(str(item)) > 500 else str(item)
            })
    
    prompt_data = {
        "episode": episode_count,
        "step": step_count,
        "timestamp": datetime.now().isoformat(),
        "content": serializable_content
    }
    
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompt_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved prompt to {prompt_path}")
    return prompt_path


def save_results(results_dir: str, content_history: List[Dict[str, Any]], gameplay_log: List[Dict[str, Any]]) -> None:
    """Write content history and gameplay logs to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    content_path = os.path.join(results_dir, f"content_history_{timestamp}.json")
    with open(content_path, "w") as content_file:
        serializable_history = json.loads(json.dumps(content_history, default=str))
        json.dump(serializable_history, content_file, indent=2)

    gameplay_path = os.path.join(results_dir, f"gameplay_log_{timestamp}.json")
    with open(gameplay_path, "w") as log_file:
        json.dump(gameplay_log, log_file, indent=2)

    logger.info(f"Results saved to {results_dir}")


def record_gameplay_step(
    gameplay_dir: str,
    episode: int,
    step: int,
    actions: List[Optional[int]],
    llm_output: str,
    screenshot: str,
    action_snapshots: List[Dict[str, Any]],
    token_usage: Optional[Dict[str, Any]] = None,
    score: Optional[str] = None,
    model_name: Optional[str] = None,
    game_name: Optional[str] = None,
) -> None:
    """Persist a single gameplay step to disk for detailed auditing."""
    step_payload = {
        "episode": episode,
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "actions": actions,
        "llm_output": llm_output,
        "screenshot": screenshot,
        "action_snapshots": action_snapshots,
    }
    if model_name:
        step_payload["model_name"] = model_name
    if game_name:
        step_payload["game_name"] = game_name
    if score:
        step_payload["score"] = score
    if token_usage:
        step_payload["token_usage"] = token_usage
    step_path = os.path.join(gameplay_dir, f"episode_{episode}_step_{step}.json")
    with open(step_path, "w") as step_file:
        json.dump(step_payload, step_file, indent=2)



"""
VLM Gameplay Script using Playwright

"""

import argparse
import time
import os
import logging
import glob
import re
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path
from PIL import Image

from dotenv import load_dotenv
# Load .env file from the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'))

from playwright.sync_api import sync_playwright, Page, Locator

from llm_interface.api import OpenAIInterface, AnthropicInterface, GoogleInterface, TogetherAIInterface, XAIInterface
from utils.parsing_utils import parse_actions, KEY_MAPPING
from utils.gameplay_utils import (
    append_history_entry,
    extract_scratchpad_text,
    save_screenshot,
    save_action_frame,
    build_frame_reference,
    save_results,
    record_gameplay_step,
    save_prompt,
)
from utils.llm_interface_utils.rich_logging import print_user_panel, print_assistant_panel

REVERSE_KEY_MAPPING = {value: key for key, value in KEY_MAPPING.items()}
REVERSE_KEY_MAPPING[None] = "NOOP"

# Mapping from key codes to Playwright key names
KEY_CODE_TO_PLAYWRIGHT_KEY = {
    37: "ArrowLeft",
    38: "ArrowUp",
    39: "ArrowRight",
    40: "ArrowDown",
    32: " ",
    13: "Enter",
    27: "Escape",
    # Letter keys A-Z (65-90)
    65: "a", 66: "b", 67: "c", 68: "d", 69: "e", 70: "f",
    71: "g", 72: "h", 73: "i", 74: "j", 75: "k", 76: "l",
    77: "m", 78: "n", 79: "o", 80: "p", 81: "q", 82: "r",
    83: "s", 84: "t", 85: "u", 86: "v", 87: "w", 88: "x",
    89: "y", 90: "z",
}


def actions_to_names(actions: List[Optional[int]]) -> str:
    """Convert key codes to human-readable names for logging/prompts."""
    if not actions:
        return "None"
    names = []
    for action in actions:
        names.append(REVERSE_KEY_MAPPING.get(action, str(action)))
    return ", ".join(names)


def format_frame_description(entry: Dict[str, Any]) -> str:
    """Produce a compact textual tag for a stored frame."""
    episode = entry.get("episode")
    step = entry.get("step")
    action_index = entry.get("action_index")
    action_name = entry.get("action_name", "UNKNOWN")
    return (
        f"Episode {episode}, Step {step}, Action #{action_index} ({action_name}) | "
    )


def append_frame_visual(sections: List[Dict[str, Any]], label: str, frame_ref: Optional[Dict[str, Optional[str]]]) -> None:
    """Attach a labeled image payload to the provided message sections."""
    if not frame_ref:
        return
    base64_data = frame_ref.get("base64")
    if not base64_data:
        return
    mime_type = frame_ref.get("mime_type", "image/png")
    sections.append({"type": "text", "text": label})
    sections.append({
        "type": "image_base64",
        "data": base64_data,
        "mime_type": mime_type
    })


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="VLM Gameplay Script (Playwright) - AI Gamestore harness")
    parser.add_argument("--model", type=str, required=True, help="Model in format 'provider:model_name' (e.g., 'openai:gpt-4o' or 'google:gemini-2.0-flash')")
    parser.add_argument("--url", type=str, required=True, help="Full game URL")
    parser.add_argument("--max_seconds", type=int, default=120, help="Maximum number of successful API calls")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    return parser.parse_args()


def parse_model_string(model_string: str) -> Tuple[str, str]:
    """Parse model string in format 'provider:model_name' into provider and model name."""
    if ":" not in model_string:
        raise ValueError(f"Model string must be in format 'provider:model_name', got: {model_string}")
    parts = model_string.split(":", 1)
    provider = parts[0].strip().lower()
    model_name = parts[1].strip()
    if not provider or not model_name:
        raise ValueError(f"Both provider and model name must be specified, got: {model_string}")
    return provider, model_name


def parse_game_name_from_url(url: str) -> str:
    """Extract game name from URL (e.g., 'http://localhost:8080/games/gvgai_aliens/0.html' -> 'gvgai_aliens').
    Also handles variants_final format: 'http://localhost:8080/variants_final/[Doodle-Hopper]_v1' -> 'doodle-hopper_v1'."""
    try:
        parsed = urlparse(url)
        path = parsed.path
        
        # Handle variants_final format: /variants_final/[game-name]_v1
        if '/variants_final/' in path:
            path_after_variants = path.split('/variants_final/')[-1]
            folder_name = path_after_variants.split('/')[0]
            # Extract game name from [game-name]_v1 format
            if folder_name.startswith('[') and ']' in folder_name:
                # Extract name from brackets
                bracket_end = folder_name.find(']')
                game_name_in_brackets = folder_name[1:bracket_end]
                # Convert to lowercase and replace spaces with hyphens
                game_name = game_name_in_brackets.lower().replace(' ', '-')
                # Get the suffix after the bracket (e.g., '_v1')
                suffix = folder_name[bracket_end + 1:]
                # Combine: game-name_v1
                if suffix:
                    return f"{game_name}{suffix}"
                else:
                    return game_name
        
        # Handle regular games format: /games/game_name
        if '/games/' in path:
            path_after_games = path.split('/games/')[-1]
            game_name = path_after_games.split('/')[0]
            if game_name:
                return game_name
        
        logger.warning(f"Could not parse game name from URL path '{path}' (full URL: {url}), using 'unknown_game'")
        return "unknown_game"
    except Exception as e:
        logger.warning(f"Error parsing game name from URL {url}: {e}, using 'unknown_game'")
        return "unknown_game"


def get_game_name_from_page(page: Page) -> Optional[str]:
    """Read game name from the webpage: meta tag name='game_name' or element #game_name."""
    try:
        meta = page.locator('meta[name="game_name"]').first
        content = meta.get_attribute("content")
        if content and content.strip():
            return content.strip()
    except Exception:
        pass
    try:
        el = page.locator("#game_name").first
        if el.is_visible():
            text = el.inner_text()
            if text and text.strip():
                return text.strip()
    except Exception:
        pass
    return None


def get_interface(model_provider: str, model_name: str):
    if model_provider == "openai":
        return OpenAIInterface(model_name=model_name, enable_thinking=True)
    elif model_provider == "anthropic":
        return AnthropicInterface(model_name=model_name, enable_thinking=True)
    elif model_provider == "gemini" or model_provider == "google":
        return GoogleInterface(model_name=model_name, enable_thinking=True)
    elif model_provider == "togetherai" or model_provider == "together":
        return TogetherAIInterface(model_name=model_name, enable_thinking=True)
    elif model_provider == "xai":
        return XAIInterface(model_name=model_name, enable_thinking=True)
    else:
        raise ValueError(f"Unknown model provider: {model_provider}")


def find_canvas(page: Page) -> Locator:
    """Find the canvas element on the page or in iframes."""
    # Try directly on the main page
    try:
        canvas = page.locator("#defaultCanvas0").first
        canvas.wait_for(state="visible", timeout=2000)
        logger.info("Found canvas on main page")
        return canvas
    except Exception:
        pass

    # Search across all frames
    for frame in page.frames:
        try:
            frame_canvas = frame.locator("#defaultCanvas0").first
            frame_canvas.wait_for(state="visible", timeout=2000)
            logger.info("Found canvas in iframe")
            return frame_canvas
        except Exception:
            continue

    # Fallback: try any canvas
    try:
        canvas = page.locator("canvas").first
        canvas.wait_for(state="visible", timeout=2000)
        logger.info("Found canvas (fallback)")
        return canvas
    except Exception:
        raise RuntimeError("Canvas element not found on page or within iframes")


def execute_action_playwright(canvas: Locator, action_code: Optional[int]) -> None:
    """Execute a single action (instant keypress) in the game using Playwright."""
    if action_code is None:
        # NOOP action - do nothing
        return
    
    try:
        # Focus on the canvas to ensure keypresses are received
        canvas.click()
        time.sleep(0.05)  # Small delay to ensure focus
        
        # Get the key to send
        if action_code in KEY_CODE_TO_PLAYWRIGHT_KEY:
            key = KEY_CODE_TO_PLAYWRIGHT_KEY[action_code]
            canvas.press(key)
        else:
            logger.warning(f"Unknown key code: {action_code}")
            
    except Exception as e:
        logger.error(f"Error executing action {action_code}: {e}")


def execute_hold_actions_playwright(canvas: Locator, key_codes: List[int], duration: float = 0.2) -> None:
    """Execute multiple HOLD actions simultaneously by sending all keydowns, waiting, then all keyups using Playwright."""
    if not key_codes:
        return
    
    try:
        # Focus on the canvas
        canvas.click()
        # Get the page from the canvas locator
        page = canvas.page
        
        # Convert key codes to Playwright key names
        keys = []
        for key_code in key_codes:
            if key_code in KEY_CODE_TO_PLAYWRIGHT_KEY:
                keys.append(KEY_CODE_TO_PLAYWRIGHT_KEY[key_code])
            else:
                logger.warning(f"Unknown key code for HOLD: {key_code}")
        
        if not keys:
            return
        
        # Send all keydowns simultaneously
        for key in keys:
            page.keyboard.down(key)
        
        # Wait for the duration (all keys are held during this time)
        time.sleep(duration)
        
        # Send all keyups simultaneously
        for key in keys:
            page.keyboard.up(key)
            
    except Exception as e:
        logger.error(f"Error executing HOLD actions {key_codes}: {e}")




def get_game_screenshot_playwright(canvas: Locator) -> bytes:
    """Get PNG bytes of the game canvas using Playwright."""
    try:
        screenshot_bytes = canvas.screenshot()
        return screenshot_bytes
    except Exception as e:
        logger.error(f"Error getting screenshot: {e}")
        return b""


def get_score_from_game_state(page: Page) -> Optional[float]:
    """
    Extract score directly from the game state using Playwright.
    Uses the iframe's contentWindow when present; else the window that owns the canvas (document.querySelector('canvas').ownerDocument.defaultView); else window.
    Tries: getGameState(), gameInstance.gameState, gameState.
    Returns score as float, or None if not found.
    """
    try:
        score = page.evaluate("""
            () => {
                const iframe = document.querySelector('iframe');
                let gameWin = window;
                if (iframe && iframe.contentWindow) {
                    gameWin = iframe.contentWindow;
                } else {
                    const canvas = document.querySelector('canvas');
                    if (canvas && canvas.ownerDocument && canvas.ownerDocument.defaultView) {
                        gameWin = canvas.ownerDocument.defaultView;
                    }
                }
                if (typeof gameWin.getGameState === 'function') {
                    const state = gameWin.getGameState();
                    if (state && typeof state.score !== 'undefined') {
                        return state.score;
                    }
                }
                if (gameWin.gameInstance && gameWin.gameInstance.gameState) {
                    const state = gameWin.gameInstance.gameState;
                    if (typeof state.score !== 'undefined') {
                        return state.score;
                    }
                }
                if (gameWin.gameState && typeof gameWin.gameState.score !== 'undefined') {
                    return gameWin.gameState.score;
                }
                return null;
            }
        """)
        
        if score is not None:
            try:
                return float(score)
            except (ValueError, TypeError):
                logger.debug(f"Score value is not numeric: {score}")
                return None
        
        return None
        
    except Exception as e:
        logger.debug(f"Error reading score from game state: {e}")
        return None


def check_game_ended(page: Page) -> bool:
    """
    Check if the game has ended. Uses iframe's contentWindow when present; else the window that owns the canvas; else window.
    Tries: game.ended, getGameState().gamePhase, gameInstance.gameState.gamePhase.
    """
    try:
        ended = page.evaluate("""
            () => {
                const iframe = document.querySelector('iframe');
                let gameWin = window;
                if (iframe && iframe.contentWindow) {
                    gameWin = iframe.contentWindow;
                } else {
                    const canvas = document.querySelector('canvas');
                    if (canvas && canvas.ownerDocument && canvas.ownerDocument.defaultView) {
                        gameWin = canvas.ownerDocument.defaultView;
                    }
                }
                if (gameWin.game && typeof gameWin.game.ended !== 'undefined') {
                    if (gameWin.game.ended === true) return true;
                }
                if (typeof gameWin.getGameState === 'function') {
                    const state = gameWin.getGameState();
                    if (state && state.gamePhase) {
                        const gamePhase = state.gamePhase;
                        if (gamePhase === 'GAME_OVER' || gamePhase === 'GAME_OVER_LOSE' ||
                            gamePhase === 'GAME_OVER_WIN' || gamePhase === 'ENDED') {
                            return true;
                        }
                    }
                }
                if (gameWin.gameInstance && gameWin.gameInstance.gameState) {
                    const gamePhase = gameWin.gameInstance.gameState.gamePhase;
                    if (gamePhase === 'GAME_OVER' || gamePhase === 'GAME_OVER_LOSE' ||
                        gamePhase === 'GAME_OVER_WIN' || gamePhase === 'ENDED') {
                        return true;
                    }
                }
                return false;
            }
        """)
        result = bool(ended)
        if result:
            logger.info("Game end detected via window.game.ended or gamePhase")
        return result
    except Exception as e:
        logger.warning(f"Error checking if game ended: {e}")
        return False


def restart_game(canvas: Locator, page: Page) -> None:
    """
    Restart the game by pressing 'R' key (restart) followed by Enter key (start).
    Uses multiple methods to ensure the key press is actually executed.
    """
    try:
        logger.info("Attempting to restart game...")
        
        # Method 1: Focus canvas and use canvas.press
        try:
            logger.info("Method 1: Using canvas.press('r')...")
            canvas.click()
            time.sleep(0.15)
            canvas.press('r')
            logger.info("Successfully sent 'R' via canvas.press")
        except Exception as e:
            logger.warning(f"canvas.press('r') failed: {e}")
        
        time.sleep(0.3)
        
        # Method 2: Use page.keyboard directly (more reliable)
        try:
            logger.info("Method 2: Using page.keyboard.press('r')...")
            page.keyboard.press('r')
            logger.info("Successfully sent 'R' via page.keyboard.press")
        except Exception as e:
            logger.warning(f"page.keyboard.press('r') failed: {e}")
        
        time.sleep(0.4)  # Wait for restart to process
        
        # Press Enter after 'R' to start the game - try multiple methods
        logger.info("Pressing Enter key to start...")
        
        # Method 1: Use execute_action_playwright
        try:
            execute_action_playwright(canvas, 13)  # Enter key
            logger.info("Successfully sent Enter via execute_action_playwright")
        except Exception as e:
            logger.warning(f"execute_action_playwright(Enter) failed: {e}")
        
        # Method 2: Use page.keyboard directly
        try:
            page.keyboard.press('Enter')
            logger.info("Successfully sent Enter via page.keyboard.press")
        except Exception as e:
            logger.warning(f"page.keyboard.press('Enter') failed: {e}")
        
        time.sleep(0.5)  # Wait for game to start
        
        # Verify restart by checking if game is no longer ended
        game_still_ended = check_game_ended(page)
        if game_still_ended:
            logger.warning("Game still appears ended after restart attempt")
            # Try one more aggressive restart
            logger.info("Trying aggressive restart with keydown/keyup...")
            try:
                canvas.click()
                time.sleep(0.1)
                # Use keydown/keyup for more control
                page.keyboard.down('r')
                time.sleep(0.05)
                page.keyboard.up('r')
                time.sleep(0.3)
                page.keyboard.down('Enter')
                time.sleep(0.05)
                page.keyboard.up('Enter')
                time.sleep(0.4)
                logger.info("Completed aggressive restart attempt")
            except Exception as e2:
                logger.warning(f"Aggressive restart also failed: {e2}")
        
        logger.info("Game restart sequence completed")
    except Exception as e:
        logger.error(f"Error in restart_game: {e}")
        # Final fallback: try page.keyboard directly
        try:
            logger.info("Final fallback: Using page.keyboard only...")
            page.keyboard.press('r')
            time.sleep(0.3)
            page.keyboard.press('Enter')
            time.sleep(0.4)
        except Exception as e2:
            logger.error(f"Final fallback also failed: {e2}")




def create_gameplay_gif(screenshot_dir: str, output_path: Optional[str] = None, duration: int = 50) -> Optional[str]:
    """
    Create a GIF from all screenshots in the specified directory.
        
        Args:
        screenshot_dir: Directory containing frame screenshots
        output_path: Optional path for the output GIF (defaults to parent_dir/gameplay.gif)
        duration: Duration of each frame in milliseconds (default: 50ms for 20 FPS)
            
        Returns:
        Path to the created GIF file, or None if creation failed
    """
    try:
        # Find all frame screenshots (sorted by filename)
        pattern = os.path.join(screenshot_dir, "frame_*.png")
        image_files = sorted(glob.glob(pattern))
        
        if not image_files:
            logger.warning(f"No screenshots found in {screenshot_dir} to create GIF")
            return None
        
        logger.info(f"Found {len(image_files)} screenshots, creating GIF...")
        
        # Load images
        images = []
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                # Convert to RGB if necessary (GIFs don't support RGBA)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue
        
        if not images:
            logger.error("No valid images loaded for GIF creation")
            return None
        
        # Set output path
        if output_path is None:
            # Use parent directory for output
            parent_dir = os.path.dirname(screenshot_dir)
            output_path = os.path.join(parent_dir, "gameplay", "gameplay.gif")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0  # Loop forever
        )
        
        logger.info(f"GIF created successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating GIF: {e}")
        return None


def run_one_game(
    game_url: str,
    args: Any,
    model_provider: str,
    model_name: str,
    prompt_template: str,
    results_base: str,
    session_id: str,
    sanitized_model_name: str,
) -> None:
    """Run the full gameplay loop for a single game URL."""
    frame_index = 0
    temp_frame_index = 0
    frame_history: List[Dict[str, Any]] = []
    current_scratchpad_text: Optional[str] = None
    last_executed_actions: List[Optional[int]] = []
    last_render_frames: List[Dict[str, Any]] = []
    
    last_screenshot_time = None
    screenshot_interval = 1.0 / 20.0
    
    interface = get_interface(model_provider, model_name)
    
    content_history: List[Dict[str, Any]] = []
    full_content_history: List[Dict[str, Any]] = []
    gameplay_log: List[Dict[str, Any]] = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page()
        
        try:
            page.goto(game_url, wait_until="domcontentloaded")
            logger.info(f"Navigated to {game_url}")
            
            # Get game name from page (tag "game_name") or fall back to URL
            game_name = get_game_name_from_page(page) or parse_game_name_from_url(game_url)
            logger.info(f"Game name: {game_name}")
            folder_name = f"{session_id}_{sanitized_model_name}_{game_name}"
            results_dir = os.path.join(results_base, folder_name)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(os.path.join(results_dir, "screenshots"), exist_ok=True)
            gameplay_dir = os.path.join(results_dir, "gameplay")
            os.makedirs(gameplay_dir, exist_ok=True)
            temp_screenshot_dir = os.path.join(results_dir, "temp_screenshots")
            os.makedirs(temp_screenshot_dir, exist_ok=True)
            
            # Find canvas
            canvas = find_canvas(page)
            
            # Ensure canvas can receive focus
            try:
                canvas.evaluate("el => { if (!el.hasAttribute('tabindex')) el.setAttribute('tabindex', '0'); }")
            except Exception:
                pass
            
            canvas.click()
            logger.info("Canvas focused and ready")
            
            # Press Enter key to start
            execute_action_playwright(canvas, 13)
            time.sleep(0.05)
            
            # Initialize screenshot capture timing
            last_screenshot_time = time.time()
            logger.info(f"Will capture screenshots at 20 FPS. Temp dir: {temp_screenshot_dir}")
            
            # Helper function to capture screenshot if enough time has passed
            def capture_screenshot_if_needed():
                """Capture a screenshot to temp folder if enough time has passed (20 FPS)."""
                nonlocal last_screenshot_time, temp_frame_index
                current_time = time.time()
                if last_screenshot_time is None or (current_time - last_screenshot_time) >= screenshot_interval:
                    try:
                        screenshot_bytes = get_game_screenshot_playwright(canvas)
                        if screenshot_bytes and len(screenshot_bytes) > 0:
                            frame_filename = f"frame_{temp_frame_index:06d}.png"
                            frame_path = os.path.join(temp_screenshot_dir, frame_filename)
                            with open(frame_path, 'wb') as f:
                                f.write(screenshot_bytes)
                            temp_frame_index += 1
                            if temp_frame_index % 100 == 0:
                                logger.info(f"Captured {temp_frame_index} screenshots so far")
                            last_screenshot_time = current_time
                    except Exception as e:
                        logger.debug(f"Screenshot capture error (non-critical): {e}")
            
            # Get game description from page (required)
            game_description = ""
            try:
                game_description_element = page.locator("#gameDescription").first
                if game_description_element.is_visible():
                    game_description = game_description_element.inner_text()
                    if game_description:
                        logger.info(f"Game description: {game_description}")
                    else:
                        raise ValueError("Game description element #gameDescription is empty")
                else:
                    raise ValueError("Game description element #gameDescription is not visible")
            except Exception as e:
                logger.error(f"Missing game description: {e}")
                raise
            
            # Get game controls text (required)
            game_controls_text = ""
            try:
                game_controls_element = page.locator("#gameControls").first
                if game_controls_element.is_visible():
                    game_controls_text = game_controls_element.inner_text()
                    if not (game_controls_text or "").strip():
                        raise ValueError("Game controls element #gameControls is empty")
                    logger.info(f"Game controls text: {game_controls_text}")
                else:
                    raise ValueError("Game controls element #gameControls is not visible")
            except Exception as e:
                logger.error(f"Missing game controls: {e}")
                raise
            
            # Ensure control information includes R for restart and Enter for starting
            game_controls_lower = game_controls_text.lower()
            additional_controls = []
            if "r" not in game_controls_lower or "restart" not in game_controls_lower:
                additional_controls.append("R - Restart the game")
            if "enter" not in game_controls_lower:
                additional_controls.append("Enter - Start the game")
            if additional_controls:
                game_controls_text += "\n\nAdditional Controls:\n" + "\n".join(additional_controls)
                logger.info(f"Added additional controls: {additional_controls}")
            
            total_actions_count = 0
            successful_api_calls = 0
            episode_count = 1
            step_count = 0
            
            # Cumulative token tracking
            cumulative_input_tokens = 0
            cumulative_output_tokens = 0
            cumulative_total_tokens = 0
            cumulative_reasoning_tokens = 0
            
            while successful_api_calls < args.max_seconds:
                # Check if page reloaded
                try:
                    canvas = find_canvas(page)
                except Exception:
                    time.sleep(1)
                    continue
                
                # Capture screenshot before pausing (20 FPS)
                capture_screenshot_if_needed()
                
                # Capture state
                screenshot_bytes = get_game_screenshot_playwright(canvas)
                screenshot_path = ""
                screenshot_frame = None
                if screenshot_bytes:
                    screenshot_path = save_screenshot(results_dir, episode_count, step_count, screenshot_bytes)
                    screenshot_frame = build_frame_reference(screenshot_path, screenshot_bytes)
                log_screenshot = os.path.relpath(screenshot_path, results_dir) if screenshot_path else ""
                
                # Check if game has ended before pausing
                game_ended_before_pause = check_game_ended(page)
                if game_ended_before_pause:
                    logger.info("Game has ended (detected before pause), will restart after LLM response")
                
                # Pause the game before querying LLM (Esc toggles pause)
                logger.info("Pausing game before LLM query (Esc)...")
                execute_action_playwright(canvas, 27)  # Press ESC to pause
                time.sleep(0.1)  # Wait for pause to take effect
                
                # Read score from game state
                current_score = get_score_from_game_state(page)
                if current_score is not None:
                    logger.info(f"Score read from game state: {current_score}")
                
                # Prepare message content by filling the prompt template
                game_description_block = f"\n\nGame Description:\n{game_description}" if game_description else ""
                game_control_block = f"\n\nGame Controls:\n{game_controls_text}" if game_controls_text else ""
                scratchpad_block = (
                    f"\n\n<scratchpad>\n{current_scratchpad_text}\n</scratchpad>" if current_scratchpad_text else ""
                )
                previous_actions_block = ""
                if last_executed_actions:
                    action_names = actions_to_names(last_executed_actions)
                    previous_actions_block = f"\n\nPrevious actions applied: {action_names}"
                available_actions_list = [
                    "UP", "DOWN", "LEFT", "RIGHT", "SPACE",
                    "HOLD_UP", "HOLD_DOWN", "HOLD_LEFT", "HOLD_RIGHT", "HOLD_SPACE",
                    "NOOP", "R", "ENTER",
                ]
                available_actions_block = "\n\nAvailable actions: " + ", ".join(available_actions_list)

                # Split prompt: context (with placeholders) before frames, output instructions after frames
                if "**Output:**" in prompt_template:
                    prompt_before_output, prompt_after_split = prompt_template.split("**Output:**", 1)
                    prompt_output_section = "**Output:**" + prompt_after_split
                else:
                    prompt_before_output = prompt_template
                    prompt_output_section = ""

                text_before_frames = (
                    prompt_before_output.replace("{GAME_DESCRIPTION}", game_description_block)
                    .replace("{GAME_CONTROL}", game_control_block)
                    .replace("{SCRATCHPAD}", scratchpad_block)
                    .replace("{PREVIOUS_ACTIONS_FRAMES}", previous_actions_block)
                    .replace("{AVAILABLE_ACTIONS}", available_actions_block)
                )

                visual_sections: List[Dict[str, Any]] = []
                if last_render_frames:
                    for frame_entry in last_render_frames:
                        label = format_frame_description(frame_entry)
                        append_frame_visual(visual_sections, label, frame_entry["frame"])
                if screenshot_frame:
                    append_frame_visual(
                        visual_sections,
                        "Current observed state before executing the next sequence of actions.",
                        screenshot_frame
                    )

                # Order: context text → frames → output instructions
                current_content = [{"type": "text", "text": text_before_frames}] + visual_sections
                if prompt_output_section:
                    current_content.append({"type": "text", "text": prompt_output_section})
                
                # Save the full prompt before sending to model
                save_prompt(results_dir, step_count, episode_count, current_content)
                
                print_user_panel(current_content)
                user_entry = {"role": "user", "content": current_content}
                append_history_entry(user_entry, content_history, full_content_history)
                
                logger.info("Requesting LLM response...")
                # Only send the current message, not the full history (to prevent token growth)
                response = interface.generate(messages=[user_entry])
                successful_api_calls += 1
                
                # Log response
                assistant_output_text = response['output']
                logger.info(f"LLM Response: {assistant_output_text}")
                try:
                    print_assistant_panel(assistant_output_text)
                except Exception as e:
                    # Ignore Rich markup parsing errors (e.g., malformed [/Reasoning] tags)
                    logger.debug(f"Error printing assistant panel (ignored): {e}")
                
                # Check if game has ended (check again after LLM response)
                game_ended = check_game_ended(page)
                if game_ended or game_ended_before_pause:
                    logger.info("=" * 80)
                    logger.info("GAME HAS ENDED - INITIATING RESTART")
                    logger.info("=" * 80)
                    # Increment episode count when game restarts
                    episode_count += 1
                    step_count = 0
                    # Clear frame history for new episode
                    frame_history = []
                    last_render_frames = []
                    # Resume game first (in case it's paused) before restarting
                    try:
                        logger.info("Resuming game before restart (Esc to unpause)...")
                        execute_action_playwright(canvas, 27)  # Press ESC to unpause
                        time.sleep(0.05)
                    except Exception as e:
                        logger.debug(f"Could not unpause before restart: {e}")
                    # Restart the game
                    restart_game(canvas, page)
                    # Re-find canvas in case page reloaded
                    try:
                        canvas = find_canvas(page)
                        canvas.click()
                        logger.info("Canvas re-found and focused after restart")
                    except Exception as e:
                        logger.warning(f"Could not re-find canvas after restart: {e}")
                    time.sleep(0.8)  # Wait longer for restart to take effect
                    
                    # Verify game is no longer ended
                    still_ended = check_game_ended(page)
                    if still_ended:
                        logger.warning("Game still appears ended after restart - may need manual intervention")
                    else:
                        logger.info("Game restart successful - game is now active")
                
                # Score is read from game state only (no LLM extraction)
                if current_score is None:
                    logger.debug("No score found in game state")
                
                # Extract token usage from response
                token_usage = response.get('token_usage', {})
                input_tokens = token_usage.get('input_tokens', 0)
                output_tokens = token_usage.get('output_tokens', 0)
                total_tokens = token_usage.get('total_tokens', 0)
                reasoning_tokens = token_usage.get('reasoning_tokens', 0)  # For OpenAI reasoning models
            
                
                # Parse actions
                actions = parse_actions(assistant_output_text)
                
                # Add assistant response to history (for archival, but not sent to model)
                assistant_entry = {
                    "role": "assistant",
                    "content": response['output']
                }
                append_history_entry(assistant_entry, content_history, full_content_history)
                
                # Clear content_history after each step to prevent token accumulation
                # (We keep full_content_history for archival purposes)
                content_history.clear()
                
                # Extract scratchpad from response
                scratchpad_text = extract_scratchpad_text(assistant_output_text)
                if scratchpad_text:
                    current_scratchpad_text = scratchpad_text
                    logger.info(f"Extracted scratchpad: {scratchpad_text[:100]}...")
                elif current_scratchpad_text is None:
                    # If no scratchpad found and we don't have one yet, use the full response
                    current_scratchpad_text = assistant_output_text
                    logger.warning("No scratchpad found in response, using full response as scratchpad")
                
                action_snapshots: List[Dict[str, Any]] = []
                executed_actions: List[Optional[int]] = []

                # Resume the game before executing actions
                logger.info("Resuming game to execute actions...")
                execute_action_playwright(canvas, 27)  # Press ESC to resume
                time.sleep(0.2)  # Wait for resume to take effect

                # Execute actions - parse_actions returns 5 segments, process each segment
                # Each segment represents a 0.2 second time window
                for segment_idx, segment in enumerate(actions):
                    if successful_api_calls >= args.max_seconds:
                        break
                    
                    # Process all actions in this segment
                    # Instant actions are applied once, HOLD actions are held for the segment duration
                    hold_actions = []
                    instant_actions = []
                    
                    for action in segment:
                        if isinstance(action, tuple) and action[0] == "HOLD":
                            # HOLD action - will be held for the segment duration
                            hold_actions.append(action[1])
                        elif action is not None:
                            # Instant action
                            instant_actions.append(action)
                    
                    # Execute instant actions first (each gets a quick press)
                    for action_code in instant_actions:
                        execute_action_playwright(canvas, action_code)
                        executed_actions.append(action_code)
                        total_actions_count += 1

                    # Execute all HOLD actions simultaneously (all keydowns, wait, all keyups)
                    if hold_actions:
                        execute_hold_actions_playwright(canvas, hold_actions, duration=0.2)
                        for key_code in hold_actions:
                            executed_actions.append(("HOLD", key_code))
                            total_actions_count += 1
                    
                    # If no actions in this segment, it's a NOOP - just wait
                    if not instant_actions and not hold_actions:
                        time.sleep(0.2)
                    
                    # Capture screenshot during action execution (20 FPS)
                    capture_screenshot_if_needed()

                    # Capture screenshot after segment
                    post_action_bytes = get_game_screenshot_playwright(canvas)
                    action_shot_path = ""
                    frame_entry = None
                    if post_action_bytes:
                        action_shot_path = save_action_frame(gameplay_dir, frame_index, post_action_bytes)
                        frame_index += 1
                        frame_entry = build_frame_reference(action_shot_path, post_action_bytes)
                    if frame_entry:
                        # Create action name string for this segment and append to frame history
                        action_names = []
                        for action in segment:
                            if isinstance(action, tuple) and action[0] == "HOLD":
                                action_names.append(f"HOLD_{REVERSE_KEY_MAPPING.get(action[1], str(action[1]))}")
                            elif action is not None:
                                action_names.append(REVERSE_KEY_MAPPING.get(action, str(action)))
                        action_name_str = ", ".join(action_names) if action_names else "NOOP"
                        frame_history.append({
                            "frame": frame_entry,
                            "episode": episode_count,
                            "step": step_count,
                            "action_index": segment_idx,
                            "action_name": action_name_str,
                        })
                    action_snapshots.append({
                        "index": segment_idx,
                        "actions": segment,
                        "screenshot": os.path.relpath(action_shot_path, results_dir) if action_shot_path else "",
                    })
            
            
                last_executed_actions = executed_actions.copy()
                # Store render frames from this step (frames added to frame_history during action execution)
                last_render_frames = []
                for frame_entry in frame_history:
                    if (frame_entry.get("step") == step_count and
                        frame_entry.get("episode") == episode_count):
                        last_render_frames.append(frame_entry)

                log_entry = {
                    "episode": episode_count,
                    "step": step_count,
                    "step_timestamp": datetime.now().isoformat(),
                    "actions_executed": executed_actions,
                    "llm_output": response['output'],
                    "score": current_score,
                    "screenshot": log_screenshot,
                    "action_snapshots": action_snapshots,
                    "token_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "reasoning_tokens": reasoning_tokens if reasoning_tokens > 0 else None,
                        "cumulative_input_tokens": cumulative_input_tokens,
                        "cumulative_output_tokens": cumulative_output_tokens,
                        "cumulative_total_tokens": cumulative_total_tokens,
                        "cumulative_reasoning_tokens": cumulative_reasoning_tokens if cumulative_reasoning_tokens > 0 else None
                    }
                }
                gameplay_log.append(log_entry)
                record_gameplay_step(
                    gameplay_dir=gameplay_dir,
                    episode=episode_count,
                    step=step_count,
                    actions=executed_actions,
                    llm_output=response['output'],
                    screenshot=log_screenshot,
                    action_snapshots=action_snapshots,
                    score=current_score,
                    model_name=model_name,
                    game_name=game_name,
                    token_usage={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "reasoning_tokens": reasoning_tokens if reasoning_tokens > 0 else None,
                        "cumulative_input_tokens": cumulative_input_tokens,
                        "cumulative_output_tokens": cumulative_output_tokens,
                        "cumulative_total_tokens": cumulative_total_tokens,
                        "cumulative_reasoning_tokens": cumulative_reasoning_tokens if cumulative_reasoning_tokens > 0 else None
                    }
                )
                
                step_count += 1
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            save_results(results_dir, full_content_history, gameplay_log)
            
            # Create GIF from temp screenshots folder
            logger.info(f"Creating GIF from {temp_frame_index} screenshots at 20 FPS...")
            if os.path.exists(temp_screenshot_dir):
                screenshot_files = glob.glob(os.path.join(temp_screenshot_dir, "frame_*.png"))
                logger.info(f"Found {len(screenshot_files)} screenshots in temp folder")
                
                if screenshot_files:
                    gif_path = create_gameplay_gif(
                        temp_screenshot_dir,
                        output_path=os.path.join(gameplay_dir, "gameplay.gif"),
                        duration=50  # 50ms per frame = 20 FPS
                    )
                    
                    if gif_path:
                        logger.info(f"GIF created successfully: {gif_path}")
                    else:
                        logger.warning("Failed to create GIF from screenshots")
                else:
                    logger.warning(f"No screenshots found in temp folder: {temp_screenshot_dir}")
                
                # Delete temp folder after GIF creation
                try:
                    shutil.rmtree(temp_screenshot_dir)
                    logger.info(f"Deleted temp screenshot folder: {temp_screenshot_dir}")
                except Exception as e:
                    logger.warning(f"Could not delete temp folder {temp_screenshot_dir}: {e}")
            else:
                logger.warning(f"Temp screenshot folder does not exist: {temp_screenshot_dir}")
            
            logger.info("Gameplay finished")
            browser.close()


def main():
    args = parse_arguments()
    game_url = args.url
    logger.info(f"Using game URL: {game_url}")
    
    model_provider, model_name = parse_model_string(args.model)
    logger.info(f"Using model provider: {model_provider}, model: {model_name}")
    
    results_base = os.path.join(script_dir, "results")
    os.makedirs(results_base, exist_ok=True)
    now = datetime.now()
    session_id = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond:06d}"
    sanitized_model_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    
    prompt_template_path = os.path.join(script_dir, "prompts", "prompt.md")
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()
    logger.info(f"Loaded prompt template from {prompt_template_path}")
    
    run_one_game(
        game_url=game_url,
        args=args,
        model_provider=model_provider,
        model_name=model_name,
        prompt_template=prompt_template,
        results_base=results_base,
        session_id=session_id,
        sanitized_model_name=sanitized_model_name,
    )


if __name__ == "__main__":
    main()

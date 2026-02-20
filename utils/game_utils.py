import time
import logging
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)

# Mapping from key codes to key names for JavaScript
KEY_CODE_TO_KEY_NAME = {
    37: 'ArrowLeft',
    38: 'ArrowUp',
    39: 'ArrowRight',
    40: 'ArrowDown',
    32: ' ',
    13: 'Enter',
    27: 'Escape',
    # Letter keys A-Z (65-90)
    65: 'a', 66: 'b', 67: 'c', 68: 'd', 69: 'e', 70: 'f',
    71: 'g', 72: 'h', 73: 'i', 74: 'j', 75: 'k', 76: 'l',
    77: 'm', 78: 'n', 79: 'o', 80: 'p', 81: 'q', 82: 'r',
    83: 's', 84: 't', 85: 'u', 86: 'v', 87: 'w', 88: 'x',
    89: 'y', 90: 'z',
}

def execute_action(driver, action_code):
    """Execute a single action in the game by sending actual keypresses to the canvas."""
    if action_code is None:
        # NOOP action - do nothing
        return
    
    try:
        # Find the canvas element
        canvas = driver.find_element(By.ID, "defaultCanvas0")
        
        # Focus on the canvas to ensure keypresses are received
        canvas.click()
        time.sleep(0.05)  # Small delay to ensure focus
        
        # Get the key name to send
        if action_code in KEY_CODE_TO_KEY_NAME:
            key_name = KEY_CODE_TO_KEY_NAME[action_code]
            
            # Send keypress using JavaScript to dispatch keyboard events
            # This is more reliable for canvas elements than Selenium's ActionChains
            # Escape the key name for JavaScript string
            key_name_escaped = key_name.replace("'", "\\'").replace("\\", "\\\\")
            
            driver.execute_script(f"""
                var canvas = arguments[0];
                var keyCode = {action_code};
                var key = '{key_name_escaped}';
                
                // Create and dispatch keydown event
                var keydownEvent = new KeyboardEvent('keydown', {{
                    keyCode: keyCode,
                    which: keyCode,
                    key: key,
                    code: keyCode === 37 ? 'ArrowLeft' : 
                          keyCode === 38 ? 'ArrowUp' : 
                          keyCode === 39 ? 'ArrowRight' : 
                          keyCode === 40 ? 'ArrowDown' : 
                          keyCode === 32 ? 'Space' : 
                          keyCode === 13 ? 'Enter' : 
                          keyCode === 27 ? 'Escape' : key,
                    bubbles: true,
                    cancelable: true
                }});
                canvas.dispatchEvent(keydownEvent);
                
                // Create and dispatch keypress event (for printable characters)
                if (keyCode >= 32 && keyCode !== 127) {{
                    var keypressEvent = new KeyboardEvent('keypress', {{
                        keyCode: keyCode,
                        which: keyCode,
                        key: key,
                        bubbles: true,
                        cancelable: true
                    }});
                    canvas.dispatchEvent(keypressEvent);
                }}
                
                // Create and dispatch keyup event
                var keyupEvent = new KeyboardEvent('keyup', {{
                    keyCode: keyCode,
                    which: keyCode,
                    key: key,
                    code: keyCode === 37 ? 'ArrowLeft' : 
                          keyCode === 38 ? 'ArrowUp' : 
                          keyCode === 39 ? 'ArrowRight' : 
                          keyCode === 40 ? 'ArrowDown' : 
                          keyCode === 32 ? 'Space' : 
                          keyCode === 13 ? 'Enter' : 
                          keyCode === 27 ? 'Escape' : key,
                    bubbles: true,
                    cancelable: true
                }});
                canvas.dispatchEvent(keyupEvent);
            """, canvas)
        else:
            logger.warning(f"Unknown key code: {action_code}")
            
    except Exception as e:
        logger.error(f"Error executing action {action_code}: {e}")


def execute_action_segment(driver, action_segment, duration=0.2):
    """
    Execute a time segment with potentially multiple simultaneous actions.
    
    Args:
        driver: Selenium driver
        action_segment: List of actions for this segment. Each action can be:
            - An integer (instant action, applied once at start)
            - A tuple ("HOLD", code) (continuous action, held for duration)
            - None (NOOP)
        duration: Duration of the segment in seconds (default 0.2)
    """
    instant_actions = []
    continuous_actions = []
    
    # Separate instant and continuous actions
    for action in action_segment:
        if action is None:
            # NOOP - do nothing
            continue
        elif isinstance(action, tuple) and action[0] == "HOLD":
            # Continuous action
            continuous_actions.append(action[1])
        else:
            # Instant action
            instant_actions.append(action)
    
    # Apply instant actions once at the start
    if instant_actions:
        # For multiple simultaneous instant actions, we need to apply them
        # The game.step() might only accept one action, so we'll apply them sequentially
        # If the game supports simultaneous actions, this might need adjustment
        for action_code in instant_actions:
            execute_action(driver, action_code)
    
    # Apply continuous actions for the duration, or wait if no continuous actions
    if continuous_actions:
        # For continuous actions, we'll call game.step() multiple times during the duration
        # Using a small interval (0.05s) to simulate continuous holding
        segment_start = time.time()
        interval = 0.05  # Call every 50ms
        while time.time() - segment_start < duration:
            for action_code in continuous_actions:
                execute_action(driver, action_code)
            remaining_time = duration - (time.time() - segment_start)
            if remaining_time > 0:
                time.sleep(min(interval, remaining_time))
    else:
        # No continuous actions - just wait for the duration
        # (instant actions were already applied at the start, or it's a NOOP)
        time.sleep(duration)

def get_game_screenshot(driver) -> bytes:
    """Get PNG bytes of the game canvas."""
    try:
        canvas = driver.find_element(By.ID, "defaultCanvas0")
        return canvas.screenshot_as_png
    except Exception as e:
        logger.error(f"Error getting screenshot: {e}")
        return b""

def is_game_ended(driver) -> bool:
    """Check if the game has ended."""
    try:
        # Check the 'ended' variable in the game instance
        return driver.execute_script("return window.game.ended")
    except Exception as e:
        # Fallback: check if retry button is visible
        try:
            retry_btn = driver.find_element(By.ID, "retry")
            return retry_btn.is_displayed()
        except:
            return False


def is_game_won(driver) -> bool:
    """Return True if the current game instance reports a win state."""
    try:
        return bool(driver.execute_script("return !!(window.game && window.game.win);"))
    except Exception as exc:
        logger.error(f"Error checking win state: {exc}")
        return False

def click_start_llm(driver):
    """Click the Start LLM button."""
    try:
        btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "start-llm"))
        )
        btn.click()
        logger.info("Clicked Start LLM button")
        time.sleep(1) # Wait for init
    except Exception as e:
        logger.error(f"Error clicking Start LLM: {e}")
        raise

def click_retry(driver):
    """Click the Retry button."""
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "retry"))
        )
        btn.click()
        logger.info("Clicked Retry button")
        time.sleep(1) # Wait for reset
        return True
    except Exception as e:
        logger.error(f"Error clicking Retry: {e}")
        return False


def click_next_instance(driver):
    """Click the Next Instance / Next Level button."""
    try:
        btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "next"))
        )
        btn.click()
        logger.info("Clicked Next Instance button")
        time.sleep(1)
        return True
    except Exception as e:
        logger.error(f"Error clicking Next Instance: {e}")
        return False


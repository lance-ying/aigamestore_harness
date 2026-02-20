import json
import re
import logging

logger = logging.getLogger(__name__)

KEY_MAPPING = {
    "UP": 38,
    "DOWN": 40,
    "LEFT": 37,
    "RIGHT": 39,
    "SPACE": 32,
    "ENTER": 13,
    "ESC": 27,
    "NOOP": None,
    # Letter keys (A-Z)
    "A": 65, "B": 66, "C": 67, "D": 68, "E": 69, "F": 70,
    "G": 71, "H": 72, "I": 73, "J": 74, "K": 75, "L": 76,
    "M": 77, "N": 78, "O": 79, "P": 80, "Q": 81, "R": 82,
    "S": 83, "T": 84, "U": 85, "V": 86, "W": 87, "X": 88,
    "Y": 89, "Z": 90,
    # Continuous actions (held for the duration)
    "HOLD_UP": ("HOLD", 38),
    "HOLD_DOWN": ("HOLD", 40),
    "HOLD_LEFT": ("HOLD", 37),
    "HOLD_RIGHT": ("HOLD", 39),
    "HOLD_SPACE": ("HOLD", 32),
}

def parse_actions(response_text: str) -> list[list]:
    """
    Extract actions from LLM response.
    Returns a list of 5 lists, where each inner list contains actions for one time segment.
    Each action can be an instant action (applied once) or a continuous action (held for duration).
    """
    match = re.search(r'<keys>(.*?)</keys>', response_text, re.DOTALL)
    if match:
        try:
            actions_str = match.group(1).strip()
            # Using json.loads to safely parse the list string
            # The LLM is expected to output valid JSON: [["UP"], ["LEFT", "RIGHT"], ["NOOP"], ...]
            # Only replace single quotes with double quotes if necessary
            if "'" in actions_str and '"' not in actions_str:
                actions_str = actions_str.replace("'", '"')
            
            action_segments = json.loads(actions_str)
            
            # Validate that we have exactly 5 segments
            if not isinstance(action_segments, list):
                logger.error("Actions must be a list")
                return []
            
            if len(action_segments) != 5:
                logger.warning(f"Expected 5 action segments, got {len(action_segments)}. Padding or truncating.")
                if len(action_segments) < 5:
                    action_segments.extend([["NOOP"]] * (5 - len(action_segments)))
                else:
                    action_segments = action_segments[:5]
            
            parsed_segments = []
            for segment_idx, segment in enumerate(action_segments):
                if not isinstance(segment, list):
                    logger.warning(f"Segment {segment_idx} is not a list, converting to list")
                    segment = [segment] if segment else ["NOOP"]
                
                parsed_segment = []
                for action_name in segment:
                    if action_name is None:
                        action_name = "NOOP"
                    action_name = str(action_name).upper()
                    
                    if action_name in KEY_MAPPING:
                        parsed_segment.append(KEY_MAPPING[action_name])
                    elif action_name == "NOOP":
                        parsed_segment.append(None)
                    else:
                        logger.warning(f"Unknown action: {action_name} in segment {segment_idx}")
                
                parsed_segments.append(parsed_segment)
            
            return parsed_segments
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse actions JSON: {e}")
            return [[None]] * 5  # Return 5 NOOP segments on error
        except Exception as e:
            logger.error(f"Error parsing actions: {e}")
            return [[None]] * 5
    else:
        logger.warning("No <keys> tag found in response")
        return [[None]] * 5


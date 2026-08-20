class SafetyInterpreter:
    """
    Translates raw YOLO bounding box classes into temporal-confirmed safety events.
    """
    
    # Map of YOLO class IDs that constitute a direct safety violation
    VIOLATION_MAP = {
        0: "Fall-Detected",
        6: "NO-Gloves",
        7: "NO-Goggles",
        8: "NO-Hardhat",
        9: "NO-Mask",
        10: "NO-Safety Vest"
    }

    def __init__(self, temporal_threshold=5):
        """
        :param temporal_threshold: Number of consecutive frames a violation must be present to be confirmed.
        """
        self.temporal_threshold = temporal_threshold
        # Tracks how many consecutive frames a specific violation has been detected
        self.violation_counters = {class_id: 0 for class_id in self.VIOLATION_MAP.keys()}
        
        # The currently confirmed active violations
        self.active_violations = set()

    def process_frame(self, detected_class_ids):
        """
        Takes a list of class IDs detected in the current frame and updates the state machine.
        Returns a list of strings describing the currently active confirmed violations.
        """
        # Convert to a set for O(1) lookups
        detected_set = set(detected_class_ids)
        
        # Check each possible violation class
        for class_id, violation_name in self.VIOLATION_MAP.items():
            if class_id in detected_set:
                # Increment the consecutive counter
                self.violation_counters[class_id] += 1
                
                # If it exceeds the threshold, mark it as an active violation
                if self.violation_counters[class_id] >= self.temporal_threshold:
                    self.active_violations.add(violation_name)
            else:
                # Reset the counter if it's not detected in this frame
                self.violation_counters[class_id] = 0
                
                # Also remove it from active violations (this means the violation has cleared)
                if violation_name in self.active_violations:
                    self.active_violations.remove(violation_name)
                    
        return list(self.active_violations)

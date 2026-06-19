import re
from typing import Dict
class SlotExtractor:
    def __init__(self):
        self.patterns = {
            "delay_time": r"(10|15|20|5)\s*(min|minute|mins|m)|(aadh|ek)\s*(ghanta|hr)",
            "delay_reason": r"(traffic|baarish|accident|bheed|tyre puncture|petrol khatam)",
            "order_ref": r"(pichla|next|current|order number\s*\d+)",
            "customer_status": r"(nahi utha|phone off|not reachable|not answering|gate band|mil nahi raha)"
        }
        self.compiled = {k: re.compile(v, re.IGNORECASE) for k, v in self.patterns.items()}
    def extract(self, text: str) -> Dict[str, str]:
        slots = {}
        for slot_name, pattern in self.compiled.items():
            match = pattern.search(text)
            if match:
                slots[slot_name] = match.group(0)
        return slots

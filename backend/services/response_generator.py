from typing import Dict

class ResponseGenerator:
    @staticmethod
    def generate(intent: str, slots: Dict[str, str]) -> str:
        if intent == "get_address":
            ref = slots.get("order_ref", "current")
            return f"Fetching address for the {ref} order."
            
        elif intent == "report_delay":
            time = slots.get("delay_time", "some time")
            reason = slots.get("delay_reason", "unspecified reasons")
            return f"Notified customer about a delay of {time} due to {reason}."
            
        elif intent == "order_issue":
            return "Order issue registered. Support team has been notified."
            
        elif intent == "customer_unavailable":
            status = slots.get("customer_status", "unavailable")
            return f"Logged customer status as: {status}. Awaiting support instruction."
            
        elif intent == "navigation_help":
            return "Rerouting navigation and restarting GPS module."
            
        return "Command received."

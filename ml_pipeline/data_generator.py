import csv
import random
import os
import itertools
from typing import List, Dict, Tuple

# Set random seed for reproducibility
random.seed(42)

# ==========================================
# 1. TEMPLATES & VOCABULARY DEFINITION
# ==========================================

GREETINGS_FORMAL = ["kripya", "mahoday", "sir"]
GREETINGS_CASUAL = ["bhai", "yaar", "boss", "bhaiya"]
GREETINGS_SLANG = ["ustad", "guru", "bawa"]

TIME_MINUTES = ["5 min", "10 minute", "15 mins", "aadh ghanta", "ek ghanta", "20 min"]
REASONS_DELAY = ["traffic", "baarish", "accident", "bheed", "tyre puncture", "petrol khatam"]

# Intent 1: get_address
GET_ADDRESS_FORMAL = [
    "{greeting} aagami aadesh ka pata pradan karein",
    "{greeting} grahak ka location bhejein"
]
GET_ADDRESS_CASUAL = [
    "{greeting} next order ka address batao",
    "{greeting} customer kidhar hai",
    "location send karo {greeting}"
]

# Intent 2: report_delay
REPORT_DELAY_FORMAL = [
    "{greeting} mujhe aane mein {time} lagenge",
    "{reason} ke karan delay hoga"
]
REPORT_DELAY_CASUAL = [
    "{greeting} {reason} ki wajah se {time} late honga",
    "{time} late pahunchunga {reason} hai",
    "late ho jaunga {time}"
]

# Intent 3: order_issue
ORDER_ISSUE_FORMAL = [
    "{greeting} parcel kshatigrast hai",
    "order radd karna hai"
]
ORDER_ISSUE_CASUAL = [
    "{greeting} order cancel kar do",
    "item missing hai",
    "package damage ho gaya {greeting}",
    "galat order de diya hai"
]

# Intent 4: customer_unavailable
CUSTOMER_UNAV_FORMAL = [
    "{greeting} grahak sampark mein nahi hain",
    "darwaza band hai"
]
CUSTOMER_UNAV_CASUAL = [
    "{greeting} customer phone nahi utha raha",
    "number not reachable hai",
    "gate par koi nahi hai {greeting}",
    "customer mil nahi raha"
]

# Intent 5: navigation_help
NAV_HELP_FORMAL = [
    "{greeting} map kaam nahi kar raha",
    "sahi marg dikhayein"
]
NAV_HELP_CASUAL = [
    "{greeting} map stuck ho gaya hai",
    "location samajh nahi aa raha",
    "rasta dikhao {greeting}",
    "gps hang ho gaya"
]

# ==========================================
# 2. GENERATION LOGIC
# ==========================================

def generate_combinations(templates: List[str], intent: str, style: str) -> List[Dict]:
    results = []
    greetings = GREETINGS_FORMAL if style == "formal_hindi" else GREETINGS_CASUAL + GREETINGS_SLANG + [""]
    
    for template in templates:
        for greeting in greetings:
            if "{time}" in template and "{reason}" in template:
                for time, reason in itertools.product(TIME_MINUTES, REASONS_DELAY):
                    text = template.format(greeting=greeting, time=time, reason=reason).strip()
                    results.append({"text": text, "intent": intent, "style": style})
            elif "{time}" in template:
                for time in TIME_MINUTES:
                    text = template.format(greeting=greeting, time=time).strip()
                    results.append({"text": text, "intent": intent, "style": style})
            elif "{reason}" in template:
                for reason in REASONS_DELAY:
                    text = template.format(greeting=greeting, reason=reason).strip()
                    results.append({"text": text, "intent": intent, "style": style})
            else:
                text = template.format(greeting=greeting).strip()
                results.append({"text": text, "intent": intent, "style": style})
    return results
ABBREVIATIONS = {
    "bhai": ["bh", "bhi"],
    "address": ["add", "adr"],
    "kya": ["k", "ky"],
    "nahi": ["ni", "nai", "nhi"],
    "karo": ["kr", "kro"],
    "hai": ["h"],
    "raha": ["rha"],
    "wajah": ["wjh", "vjh"]
}
def apply_abbreviations(text: str) -> str:
    words = text.split()
    for i, word in enumerate(words):
        if word in ABBREVIATIONS and random.random() < 0.7:
            words[i] = random.choice(ABBREVIATIONS[word])
    return " ".join(words)
def apply_spelling_mistake(text: str) -> str:
    if len(text) < 5 or random.random() < 0.3:
        return text
    chars = list(text)
    idx = random.randint(0, len(chars) - 2)
    
    if random.random() < 0.5:
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    else:
        if chars[idx] != ' ':
            chars.pop(idx)
            
    return "".join(chars)

def create_augmented_dataset(base_data: List[Dict], target_count: int) -> List[Dict]:
    augmented = list(base_data)
    
    while len(augmented) < target_count:
        sample = random.choice(base_data).copy()
        
        aug_type = random.choice(["abbreviated", "spelling_mistake"])
        
        if aug_type == "abbreviated":
            sample["text"] = apply_abbreviations(sample["text"])
            sample["style"] = "abbreviated"
        else:
            sample["text"] = apply_spelling_mistake(sample["text"])
            sample["style"] = "augmented_spelling"
            
        augmented.append(sample)
        
    return augmented

# ==========================================
# 4. PIPELINE EXECUTION
# ==========================================

def main():
    print("Initializing Data Generation Pipeline...")
    
    base_dataset = []
    
    # 1. Generate Formal & Casual
    base_dataset.extend(generate_combinations(GET_ADDRESS_FORMAL, "get_address", "formal_hindi"))
    base_dataset.extend(generate_combinations(GET_ADDRESS_CASUAL, "get_address", "casual_hindi"))
    
    base_dataset.extend(generate_combinations(REPORT_DELAY_FORMAL, "report_delay", "formal_hindi"))
    base_dataset.extend(generate_combinations(REPORT_DELAY_CASUAL, "report_delay", "casual_hindi"))
    
    base_dataset.extend(generate_combinations(ORDER_ISSUE_FORMAL, "order_issue", "formal_hindi"))
    base_dataset.extend(generate_combinations(ORDER_ISSUE_CASUAL, "order_issue", "casual_hindi"))
    
    base_dataset.extend(generate_combinations(CUSTOMER_UNAV_FORMAL, "customer_unavailable", "formal_hindi"))
    base_dataset.extend(generate_combinations(CUSTOMER_UNAV_CASUAL, "customer_unavailable", "casual_hindi"))
    
    base_dataset.extend(generate_combinations(NAV_HELP_FORMAL, "navigation_help", "formal_hindi"))
    base_dataset.extend(generate_combinations(NAV_HELP_CASUAL, "navigation_help", "casual_hindi"))
    
    # Clean up multiple spaces
    for item in base_dataset:
        item["text"] = " ".join(item["text"].split())
        
    print(f"Generated {len(base_dataset)} base examples.")
    
    # 2. Augment to 5000+ examples
    TARGET_SIZE = 5500
    final_dataset = create_augmented_dataset(base_dataset, TARGET_SIZE)
    random.shuffle(final_dataset)
    print(f"Augmented dataset to {len(final_dataset)} examples.")
    
    # 3. Split Dataset (80/10/10)
    train_split = int(len(final_dataset) * 0.8)
    val_split = int(len(final_dataset) * 0.9)
    
    train_data = final_dataset[:train_split]
    val_data = final_dataset[train_split:val_split]
    test_data = final_dataset[val_split:]
    
    # 4. Save to CSV
    os.makedirs(os.path.join(os.path.dirname(__file__), "../data"), exist_ok=True)
    
    def save_csv(filename: str, data: List[Dict]):
        path = os.path.join(os.path.dirname(__file__), "../data", filename)
        with open(path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["text", "intent", "style"])
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved {len(data)} rows to {path}")
        
    save_csv("train.csv", train_data)
    save_csv("val.csv", val_data)
    save_csv("test.csv", test_data)
    save_csv("full_dataset.csv", final_dataset)
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()

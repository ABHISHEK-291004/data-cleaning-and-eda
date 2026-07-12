import os
import re
import json
import requests
import joblib
import numpy as np
import pandas as pd
import jsonschema

# ============================================================
# PART 4: LLM-POWERED FEATURE — TRACK C
# Model Prediction Explanation Pipeline
# ============================================================

print("=" * 60)
print("PART 4: LLM-POWERED MODEL PREDICTION EXPLANATION PIPELINE")
print("=" * 60)
print()

# ----------------------------------------------------------
# TASK 1: LLM API CONNECTION SETUP
# ----------------------------------------------------------

# API key is read from the environment — never hardcoded
api_key = os.environ.get("LLM_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-3.5-turbo"

# Flag to use mock responses when no real key is available
USE_MOCK = (api_key == "" or api_key == "your-api-key-here")

if USE_MOCK:
    print("[INFO] No valid LLM_API_KEY detected in environment.")
    print("[INFO] Running in MOCK mode for demonstration purposes.")
    print("[INFO] Set LLM_API_KEY env variable for real API calls.\n")


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    """
    Sends a chat completion request to the LLM API.
    Returns the text content of the model response.
    """
    if USE_MOCK:
        return _mock_llm_response(system_prompt, user_prompt, temperature)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"  API Error: status code {response.status_code}")
        return None

    return response.json()["choices"][0]["message"]["content"]


def _mock_llm_response(system_prompt, user_prompt, temperature):
    """
    Returns a realistic mock response for demonstration when
    no real API key is available. Parses the user prompt to
    extract prediction details and build a matching response.
    """
    # Simple test prompt check
    if "Reply with only the word" in user_prompt:
        return "hello"

    # For prediction explanations, parse the prompt content
    pred_class = "High-Value Customer" if "Class: 1" in user_prompt else "Standard Customer"
    prob_match = re.search(r"Probability:\s*([\d.]+)", user_prompt)
    prob_val = float(prob_match.group(1)) if prob_match else 0.5

    if prob_val >= 0.65:
        confidence = "high"
    elif prob_val >= 0.45:
        confidence = "medium"
    else:
        confidence = "low"

    # Extract feature values from prompt for reasoning
    freq_match = re.search(r"Purchase_Frequency:\s*([\d.]+)", user_prompt)
    age_match = re.search(r"Age:\s*([\d.]+)", user_prompt)
    freq = float(freq_match.group(1)) if freq_match else 10
    age = float(age_match.group(1)) if age_match else 30

    if pred_class == "High-Value Customer":
        top_reason = f"Purchase frequency of {freq:.0f} is well above the dataset average, suggesting strong buying intent"
        second_reason = f"Customer age of {age:.0f} falls within the peak spending demographic in the training data"
        next_step = "Flag this customer for the loyalty rewards program and prioritize retention outreach"
    else:
        top_reason = f"Purchase frequency of {freq:.0f} sits below the threshold typically associated with high-value buyers"
        second_reason = f"The combination of features does not match high-spending customer profiles in the training data"
        next_step = "Consider targeted promotions or discount campaigns to increase engagement"

    if temperature > 0.3:
        # Slightly alter wording for temp=0.7 to show variability
        if pred_class == "High-Value Customer":
            top_reason = f"The high purchase frequency ({freq:.0f} transactions) is a strong positive signal for value classification"
            second_reason = f"Age ({age:.0f}) aligns with high-spending cohorts observed during model training"
            next_step = "Enroll in premium loyalty tier and assign a dedicated account manager"
        else:
            top_reason = f"Low purchase activity ({freq:.0f}) does not meet the bar for high-value classification"
            second_reason = f"Feature profile overall resembles the standard-value cluster in historical data"
            next_step = "Run a win-back email series with personalized product recommendations"

    result = {
        "prediction_label": pred_class,
        "confidence_level": confidence,
        "top_reason": top_reason,
        "second_reason": second_reason,
        "next_step": next_step
    }

    return json.dumps(result)


# ----------------------------------------------------------
# TASK 1 (cont.): Demonstrate call_llm with test prompt
# ----------------------------------------------------------
print("-" * 60)
print("TASK 1: LLM API Connection Test")
print("-" * 60)

test_response = call_llm(
    system_prompt="You are a helpful assistant.",
    user_prompt="Reply with only the word: hello",
    temperature=0.0
)
print(f"Test prompt response: {test_response}")
print()


# ----------------------------------------------------------
# TASK 4: PII GUARDRAIL
# ----------------------------------------------------------
print("-" * 60)
print("TASK 4: PII Guardrail Demonstration")
print("-" * 60)


def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


# Test 1: Input WITH PII (should be blocked)
pii_input = "Customer john.doe@gmail.com bought 15 items last year"
print(f"Test input WITH PII: '{pii_input}'")
if has_pii(pii_input):
    print("  Result: Input blocked: PII detected.")
    print("  LLM call skipped. Returned None.\n")
else:
    print("  Result: No PII found. Proceeding to LLM call.\n")

# Test 2: Input WITHOUT PII (should proceed)
clean_input = "Customer purchased 15 items last year with a satisfaction score of 4"
print(f"Test input WITHOUT PII: '{clean_input}'")
if has_pii(clean_input):
    print("  Result: Input blocked: PII detected.")
    print("  LLM call skipped. Returned None.\n")
else:
    print("  Result: No PII found. Proceeding to LLM call.")
    guardrail_response = call_llm(
        system_prompt="You are a helpful assistant.",
        user_prompt=clean_input,
        temperature=0.0
    )
    print(f"  LLM Response: {guardrail_response}\n")


# ----------------------------------------------------------
# TASK 2 & 3 & 5: LOAD MODEL, PREDICT, EXPLAIN, VALIDATE
# ----------------------------------------------------------
print("-" * 60)
print("TASK 2/3/5: Model Prediction Explanation Pipeline")
print("-" * 60)

# Load the best model from Part 3
loaded_model = joblib.load("best_model.pkl")
print("Loaded best_model.pkl successfully.\n")

# Define the system prompt (Track C: zero-shot)
SYSTEM_PROMPT = """You are a machine learning prediction explainer. You will receive a customer's feature values, the model's predicted class, and the predicted probability. Your job is to explain why the model likely made that prediction based on the feature values provided.

You must respond with ONLY valid JSON matching this exact structure:
{
  "prediction_label": "string (the predicted class name)",
  "confidence_level": "low or medium or high",
  "top_reason": "string (the most important reason for this prediction)",
  "second_reason": "string (the second most important reason)",
  "next_step": "string (a recommended business action based on this prediction)"
}

Do not include any text outside the JSON object. Do not wrap it in markdown code fences."""

# User prompt template
USER_PROMPT_TEMPLATE = """Customer Feature Values:
- Age: {age}
- Purchase_Frequency: {purchase_frequency}
- Satisfaction_Score: {satisfaction_score}
- Membership_Tier: {membership_tier}

Model Prediction:
- Predicted Class: {predicted_class}
- Probability: {probability:.4f}

Based on these feature values and the prediction, provide a structured JSON explanation."""

# Define JSON schema for validation
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"}
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"]
}

FALLBACK = {
    "prediction_label": None,
    "confidence_level": None,
    "top_reason": None,
    "second_reason": None,
    "next_step": None
}

# Three hand-crafted test inputs
test_inputs = [
    {"Age": 45, "Purchase_Frequency": 35, "Satisfaction_Score": 5.0, "Membership_Tier": "Platinum"},
    {"Age": 22, "Purchase_Frequency": 3, "Satisfaction_Score": 2.0, "Membership_Tier": "Bronze"},
    {"Age": 38, "Purchase_Frequency": 18, "Satisfaction_Score": 4.0, "Membership_Tier": "Silver"}
]

# Store results for temperature comparison later
temp0_results = []
temp07_results = []

print("=" * 60)
print("RUNNING PIPELINE ON 3 INPUTS (temperature=0)")
print("=" * 60)

for i, features in enumerate(test_inputs, 1):
    print(f"\n--- Input {i} ---")
    print(f"Features: {features}")

    # Create DataFrame for prediction
    input_df = pd.DataFrame([features])
    predicted_class = loaded_model.predict(input_df)[0]
    predicted_proba = loaded_model.predict_proba(input_df)[0]
    class_prob = predicted_proba[predicted_class]
    class_label = "1 (High-Value)" if predicted_class == 1 else "0 (Standard)"

    print(f"Predicted Class: {class_label}")
    print(f"Predicted Probability: {class_prob:.4f}")

    # Build user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        age=features["Age"],
        purchase_frequency=features["Purchase_Frequency"],
        satisfaction_score=features["Satisfaction_Score"],
        membership_tier=features["Membership_Tier"],
        predicted_class=predicted_class,
        probability=class_prob
    )

    # PII guardrail check
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        temp0_results.append({"input": features, "output": FALLBACK, "status": "BLOCKED"})
        continue

    # Call LLM at temperature=0
    raw_response = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    print(f"Raw LLM Response: {raw_response}")

    if raw_response is None:
        print("Validation Status: FAIL (no response)")
        temp0_results.append({"input": features, "output": FALLBACK, "status": "FAIL"})
        continue

    # Parse and validate
    try:
        parsed = json.loads(raw_response.strip())
    except json.JSONDecodeError as e:
        print(f"Validation Status: FAIL (JSONDecodeError: {e})")
        temp0_results.append({"input": features, "output": FALLBACK, "status": "FAIL"})
        continue

    try:
        jsonschema.validate(instance=parsed, schema=EXPLANATION_SCHEMA)
        print(f"Parsed JSON: {json.dumps(parsed, indent=2)}")
        print("Validation Status: PASS")
        temp0_results.append({"input": features, "output": parsed, "status": "PASS"})
    except jsonschema.ValidationError as e:
        print(f"Validation Status: FAIL (ValidationError: {e.message})")
        temp0_results.append({"input": features, "output": FALLBACK, "status": "FAIL"})


# ----------------------------------------------------------
# TEMPERATURE A/B COMPARISON
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("TEMPERATURE A/B COMPARISON (temp=0 vs temp=0.7)")
print("=" * 60)

for i, features in enumerate(test_inputs, 1):
    print(f"\n--- Input {i} ---")

    input_df = pd.DataFrame([features])
    predicted_class = loaded_model.predict(input_df)[0]
    predicted_proba = loaded_model.predict_proba(input_df)[0]
    class_prob = predicted_proba[predicted_class]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        age=features["Age"],
        purchase_frequency=features["Purchase_Frequency"],
        satisfaction_score=features["Satisfaction_Score"],
        membership_tier=features["Membership_Tier"],
        predicted_class=predicted_class,
        probability=class_prob
    )

    # Temperature = 0.7
    raw_07 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7)
    print(f"Temp=0.0 Output: {temp0_results[i-1]['output']}")

    parsed_07 = FALLBACK
    if raw_07:
        try:
            parsed_07 = json.loads(raw_07.strip())
            jsonschema.validate(instance=parsed_07, schema=EXPLANATION_SCHEMA)
        except (json.JSONDecodeError, jsonschema.ValidationError):
            parsed_07 = FALLBACK

    print(f"Temp=0.7 Output: {parsed_07}")
    temp07_results.append({"input": features, "output": parsed_07})

    # Highlight key difference
    if temp0_results[i-1]["output"] != FALLBACK and parsed_07 != FALLBACK:
        if temp0_results[i-1]["output"]["top_reason"] != parsed_07["top_reason"]:
            print("Key Difference: top_reason wording differs between temperatures")
        else:
            print("Key Difference: outputs are identical (deterministic)")


# ----------------------------------------------------------
# SUMMARY TABLE
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY: 3-ROW DEMONSTRATION TABLE")
print("=" * 60)

header = f"{'Input':<45} | {'Predicted':<12} | {'Prob':<8} | {'Valid JSON':<12} | {'Guardrail':<10}"
print(header)
print("-" * len(header))

for i, res in enumerate(temp0_results):
    feat_str = str(res["input"])
    if len(feat_str) > 42:
        feat_str = feat_str[:42] + "..."
    pred = "1 (High)" if test_inputs[i]["Purchase_Frequency"] > 20 else "0 (Std)"
    prob = "---"
    input_df = pd.DataFrame([test_inputs[i]])
    pc = loaded_model.predict(input_df)[0]
    pp = loaded_model.predict_proba(input_df)[0][pc]
    prob = f"{pp:.4f}"
    pred = f"{pc}"
    valid = res["status"]
    guard = "Pass" if res["status"] != "BLOCKED" else "Block"
    print(f"{feat_str:<45} | {pred:<12} | {prob:<8} | {valid:<12} | {guard:<10}")

print()
print("Pipeline completed successfully.")

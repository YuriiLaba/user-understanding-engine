import json
from typing import List, Dict, Any
from openai import OpenAI
import csv
import os
from dotenv import load_dotenv

load_dotenv()

def check_propositions_against_profile(
    profile_text: str,
    propositions: List[str],
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    batch_size: int = 10,   # process N propositions at a time
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: { 'proposition': str, 'supported': bool}
    'supported' is True only if the profile explicitly or implicitly substantiates the claim.
    """
    client = OpenAI(api_key=api_key)
    all_results: List[Dict[str, Any]] = []

    system_msg = (
        "You are a careful fact-checker. "
        "Given a short user profile and a list of propositions, decide for each proposition "
        "whether the profile SUPPORTS it (True) or NOT SUPPORTED (False).\n\n"
        "Definition of SUPPORTS:\n"
        "- True if the profile explicitly states the proposition.\n"
        "- True if the profile clearly and strongly implies the proposition.\n"
        "- True if the profile describes a GENERAL CATEGORY that covers the proposition "
        "  (e.g., 'works in software development' covers 'debugging Python scripts').\n"
        "- False if the profile does not mention it, only vaguely hints at it, or contradicts it.\n\n"
        "Be strict: do not guess beyond what is explicit, clearly implied, or included in a general category."
    )

    # process in batches
    for i in range(0, len(propositions), batch_size):
        batch = propositions[i : i + batch_size]

        user_msg = f"""
        USER PROFILE
        ---
        {profile_text.strip()}

        PROPOSITIONS
        ---
        {json.dumps(batch, ensure_ascii=False, indent=2)}

        OUTPUT FORMAT
        ---
        Return a JSON object with this exact shape:
        {{
        "results": [
            {{"proposition": "...", "supported": true/false}}
        ]
        }}

        RULES
        - "supported": true only if the profile explicitly states or strongly implies the proposition.
        - If the profile doesn't mention it or it's speculative → supported = false.
        """

        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "propositions_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "proposition": {"type": "string"},
                                        "supported": {"type": "boolean"}
                                    },
                                    "required": ["proposition", "supported"]
                                }
                            }
                        },
                        "required": ["results"]
                    }
                },
            },
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        content = resp.choices[0].message.content

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print("JSON decoding failed, raw content:\n", content[:1000])
            fixed = content.strip()
            fixed = fixed.replace("\n", " ")  # flatten
            fixed = fixed.replace(",}", "}")  # remove trailing commas
            fixed = fixed.replace(",]", "]")  # remove trailing commas
            data = json.loads(fixed)

        all_results.extend(data["results"])

    return all_results


if __name__ == "__main__":
    profile = """
    Basic Information
        - Likely located in an academic or professional environment with access to Google Cloud, GitHub, Slack, and Zoom.
        - Works in a technical role involving software development, machine learning, and project collaboration.
        - Uses English for professional communication.
        - Likely a dog owner, as evidenced by regular purchases of dog food, grooming supplies, and medications.
    Interests & Focus Areas
        - Technical/Research Interests:
        - Artificial intelligence (AI), machine learning, reinforcement learning.
        - AI memory systems (GUM, Gumbo, A-MEM, BondAI, Memonto).
        - Robotics and action reasoning (e.g., MolmoAct).
        - Computer vision, image processing, and 3D part segmentation.
        - API integration, CI/CD, and cloud computing (Google Cloud, Vast.ai).
        - Accessibility in video content and animation captioning.
        - Personal Interests:
        - Online shopping for pet care products (Simparica, food, grooming supplies).
    Current Activities
        - Actively coding and debugging in Python, often using Visual Studio Code.
        - Managing workflows and troubleshooting errors in GitHub Actions.
        - Configuring cloud environments (Google Cloud, Vast.ai), including storage and compute instances.
        - Engaged in projects involving accessibility in video, captioning animations, and UI improvements.
        - Preparing and organizing project-related documents in Google Docs and planning events/workshops in Google Calendar.
        - Shopping for pet supplies online (Petslike, Practik, Rozetka).
    Skills & Expertise
        - Proficient in Python programming (data processing, APIs, machine learning).
        - Experienced with Git/GitHub for version control and CI/CD.
        - Skilled in cloud infrastructure management (Google Cloud, Vast.ai).
        - Strong debugging and troubleshooting ability in development environments.
        - Familiar with machine learning models (T5, BART, DistilBERT, OpenAI/Anthropic models).
    Goals & Intentions
    - Short-term:
        - Resolve current debugging and CI/CD workflow issues.
        - Complete purchases and logistics for pet care supplies.
    - Long-term:
        - Enhance expertise in AI memory systems, decision-making frameworks, and robotics action reasoning.
        - Advance professional development in AI/ML and secure impactful applications of learned methodologies.
    Other Notable Traits
        - Work Style: Organized, methodical, and detail-oriented; relies heavily on scheduling and documentation.
        - Communication Style: Prefers structured, collaborative environments (Slack, Zoom Q&A), often more observant than leading. Uses humor and emojis in informal settings.
        - Behavioral Patterns:
        - Multitasks across music, coding, meetings, and shopping.
        - Prioritizes technical problem-solving and academic discussions over casual or administrative tasks.
        - Price-sensitive and detail-focused when shopping.
    """

    propositions = []
    with open("../analysis_output/propositions_qwen.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                propositions.append(line)

    results = check_propositions_against_profile(
        profile, propositions,
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        batch_size=100
    )
    
    with open("results_qwen.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["proposition", "supported"])
        writer.writeheader()
        writer.writerows(results)
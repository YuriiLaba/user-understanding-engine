import json
from openai import OpenAI
import os
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


def generate_user_profile(path_to_propositions: str, path_to_store_user_profile: str) -> None:
    """Generate user profile from propositions CSV file."""
    
    df = pd.read_csv(path_to_propositions)
    
    propositions_yaml = "Propositions:\n\n```yaml\n"

    for idx, row in df.iterrows():
        propositions_yaml += (
            f"- id: {idx}\n"
            f"  proposition: \"{row['proposition']}\"\n"
            f"  confidence: {row['confidence']}\n"
            f"  decay: {row['decay']}\n"
        )
    
    prompt = f"""
    You are an AI system that builds a **structured user profile** of a computer user
    from noisy, partial observations of their activity.

    You receive machine-generated **propositions** about the user.
    These propositions may be incomplete, overlapping, or sometimes incorrect.

    Your job is to:
    1. Aggregate the evidence across propositions.
    2. Separate **static** and **dynamic** aspects of the user.
    3. Produce a **concise, structured JSON profile**.
    4. Clearly express uncertainty instead of hallucinating details.

    ---------------------
    KEY CONCEPTS
    ---------------------

    1) STATIC CONTENT (slow-changing, foundational)
    Static attributes change rarely and usually only when the user explicitly changes something
    about their life or settings. They are long-term characteristics.

    Include in the static profile:

    - demographics: basic information about the person.
    Examples: name (if known), likely country/region, native language(s),
    spoken languages, approximate age range, education level, etc.

    - background_knowledge_skills:
    - background: educational and professional background, domain(s) they likely work or study in.
    - knowledge: domains where they seem to have non-trivial expertise (e.g. “machine learning”,
        “front-end development”, “economics of China”, “graphic design”).
    - skills: concrete competencies, both technical and soft (e.g. “Python programming”,
        “LaTeX”, “data analysis”, “project management”, “teaching”, “presentation skills”).

    - goals_and_needs:
    - long_term_goals: enduring objectives that span weeks/months/years
        (e.g. completing a PhD, launching a product, learning a new language).
    - short_term_goals: nearer-term objectives (e.g. finish a paper draft, debug a model,
        prepare a presentation, annotate a dataset).
    - needs: what the user seems to need from tools or systems
        (e.g. better organization, assistance with coding, help understanding research papers).

    - personality_traits:
    - cognitive/working_style: how they seem to approach work (e.g. detail-oriented,
        fast experimenter, prefers structure vs exploration, multitasker vs single-tasker).
    - collaboration_communication_style: hints about how they collaborate (e.g. prefers async,
        heavy Slack use, frequent meetings, mentoring role).
    - other_traits: any stable personality-like traits you can cautiously infer from evidence.
        Only include traits that are clearly supported by multiple propositions.
        If unsure, leave null or add to uncertainties.

    2) DYNAMIC CONTENT (fast-changing, time- and context-dependent)
    Dynamic attributes change frequently as the user works.

    Include in the dynamic profile:

    - behaviors:
    - recent_tasks: what the user appears to be doing in the captured time window
        (e.g. “coding in VS Code”, “reading academic PDFs”, “editing slides”, “chatting in Slack”).
    - workflows_patterns: recurring sequences or routines
        (e.g. “often switches between IDE and browser”, “frequently checks email between tasks”).
    - time_use: rough characterization such as “deep focus work”, “context switching”,
        “administrative work”, “communication-heavy”.

    - preferences_interests:
    - short_term_interests: topics or tools that seem important in the **recent** window
        (e.g. a particular project, paper, framework, dataset).
    - long_term_interests: topics that appear repeatedly across time and look stable.
    - tool_preferences: specific apps, services, or workflows they appear to favor
        (e.g. prefers VS Code over PyCharm, uses Notion instead of Google Docs).
    - interaction_preferences: how they seem to prefer interacting with systems
        (e.g. keyboard shortcuts vs mouse, command line vs GUI, etc., if clearly supported).

    ---------------------
    INPUT FORMAT
    ---------------------

    You will receive a **YAML list of propositions** in a ```yaml``` block at the end of this message, like:

    ```yaml
    - id: 0
    proposition: "User is reading a research paper about multimodal transformers."
    confidence: 8
    decay: 7
    - id: 1
    proposition: "User is using VS Code for Python development."
    confidence: 9
    decay: 9
    ...


    Fields:
        1. id: integer identifier of the proposition.
        2. proposition: natural-language statement about the user or their activity.
        3. confidence: 0–10 (10 = very likely correct). Higher is more trustworthy.
        4. decay: 0–10 (higher = more relevant to the current moment; lower = older / less relevant).

    Assume propositions with higher confidence are more trustworthy.
    Assume higher decay means more relevant to **current** behavior/interests;
    lower decay suggests more historical context.

    The propositions to use will be inserted where indicated below.

    ---------------------
    OUTPUT FORMAT (CRITICAL)
    ---------------------

    Return **ONLY** a single valid JSON object (no markdown, no comments, no extra text).
    Use this exact structure:

    {{
    "static_profile": {{
        "demographics": {{
        "name_or_alias": null or string,
        "country_or_region": null or string,
        "native_language": null or string,
        "other_languages": [ ... ],
        "age_range": null or string,     // e.g. "20-30", "30-40"
        "education_level": null or string, // e.g. "undergraduate", "graduate", "PhD", "unknown"
        "other_demographic_notes": []
        }},
        "background_knowledge_skills": {{
        "background": [
            // short phrases like "computer science student", "data scientist in industry"
        ],
        "knowledge_domains": [
            // topics where they likely have strong knowledge, e.g. "machine learning", "econometrics"
        ],
        "skills": [
            // concrete skills, e.g. "Python", "LaTeX", "data visualization", "teaching"
        ]
        }},
        "goals_and_needs": {{
        "long_term_goals": [
            // long-term objectives inferred from repeated patterns
        ],
        "short_term_goals": [
            // nearer-term objectives tied to recent activity
        ],
        "needs": [
            // what they seem to need from tools/systems (organization, automation, etc.)
        ]
        }},
        "personality_traits": {{
        "cognitive_working_style": [
            // e.g. "prefers structured workflows", "frequent context switching"
        ],
        "collaboration_communication_style": [
            // e.g. "works a lot in team chats", "frequent meeting participation"
        ],
        "other_traits": [
            // any carefully inferred trait; only if clearly supported
        ]
        }}
    }},
    "dynamic_profile": {{
        "behaviors": {{
        "recent_tasks": [
            // current or very recent tasks, using short phrases
        ],
        "workflows_patterns": [
            // recurring routines or multi-step patterns
        ],
        "time_use_characterization": [
            // e.g. "deep focused work", "administrative tasks", "heavy communication"
        ]
        }},
        "preferences_interests": {{
        "short_term_interests": [
            // topics/tools important in the immediate time window
        ],
        "long_term_interests": [
            // more stable interests inferred from repeated evidence
        ],
        "tool_preferences": [
            // e.g. "prefers VS Code for coding", "uses Notion for note-taking"
        ],
        "interaction_preferences": [
            // e.g. "relies on keyboard shortcuts", "uses terminal frequently"
        ]
        }}
    }}
    }}

    ---------------------
    REASONING AND AGGREGATION RULES
    ---------------------

    - Do **not** copy propositions verbatim; synthesize them into concise, non-redundant bullet points.
    - Use **higher-confidence** and **higher-decay** propositions as primary evidence.
    - If something is **not reasonably supported**, leave the field as null or an empty list.
    Do NOT invent demographic details or personality traits.
    - If evidence is weak or ambiguous, prefer to:
    - keep the field empty, and/or
    - add a short note in "meta.uncertainties".
    - When propositions conflict:
    - Prefer the cluster with higher total confidence and more recent (higher decay) evidence.
    - Document the conflict in "meta.contradictions" in 1–2 sentences.
    - Distinguish clearly between:
    - long_term_interests vs short_term_interests (based on decay / repeated evidence),
    - long_term_goals vs short_term_goals (longer vs nearer horizon),
    - static traits vs dynamic behaviors.

    ---------------------
    YOUR TASK NOW
    ---------------------

    Build the best possible **static_profile** and **dynamic_profile** you can from the propositions below,
    following the schema above and returning ONLY a JSON object.

    Here are the propositions (YAML):
    {propositions_yaml}
    """


    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10_000,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    with open(path_to_store_user_profile, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
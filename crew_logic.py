import os
import re
import json
import time
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq

def clean_text(text: str) -> str:
    """Removes excess whitespace to save tokens."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_json_payload(raw_text: str) -> dict:
    """Robustly extracts JSON from raw LLM text without requiring tool-calling."""
    try:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except Exception:
        score_match = re.search(r'"score":\s*(\d+)', raw_text)
        feedback_match = re.search(r'"feedback":\s*"([^"]+)"', raw_text)
        return {
            "score": int(score_match.group(1)) if score_match else 0,
            "feedback": feedback_match.group(1) if feedback_match else "Could not extract feedback.",
            "flagged_sections": []
        }

def optimize_cv_adversarial(cv_text: str, jd_text: str, target_score: int = 8, max_iterations: int = 3, log_callback=None):
    cv_clean = clean_text(cv_text)
    jd_clean = clean_text(jd_text)

    # THE FIX: Use ChatGroq. This forces the request to Groq's API 
    # regardless of the "openai/" prefix in the model name.
    main_llm = ChatGroq(
        model="openai/gpt-oss-20b", 
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.1
    )

    screener = Agent(
        role="Senior ATS Evaluator",
        goal="Strictly evaluate the CV against the JD and return raw JSON.",
        backstory="You are a ruthless technical recruiter. You only accept perfect matches.",
        llm=main_llm,
        verbose=False
    )

    tailorer = Agent(
        role="Targeted Resume Engineer",
        goal="Rewrite the flagged sections of the CV to address feedback.",
        backstory="You surgically update specific bullet points to maximize ATS visibility.",
        llm=main_llm,
        verbose=False
    )

    current_cv = cv_clean
    iteration = 1
    best_evaluation = {}

    while iteration <= max_iterations:
        if log_callback:
            log_callback(f"**Iteration {iteration}**: Screener is evaluating the CV...")

        eval_task = Task(
            description=(
                f"Evaluate this CV against this Job Description.\n\n"
                f"CV:\n{current_cv}\n\n"
                f"JD:\n{jd_clean}\n\n"
                f"Respond with ONLY a raw JSON object (no markdown ticks, no commentary) with these exact keys:\n"
                f"{{\n"
                f'  "score": <integer from 1 to 10>,\n'
                f'  "feedback": "<concise feedback on missing skills/experience>",\n'
                f'  "flagged_sections": ["<specific bullet point or section to rewrite>"]\n'
                f"}}"
            ),
            expected_output="A raw JSON object with keys score, feedback, and flagged_sections.",
            agent=screener
        )

        eval_crew = Crew(agents=[screener], tasks=[eval_task])
        eval_output = eval_crew.kickoff()

        parsed_eval = extract_json_payload(eval_output.raw)
        current_score = parsed_eval.get("score", 0)
        feedback = parsed_eval.get("feedback", "No feedback provided.")
        best_evaluation = parsed_eval

        if log_callback:
            log_callback(f"**Iteration {iteration} Score**: {current_score}/10\n\n*Feedback*: {feedback}")

        if current_score >= target_score:
            if log_callback:
                log_callback("✅ Target score achieved!")
            break

        if iteration == max_iterations:
            if log_callback:
                log_callback("⚠️ Max iterations reached.")
            break

        if log_callback:
            log_callback("⏳ Pausing 10s to respect Groq TPM rate limits before editing...")
        time.sleep(10)

        if log_callback:
            log_callback("⚙️ Editor is rewriting flagged sections based on feedback...")

        update_task = Task(
            description=(
                f"Revise the following CV based on this feedback:\n{feedback}\n\n"
                f"Flagged items: {parsed_eval.get('flagged_sections', [])}\n\n"
                f"Current CV:\n{current_cv}\n\n"
                f"Provide the complete revised CV text only."
            ),
            expected_output="The revised CV text.",
            agent=tailorer
        )

        update_crew = Crew(agents=[tailorer], tasks=[update_task])
        update_output = update_crew.kickoff()

        current_cv = update_output.raw
        iteration += 1
        
        if iteration <= max_iterations:
            time.sleep(10)

    return {
        "final_score": best_evaluation.get("score", 0),
        "final_feedback": best_evaluation.get("feedback", ""),
        "updated_cv": current_cv,
        "iterations_used": min(iteration, max_iterations)
    }

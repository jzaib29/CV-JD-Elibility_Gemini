import os
import json
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM

# Define strict output schema for the Evaluator
class EvaluationResult(BaseModel):
    score: int = Field(description="The ATS compatibility score from 1 to 10")
    feedback: str = Field(description="Specific feedback on what is missing or needs improvement")
    flagged_sections: list[str] = Field(description="Exact sections or bullet points to rewrite")

def optimize_cv_adversarial(cv_text: str, jd_text: str, target_score: int = 8, max_iterations: int = 3, log_callback=None):
    # Initialize the specific gpt-oss-120b model and explicitly pass the key
    main_llm = LLM(
        model="openai/gpt-oss-120b",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.1 
    )

    screener = Agent(
        role="Senior ATS Evaluator",
        goal="Strictly evaluate the CV against the JD and output a JSON evaluation.",
        backstory="You are a ruthless technical recruiter. You only accept perfect matches. You never inflate scores.",
        llm=main_llm,
        verbose=False
    )

    tailorer = Agent(
        role="Targeted Resume Engineer",
        goal="Rewrite the flagged sections of the CV to address the recruiter's feedback.",
        backstory="You surgically update specific bullet points to maximize ATS visibility without altering the document structure.",
        llm=main_llm,
        verbose=False
    )

    current_cv = cv_text
    iteration = 1
    best_evaluation = None

    while iteration <= max_iterations:
        if log_callback:
            log_callback(f"**Iteration {iteration}**: Screener is evaluating the CV...")
        
        # Step 1: Evaluate
        eval_task = Task(
            description=f"Evaluate this CV:\n{current_cv}\n\nAgainst this JD:\n{jd_text}\n\nBe ruthless.",
            expected_output="JSON containing score, feedback, and flagged_sections.",
            output_json=EvaluationResult,
            agent=screener
        )
        
        eval_crew = Crew(agents=[screener], tasks=[eval_task])
        eval_output = eval_crew.kickoff()
        
        try:
            parsed_eval = eval_output.json_dict
            current_score = parsed_eval.get('score', 0)
            feedback = parsed_eval.get('feedback', '')
        except Exception:
            current_score = 0
            feedback = "Failed to parse feedback."

        best_evaluation = parsed_eval
        
        if log_callback:
            log_callback(f"**Iteration {iteration} Score**: {current_score}/10\n\n*Feedback*: {feedback}")

        if current_score >= target_score:
            if log_callback:
                log_callback("✅ Target score achieved!")
            break
            
        if iteration == max_iterations:
            if log_callback:
                log_callback("⚠️ Max iterations reached. Ending optimization loop.")
            break

        if log_callback:
            log_callback("⚙️ Editor is rewriting flagged sections based on feedback...")

        # Step 2: Revise
        update_task = Task(
            description=f"Revise the following CV based on this strict feedback: {feedback}\n\nCurrent CV:\n{current_cv}",
            expected_output="The fully updated CV text. Do not output anything else.",
            agent=tailorer
        )
        
        update_crew = Crew(agents=[tailorer], tasks=[update_task])
        update_output = update_crew.kickoff()
        
        current_cv = update_output.raw
        iteration += 1

    return {
        "final_score": best_evaluation.get('score', 0) if best_evaluation else 0,
        "final_feedback": best_evaluation.get('feedback', '') if best_evaluation else '',
        "updated_cv": current_cv,
        "iterations_used": min(iteration, max_iterations)
    }

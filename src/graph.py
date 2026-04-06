from typing import Literal
from langchain_ollama import ChatOllama
import pdfplumber
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
import asyncio
from typing import TypedDict
import os
import tempfile
import subprocess
import subprocess
import sys
from pypdf import PdfReader
import shutil
from configuration import LLMConfiguration
from states import (
    ResumeSections,
    ResumeSection,
    JDKeywords,
    AgentState,
    EvaluatorOutput,

)
from prompts import (
    rewrite_section_prompt,
    section_resume_prompt,
   # correct_latex_prompt,
    extract_jd_keyword_prompt,
    rewrite_prompt,
    evaluator_prompt
)
 
async def section_resume(state: AgentState, config: RunnableConfig)  :
    cfg = LLMConfiguration.from_runnable_config(config)
    llm = cfg.get_smart_llm().with_structured_output(ResumeSections)

    chain = section_resume_prompt | llm 
    
    res: ResumeSections = await chain.ainvoke({
        "stringified_resume": state.get("stringified_resume")
    })

    return {'sections': res.sections }

async def rewrite_sections(state: AgentState, config: RunnableConfig ):
    cfg = LLMConfiguration.from_runnable_config(config)
    num_workers = cfg.get_optimal_workers()

    concurrency = len(state["sections"]) if num_workers == -1 else num_workers
    semaphore = asyncio.Semaphore(concurrency)

    async def rewrite_section(section, index):
        async with semaphore:
            llm = cfg.get_smart_llm(endpoint_index=index).with_structured_output(ResumeSection)
            return await (rewrite_section_prompt | llm).ainvoke(
                {
                    "section": f"{section.name}:\n{section.content}",
                    "keywords_and_skills": state["keywords_and_skills"]
                })

    if num_workers == -1:
        tasks = [rewrite_section(section, 0) for section in state["sections"]]
    else:
        tasks = [
            rewrite_section(section, i % num_workers)
            for i, section in enumerate(state["sections"])
        ]
    results: list[ResumeSection] = await asyncio.gather(*tasks)
    return {"rewritten_sections": results}

async def extract_jd_keyword(state: AgentState, config: RunnableConfig ):
    cfg = LLMConfiguration.from_runnable_config(config)
    llm = cfg.get_smart_llm().with_structured_output(JDKeywords)

    chain = extract_jd_keyword_prompt | llm 

    res = await chain.ainvoke({'job_description': state["job_description"]})

    keywords_and_skills = (', ').join(res.keywords)

    return {"keywords_and_skills": keywords_and_skills}
    
async def rewrite_full_resume(
        state: AgentState, 
        config: RunnableConfig,  
    ):
  
    cfg = LLMConfiguration.from_runnable_config(config)
    llm = cfg.get_smart_llm() 

    plain_text = '\n'.join(
        f'{s.name}:{s.content}' if isinstance(s, ResumeSection) else str(s)
        for s in state["rewritten_sections"]
    )

    consumed_suggestions = state.get("suggestions") or ["first attempt"]
    consumed_errors = state.get("code_errors") or ["first attempt"]
    prompt_input = {
        'plain_text_resume': plain_text,
        'suggestions': consumed_suggestions,
        'previous_attempt': state.get("resume_code") or "first attempt",
        'resume_output_path': state["resume_output_path"],
        'code_errors':  consumed_errors
    }

    chain = rewrite_prompt | llm
 
    res = await chain.ainvoke(prompt_input)
    return {
        'resume_code': res.content,
        'num_attempts': state.get("num_attempts", 0) + 1,
    }

async def evaluator(state, config: RunnableConfig, ):
    code_errors = []
    suggestions = []

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(state["resume_code"])
        tmp_path = f.name

    def _run_script() -> subprocess.CompletedProcess:
        """Blocking call — safe to run in a thread."""
        return subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    
    try:
        result = await asyncio.to_thread(_run_script)
        if result.returncode != 0:
            code_errors.append(result.stderr)
    except subprocess.TimeoutExpired:
        code_errors.append("Script timed out after 30 seconds.")
    except Exception as exc:
        code_errors.append(f"Script execution error: {exc}")
    finally:
        os.unlink(tmp_path)

    verdict = False

    if not code_errors:
        try:
            reader = PdfReader(state["resume_output_path"])
            if len(reader.pages) != 1:
                suggestions.append(
                    f"PDF is {len(reader.pages)} pages — must be exactly 1. "
                    "Reduce font sizes or margins."
                )
        except Exception as e:
            code_errors.append(f"Could not read output PDF: {e}")

        verdict = False
    if not code_errors:   
        cfg = LLMConfiguration.from_runnable_config(config)
        llm = cfg.get_smart_llm().with_structured_output(EvaluatorOutput)
        chain = evaluator_prompt | llm
 
        res: EvaluatorOutput = await chain.ainvoke({
            'resume_code': state["resume_code"]
        })
        suggestions.extend(res.suggestions)
        verdict = res.verdict and len(suggestions) == 0

    return {
        'code_errors': code_errors,
        'suggestions': suggestions,
        'verdict': verdict,
    }

def should_rewrite(state: AgentState, config: RunnableConfig):
    cfg = LLMConfiguration.from_runnable_config(config)
    if state["verdict"] or state["num_attempts"] >= cfg.max_rewrite_attempts:
        return "PASS"
    return 'FAIL'

def build_graph():
    
    graph = StateGraph(AgentState)

    graph.add_node("section_resume", section_resume)
    graph.add_node("rewrite_sections", rewrite_sections)
    graph.add_node("extract_jd_keyword", extract_jd_keyword)
    graph.add_node("rewrite_full_resume", rewrite_full_resume)
    graph.add_node("evaluator", evaluator)

    graph.add_edge(START, "extract_jd_keyword")
    graph.add_edge("extract_jd_keyword", "section_resume")
    graph.add_edge("section_resume", "rewrite_sections")
    graph.add_edge("rewrite_sections", "rewrite_full_resume")
    graph.add_edge("rewrite_full_resume", "evaluator")

    graph.add_conditional_edges(
        "evaluator",
        should_rewrite,
        {
            "PASS": END,
            "FAIL": "rewrite_full_resume",
        }

    )
    app = graph.compile()
    return app
 
 
if __name__ == "__main__":
    import textwrap
    import argparse
    parser = argparse.ArgumentParser(description="Resume tailoring pipeline")
 
    args = parser.parse_args()
 
    
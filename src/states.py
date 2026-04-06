from typing import Dict, List, Optional, TypedDict, Any, Annotated
from pydantic import BaseModel, Field
from operator import add
class SummarizerState(BaseModel):
    Name: str
    sections: Optional[ Dict[ str, Any ] ]



class ResumeSection(BaseModel):
    name: str = Field(description="Section name e.g. 'name', 'skills', 'experience' ")
    content: str = Field(description="Raw content of the section")

class ResumeSections(BaseModel):
    sections: list[ResumeSection] = Field(description="List of resume sections")

 

class AgentState(TypedDict):
    stringified_resume: str
    sections: list[ResumeSection]
    rewritten_sections: list[ResumeSection]
    job_description: str
    final_resume: str
    tex_path: str
    keywords_and_skills: str
    resume_output_path: str
    resume_code: str
    code_errors: list[str]
    suggestions: list[str]
    num_attempts: int
    verdict: bool

class JDKeywords(BaseModel):
    keywords: list[str]

class EvaluatorOutput(BaseModel):
    suggestions: list[str]
    verdict: bool
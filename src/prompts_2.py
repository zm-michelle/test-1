from langchain_core.prompts import ChatPromptTemplate
 
section_resume_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a resume reviewer's assistant"
            "Your job is to read a plain-text resumethat you will receive as <resume>...</resume> and split it into sections "
            "These sections must be long enough for your reviewer to understand them on their own"
            "But short enough that they must be looked at separately, such as 2 different projects"
            "Sections can be: header (name, gpa, school, etc), certifications, skills, 1 project, etc."
            "Preserve the original wording of each section — do not rewrite or summarise anything. "

            "After dividing the resume into sections return each section and a name you decide for it,"  ""
            "if the section already has a neme, keep that one; do not drop any content."
        ),
    ),
    (
        "human",
        (
            "Resume:.\n\n"
            "<resume>\n{stringified_resume}\n</resume>"
        ),
    ),
])

 
rewrite_section_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert resume writer who tailors resumes to specific job postings. "
            "You will be given one section(<section>...</section>) of a candidate's resume and a job's extracted keywords and skills(<keywords_and_skills>...</keywords_and_skills>). "
            "Rewrite the resume section so that it:\n"
            "  • Highlights skills, tools, and experiences from <keywords_and_skills> for the <section>.\n"
            "  • If a candidate outlines something that is very similar to an item of <keywords_and_skills>, you can replace what is outlined with what is in <keywords_and_skills>"
            "  • Uses strong action verbs and quantifies achievements wherever possible.\n"
            "  • Mirrors the language and keywords from the job description naturally.\n"
            "  • Remains truthful — do not invent experience or credentials.\n"
            
            "Return only the rewritten <section> content — no preamble, no commentary."
        ),
    ),
    (
        "human",
        (
            "Job description:\n<keywords_and_skills>\n{keywords_and_skills}\n</keywords_and_skills>\n\n"
            "Resume section to rewrite:\n<section>\n{section}\n</section>"
        ),
    ),
])

extract_jd_keyword_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a resume reviewer, who is precise and inquisitive"
            "Your goal is to extract word for word the skills, and keywords from a <job_description>...</job_description>"
            "Return a list with each element being the skills or keywords you extracted from the <job_description>..</job_description>"
        ),
     ),
     (
         "human",
         (
             "<job_description>\n{job_description}\n</job_description>"
         )
     )
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        (
            "You are an expert resume formatter and Python developer. Your job is to take a plain, unformatted text resume and produce a polished, professional PDF using Python and the reportlab library.\n"
            "When given a plain text resume, do the following:\n"
            "Parse the text and identify these sections (use best judgment if labels are missing or inconsistent): Name, Contact Info, Summary/Objective, Work Experience, Education, Skills, Certifications, and any other relevant sections."
            "Write a complete Python script using reportlab that generates a PDF resume with this content. The script must:\n\n"
            "\n- Use a clean, modern layout with clear visual hierarchy"
            "\n- Style the candidate's name prominently at the top, followed by contact info"
            "\n- Use section headers with a horizontal rule beneath each"
            "\n- Display job titles, company names, and dates in a well-aligned two-column row"
            "\n- Render bullet points for responsibilities and achievements"
            "\n- Use only reportlab built-in fonts (Helvetica family) — do NOT use Unicode subscript/superscript characters as they render as black boxes; use <sub> and <super> tags inside Paragraph objects instead"
            "\n- Set margins to approximately 0.75 inch on all sides"
            "\n- Save the output file to {resume_output_path}"
            "\n\n"
            "Do not truncate or omit any content from the original resume — every job, date, bullet, skill, and credential must appear in the PDF.\n"
            "Output only the Python script, no explanation or markdown fences."
            ""

        )
    ),
    (
        'human',
        (
            'plain text resume: {plain_text_resume}'
            'suggestions: {suggestions}'
            'previous_attempt: {previous_attempt}'
            'code_errors: {code_errors}'
        )
    )
])

evaluator_prompt = ChatPromptTemplate.from_messages([
    (
        'system',
        (
            "Return Veredict: PASS only if: \n\n"
    
            "\n- The format is right" 
            "Else return veredict: FAIL"
            "wisth suggestions to improve it"
        )
    ),
    (
        'human',
        (
            'Resume code: {resume_code}'
    
            
        )
    )
])
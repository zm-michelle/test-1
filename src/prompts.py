from langchain_core.prompts import ChatPromptTemplate

section_resume_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a resume reviewer's assistant. "
            "Your job is to read a plain-text resume that you will receive inside <resume>...</resume> "
            "and split it into its distinct sections. "
            "Each section must be self-contained and long enough for a reviewer to understand it on its own, "
            "but scoped to a single topic — for example, one project, one job, or one credential. "
            "Valid section types include: header (name, contact info, GPA, school), summary, "
            "skills, one work experience entry, one project, certifications, awards, etc. "
            "Preserve every word of the original text exactly — do not rewrite, summarise, or omit anything. "
            "If a section already has a heading, keep it. If it does not, assign a short descriptive name."
        ),
    ),
    (
        "human",
        "Resume:\n\n<resume>\n{stringified_resume}\n</resume>",
    ),
])


rewrite_section_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert resume writer who tailors resumes to specific job postings. "
            "You will receive one section of a candidate's resume inside <section>...</section> "
            "and a list of keywords and skills extracted from the target job inside <keywords_and_skills>...</keywords_and_skills>.\n\n"
            "Rewrite the section so that it:\n"
            "  • Highlights skills, tools, and experiences that overlap with <keywords_and_skills>.\n"
            "  • Substitutes a candidate's phrasing with a keyword from <keywords_and_skills> ONLY when "
            "the candidate's described experience is functionally equivalent — never invent, exaggerate, "
            "or imply experience the candidate does not have.\n"
            "  • Uses strong action verbs and quantifies achievements wherever the original text supports it.\n"
            "  • Mirrors the language and terminology from the job description naturally.\n"
            "  • Remains completely truthful — do not add credentials, tools, or responsibilities "
            "that are not present in the original section.\n\n"
            "Return only the rewritten section content — no preamble, no commentary, no labels."
        ),
    ),
    (
        "human",
        (
            "Keywords and skills from job description:\n"
            "<keywords_and_skills>\n{keywords_and_skills}\n</keywords_and_skills>\n\n"
            "Resume section to rewrite:\n"
            "<section>\n{section}\n</section>"
        ),
    ),
])


extract_jd_keyword_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a precise job description analyst. "
            "Your goal is to extract every skill, tool, technology, methodology, and domain keyword "
            "from the job description provided inside <job_description>...</job_description>.\n\n"
            "Rules:\n"
            "  • Return each item as a short, normalised term — for example 'Python', 'REST APIs', "
            "'CI/CD', 'cross-functional collaboration', 'AWS Lambda'.\n"
            "  • Do NOT copy full sentences or fragments. Extract the concept, not the surrounding prose.\n"
            "  • Include both hard skills (tools, languages, frameworks) and soft skills or domain terms "
            "if the job description explicitly calls them out.\n"
            "  • Do not add anything that is not present in the job description."
        ),
    ),
    (
        "human",
        "<job_description>\n{job_description}\n</job_description>",
    ),
])


rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert resume formatter and Python developer. "
            "Your job is to take a plain-text resume and produce a polished, professional, single-page PDF "
            "using Python and the reportlab library.\n\n"
            "Instructions:\n"
            "  • Parse the text and identify all sections: Name, Contact Info, Summary/Objective, "
            "Work Experience, Education, Skills, Certifications, and any other sections present.\n"
            "  • Write a complete, runnable Python script using reportlab that generates the PDF. "
            "The script must:\n"
            "      - Use a clean, modern layout with clear visual hierarchy.\n"
            "      - Style the candidate's name prominently at the top, followed by contact info.\n"
            "      - Use section headers with a horizontal rule beneath each.\n"
            "      - Display job titles, company names, and dates in a well-aligned two-column row.\n"
            "      - Render bullet points for responsibilities and achievements.\n"
            "      - Use only reportlab built-in fonts (Helvetica family). "
            "Do NOT use Unicode subscript or superscript characters — they render as black boxes. "
            "Use <sub> and <super> tags inside Paragraph objects instead.\n"
            "      - Set margins to approximately 0.75 inch on all sides.\n"
            "      - Fit all content on exactly ONE page — reduce font sizes or margins if needed.\n"
            "      - Save the output file to: {resume_output_path}\n\n"
            "  • Do not truncate or omit any content — every job, date, bullet, skill, "
            "and credential must appear in the PDF.\n\n"
            "Output raw Python code only. "
            "Do not include markdown fences, backticks, language tags, or any explanatory prose. "
            "The very first character of your response must be the letter 'i' (from 'import')."
        ),
    ),
    (
        "human",
        (
            "Plain text resume:\n{plain_text_resume}\n\n"
            "Suggestions from previous attempt:\n{suggestions}\n\n"
            "Previous attempt code:\n{previous_attempt}\n\n"
            "Errors from previous attempt:\n{code_errors}"
        ),
    ),
])


evaluator_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a strict resume code reviewer. "
            "You will receive a Python script that generates a PDF resume using reportlab.\n\n"
            "Evaluate the script against ALL of the following criteria:\n"
            "  1. Every standard resume section is present: Name, Contact Info, at least one of "
            "(Work Experience / Projects / Education), and Skills.\n"
            "  2. Bullet points use strong action verbs — not passive phrases like 'was responsible for'.\n"
            "  3. No placeholder text remains (e.g. 'Lorem ipsum', 'TBD', 'INSERT HERE', '[Company]').\n"
            "  4. No content from the plain-text resume appears to have been dropped or truncated.\n"
            "  5. The output path passed to SimpleDocTemplate is not hardcoded to a different value "
            "than what was requested.\n"
            "  6. No Unicode characters likely to render as black boxes are used outside of "
            "<sub>/<super> Paragraph tags.\n\n"
            "Return verdict: true (PASS) only if ALL criteria are met and you have no suggestions. "
            "Return verdict: false (FAIL) with a specific, actionable list of suggestions otherwise. "
            "Keep each suggestion to one sentence."
        ),
    ),
    (
        "human",
        "Resume generation script:\n{resume_code}",
    ),
])
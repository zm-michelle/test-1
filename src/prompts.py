from langchain_core.prompts import ChatPromptTemplate
 

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
            "skills, certifications, awards, etc. "
            "CRITICAL: each individual job must be its own separate section — never merge "
            "two jobs into one section. Each project must also be its own separate section. "
            "If the resume has 2 jobs, you must return 2 separate experience sections. "
            "If it has 3 projects, return 3 separate project sections. "
            "Count the jobs and projects in the resume and verify your output has the same count.\n"
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
            "You are an expert LinkedIn resume writer who tailors resumes to specific job postings. "
            "You will receive one section of a candidate's resume inside <section>...</section> "
            "and a list of keywords and skills extracted from the target job inside <keywords_and_skills>...</keywords_and_skills>.\n\n"

            "STRICT RULES — follow all of these without exception:\n"
            "  • You may ONLY rephrase what already exists in the original section. "
            "Never swap, add, or imply a tool, technology, or platform that is not "
            "explicitly named in the original. "
            "For example: if the original says 'SQLite', do NOT change it to 'Redis' or 'PostgreSQL'. "
            "If the original says 'sockets', do NOT change it to 'FastAPI'. "
            "If a keyword from the job description is not already in the section, skip it entirely.\n"
            "  • For skills sections: preserve EVERY item listed. Do not drop any language, "
            "framework, or tool — if it appears in the original, it must appear in the rewrite. "
            "You may add a keyword from <keywords_and_skills> to the skills section ONLY if it "
            "is functionally identical to something already listed.\n"
            "  • For header sections: always include the candidate's full name as the "
            "very first line of the rewritten content — never drop it.\n"
            "  • Use strong action verbs and quantify achievements wherever the original text supports it.\n"
            "  • Remains completely truthful — do not add credentials, tools, or responsibilities "
            "that are not present in the original section.\n"
            "  • Expand each bullet point to be descriptive and impactful — aim for 1-2 full sentences "
            "per bullet. Add context about the impact, scale, or outcome wherever the original supports it. "
            "For example, instead of '- Built REST API' write "
            "'- Engineered and deployed a RESTful API using FastAPI, enabling real-time data access "
            "for internal stakeholders and reducing manual reporting time.' "
            "Only expand using information implied or stated in the original — do not invent metrics "
            "or outcomes that aren't there.\n\n"

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


rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a resume data extractor. "
            "You will receive a plain-text resume and must output a single Python script "
            "by filling in the TEMPLATE below.\n\n"

            "RULES:\n"
            "  1. Copy the template EXACTLY — do not rename variables, reorder blocks, "
            "add functions, add imports, or wrap anything in if __name__ == '__main__'.\n"
            "  2. Replace every <PLACEHOLDER> with the real value from the resume.\n"
            "  3. For bullet lists, produce a Python list of plain strings, "
            "one string per bullet, e.g. ['Led team of 5 engineers', 'Reduced latency by 30%'].\n"
            "  4. If a section is absent from the resume, set its list to [].\n"
            "  5. Output raw Python only — no markdown fences, no backticks, no comments.\n"
            "  6. In EXPERIENCE and PROJECTS tuples, the date field must ALWAYS be a short "
            "date string like 'Jan 2023 – Present' or 'June 2023 – Aug 2023' or empty string ''. "
            "NEVER put sentence text, bullet content, or descriptions into the date field.\n\n"

            "════════════ TEMPLATE ════════════\n"
            "from reportlab.lib.pagesizes import letter\n"
            "from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle\n"
            "from reportlab.lib.units import inch\n"
            "from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY\n"
            "from reportlab.lib import colors\n"
            "from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle\n"
            "\n"
            "OUTPUT_PATH = '{resume_output_path}'\n"
            "NAME        = '<CANDIDATE_FULL_NAME>'\n"
            "CONTACT     = '<EMAIL>  •  <PHONE>  •  <LINKEDIN>  •  <LOCATION>'\n"
            "SUMMARY     = '<ONE_PARAGRAPH_SUMMARY_OR_EMPTY_STRING>'\n"
            "\n"
            "EXPERIENCE = [\n"
            "    ('<JOB_TITLE>', '<COMPANY, CITY>', '<DATE_RANGE>', ['<BULLET_1>', '<BULLET_2>']),\n"
            "]\n"
            "\n"
            "PROJECTS = [\n"
            "    ('<PROJECT_NAME>', '<DATE_OR_EMPTY_STRING>', ['<BULLET_1>', '<BULLET_2>']),\n"
            "    # DATE_OR_EMPTY_STRING must be a short date like 'Jan 2023 - Present' or ''.\n"
            "    # NEVER put sentence text or bullet content into the date field.\n"
            "    # All descriptive content belongs in the bullets list only.\n"
            "]\n"
            "\n"
            "EDUCATION = [\n"
            "    ('<DEGREE>', '<SCHOOL, LOCATION>', '<DATE>', '<GPA or empty string>'),\n"
            "]\n"
            "\n"
            "CERTIFICATIONS = ['<CERT_1>']\n"
            "\n"
            "SKILLS = [\n"
            "    ('<CATEGORY>', '<SKILLS_COMMA_SEPARATED>'),\n"
            "]\n"
            "\n"
            "doc = SimpleDocTemplate(\n"
            "    OUTPUT_PATH, pagesize=letter,\n"
            "    leftMargin=0.65*inch, rightMargin=0.65*inch,\n"
            "    topMargin=0.6*inch, bottomMargin=0.6*inch,\n"
            ")\n"
            "W = letter[0] - 1.3*inch\n"
            "story = []\n"
            "ss = getSampleStyleSheet()\n"
            "def S(**kw): return ParagraphStyle('_', **kw)\n"
            "name_style    = S(fontName='Helvetica-Bold', fontSize=17, leading=20, alignment=TA_CENTER)\n"
            "contact_style = S(fontName='Helvetica', fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor('#444444'))\n"
            "summary_style = S(fontName='Helvetica', fontSize=9.5, leading=13, alignment=TA_JUSTIFY)\n"
            "hdr_style     = S(fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=colors.black, spaceAfter=1)\n"
            "title_style   = S(fontName='Helvetica-Bold', fontSize=9.5, leading=12)\n"
            "right_style   = S(fontName='Helvetica', fontSize=9, leading=12, alignment=TA_RIGHT)\n"
            "date_style    = S(fontName='Helvetica-Oblique', fontSize=9, leading=11, alignment=TA_RIGHT)\n"
            "bullet_style  = S(fontName='Helvetica', fontSize=8.5, leading=11, leftIndent=12, firstLineIndent=0, spaceBefore=1)\n"
            "skill_style   = S(fontName='Helvetica', fontSize=9, leading=12)\n"
            "\n"
            "def section_header(title):\n"
            "    story.append(Spacer(1, 6))\n"
            "    story.append(Paragraph(title, hdr_style))\n"
            "    story.append(HRFlowable(width='100%', thickness=0.75, color=colors.black, spaceAfter=3))\n"
            "\n"
            "story.append(Paragraph(NAME, name_style))\n"
            "story.append(Paragraph(CONTACT, contact_style))\n"
            "story.append(Spacer(1, 4))\n"
            "\n"
            "if SUMMARY:\n"
            "    section_header('PROFESSIONAL SUMMARY')\n"
            "    story.append(Paragraph(SUMMARY, summary_style))\n"
            "\n"
            "if EXPERIENCE:\n"
            "    section_header('PROFESSIONAL EXPERIENCE')\n"
            "    for job_title, company, date, bullets in EXPERIENCE:\n"
            "        row = Table([[Paragraph(job_title, title_style), Paragraph(date, date_style)]],\n"
            "                    colWidths=[W*0.65, W*0.35])\n"
            "        row.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),0)]))\n"
            "        story.append(row)\n"
            "        story.append(Paragraph(company, right_style))\n"
            "        for b in bullets:\n"
            "            story.append(Paragraph(f'• {{b}}', bullet_style))\n"
            "        story.append(Spacer(1, 4))\n"
            "\n"
            "if PROJECTS:\n"
            "    section_header('PROJECTS')\n"
            "    for proj_name, date, bullets in PROJECTS:\n"
            "        row = Table([[Paragraph(proj_name, title_style), Paragraph(date, date_style)]],\n"
            "                    colWidths=[W*0.65, W*0.35])\n"
            "        row.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),0)]))\n"
            "        story.append(row)\n"
            "        for b in bullets:\n"
            "            story.append(Paragraph(f'• {{b}}', bullet_style))\n"
            "        story.append(Spacer(1, 4))\n"
            "\n"
            "if EDUCATION:\n"
            "    section_header('EDUCATION')\n"
            "    for degree, school, date, gpa in EDUCATION:\n"
            "        row = Table([[Paragraph(degree, title_style), Paragraph(date, date_style)]],\n"
            "                    colWidths=[W*0.65, W*0.35])\n"
            "        row.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),0)]))\n"
            "        story.append(row)\n"
            "        story.append(Paragraph(school + (f'  •  GPA: {{gpa}}' if gpa else ''), right_style))\n"
            "        story.append(Spacer(1, 4))\n"
            "\n"
            "if CERTIFICATIONS:\n"
            "    section_header('CERTIFICATIONS')\n"
            "    for cert in CERTIFICATIONS:\n"
            "        story.append(Paragraph(f'• {{cert}}', bullet_style))\n"
            "\n"
            "if SKILLS:\n"
            "    section_header('SKILLS')\n"
            "    for category, items in SKILLS:\n"
            "        story.append(Paragraph(f'<b>{{category}}:</b> {{items}}', skill_style))\n"
            "        story.append(Spacer(1, 2))\n"
            "\n"
            "doc.build(story)\n"
            "════════════ END TEMPLATE ════════════\n"
        ),
    ),
    (
        "human",
        (
            "Plain text resume (fill the placeholders from this):\n{plain_text_resume}\n\n"
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
            "You are reviewing a Python script that fills data into a fixed reportlab template. "
            "The structure is correct — only evaluate the DATA values.\n\n"
            "Check ALL of the following:\n"
            "  1. NAME is the candidate's real full name, not a placeholder like '<CANDIDATE_FULL_NAME>'.\n"
            "  2. CONTACT contains real email, phone, and location — no angle brackets, "
            "no 'INSERT' text, and no consecutive '•' separators with nothing between them.\n"
            "  3. EXPERIENCE, EDUCATION, and SKILLS are all populated — none are empty lists [].\n"
            "  4. No bullet string contains placeholder text like 'TBD', '[Company]', or 'Lorem ipsum'.\n"
            "  5. The OUTPUT_PATH variable has not been changed from the value assigned at the top of the script.\n"
            "  6. In EXPERIENCE and PROJECTS tuples, the date field (second or third element) must be "
            "a short date string like 'Jan 2023 – Present' or empty string '' — never a sentence, "
            "never bullet content, never descriptive text.\n"
            "  7. GPA must be copied exactly from the resume — do not append honours, "
            "classifications, or anything not present in the original.\n\n"
            "Return verdict: true only if ALL criteria pass and you have zero suggestions. "
            "Return verdict: false with a short, specific list of what needs fixing otherwise."
        ),
    ),
    (
        "human",
        "Resume generation script:\n{resume_code}",
    ),
])

 
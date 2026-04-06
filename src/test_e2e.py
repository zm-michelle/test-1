"""
End-to-end smoke test for ResuMatch.

Run from inside the container:
    python test_e2e.py

It will:
1. Create a session
2. Submit a resume + job description
3. Poll the SSE stream until done or failed
4. Download the PDF to ./test_resume_output.pdf
"""

import json
import time
import httpx

#BASE = "http://localhost:8000"
BASE= "http://api:8000"
RESUME = """
John Doe
john.doe@email.com | (812) 555-1234 | linkedin.com/in/johndoe | github.com/johndoe

EDUCATION
Bachelor of Science in Computer Science
Indiana University, Bloomington — GPA: 3.7
May 2024

EXPERIENCE
Software Engineering Intern — Acme Corp (June 2023 – Aug 2023)
- Built REST endpoints using Flask and PostgreSQL for an internal reporting tool
- Reduced query latency by 40% by adding indexes and rewriting N+1 queries
- Wrote unit tests with pytest achieving 85% code coverage

Research Assistant — IU Luddy School (Jan 2023 – May 2023)
- Scraped and cleaned 50k+ records using BeautifulSoup and pandas
- Trained a logistic regression classifier with scikit-learn; achieved 91% accuracy
- Presented findings at departmental research symposium

PROJECTS
Personal Budget Tracker (Python, FastAPI, SQLite, React)
- Designed REST API with FastAPI and JWT auth; deployed on a DigitalOcean droplet
- Built React dashboard with Chart.js for spending visualisation

Distributed Key-Value Store (Python, sockets)
- Implemented a simplified Raft consensus algorithm across 3 nodes
- Handled leader election, log replication, and fault tolerance

SKILLS
Languages: Python, JavaScript, SQL, Java, C
Frameworks: FastAPI, Flask, React, LangChain
Tools: Docker, Git, Redis, PostgreSQL, pytest, pandas, scikit-learn
"""

JOB_DESCRIPTION = """
Backend Software Engineer — DataFlow Inc.

We are looking for a Python backend engineer to join our platform team.

Requirements:
- 1+ years of experience with Python backend development
- Proficiency with FastAPI or Django REST Framework
- Experience with Redis for caching and async task queues (Celery)
- Strong SQL skills — PostgreSQL preferred
- Familiarity with Docker and containerised deployments
- Experience writing tests with pytest
- REST API design best practices
- Familiarity with LangChain or LLM tooling is a strong plus
- Git and code review workflows

Nice to have:
- React or any frontend experience
- Experience with distributed systems or consensus protocols
- Cloud deployment (AWS, GCP, or DigitalOcean)

We value clean, well-tested code and clear communication.
"""


def main():
    with httpx.Client(base_url=BASE,timeout=None) as client:

        # ── 1. Create session ────────────────────────────────────────────────
        print("Creating session...")
        resp = client.post("/auth/session")
        resp.raise_for_status()
        session_id = resp.cookies.get("session_id")
        print(f"  session_id: {session_id[:12]}...")

        # ── 2. Submit resume + JD ────────────────────────────────────────────
        print("Submitting resume and job description...")
        resp = client.post(
            "/start_session",
            json={"resume_text": RESUME.strip(), "job_description": JOB_DESCRIPTION.strip()},
            cookies={"session_id": session_id},
        )
        resp.raise_for_status()
        body = resp.json()
        job_id = body["job_id"]
        print(f"  job_id: {job_id}")
        print(f"  stream: {body['stream_url']}")

        # ── 3. Poll SSE stream ───────────────────────────────────────────────
        print("Streaming progress (waiting for done/error)...")
        with client.stream(
            "GET",
            f"/start_session/{job_id}/stream",
            cookies={"session_id": session_id},
            timeout=None,  # graph can take a while on CPU
        ) as stream:
            for line in stream.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[len("data:"):].strip())
                event = payload.get("event")
                node  = payload.get("node", "")
                print(f"  [{event}] node={node}")
                if event in ("done", "error"):
                    if event == "error":
                        print(f"  ERROR: {payload.get('data', {}).get('error')}")
                        return
                    break

        # ── 4. Download PDF ──────────────────────────────────────────────────
        print("Downloading PDF...")
        resp = client.get("/pdf", cookies={"session_id": session_id})
        resp.raise_for_status()
        out = "test_resume_output.pdf"
        with open(out, "wb") as f:
            f.write(resp.content)
        print(f"  Saved to {out} ({len(resp.content):,} bytes)")
        print("Done! Copy to your Mac with:")
        print("  docker cp resumemaxer-api-1:/app/src/test_resume_output.pdf ~/Desktop/resume.pdf")


if __name__ == "__main__":
    main()

"""
Seed script — populates the database with 10 realistic candidates
and pre-computes rankings for job_id=1 (Senior ML Engineer).
Scores match the frontend mock data exactly for a consistent demo.
"""
import sys
import os

# Allow running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import SessionLocal, create_tables
from app.models.candidate import Candidate, CandidateScore
from app.models.job import Job

JOB_DESCRIPTION = """Senior ML Engineer — LLM Infrastructure

We're building the next generation of LLM-powered products and need a senior engineer to own our ML infrastructure stack. This role sits at the intersection of research and production.

You'll:
• Design and maintain our model training and serving infrastructure
• Lead fine-tuning efforts for domain-specific models
• Optimize inference pipelines for cost and latency
• Collaborate with research to ship new capabilities quickly

Requirements:
• 5+ years ML engineering experience
• Production LLM experience (fine-tuning, serving, RLHF a strong plus)
• Deep familiarity with distributed training
• Strong Python; C++/CUDA experience valued
• Experience with MLOps tooling (MLflow, Kubeflow, or equivalent)"""

CANDIDATES = [
    {
        "name": "Priya Venkataraman",
        "email": "priya.v@techmail.com",
        "title": "Senior ML Engineer",
        "location": "San Francisco, CA",
        "skills": ["python", "pytorch", "distributed training", "llm", "fine-tuning", "cuda", "mlops"],
        "experience_years": 6.0,
        "highlights": ["Led 3 production LLM pipelines at scale", "PyTorch, distributed training expert", "6 YoE at FAANG"],
        "resume_snippet": "Principal engineer on core ML platform team…",
        "resume_text": """Priya Venkataraman
Senior ML Engineer | San Francisco, CA | priya.v@techmail.com

Principal engineer on core ML platform team at leading AI company. Led 3 production LLM pipelines serving 2M+ requests/day.

Experience:
2020-2024: Senior ML Engineer, AI Corp (FAANG-adjacent)
• Scaled ML inference pipeline from 10K to 2M req/day
• Led 4-engineer team building real-time recommendation engine
• Implemented distributed training infrastructure for 70B parameter models
• PyTorch, CUDA, Kubernetes, MLflow expertise

2018-2020: ML Engineer, StartupCo
• Built initial ML infrastructure from scratch
• Deployed first production ML models

Skills: Python, PyTorch, CUDA, distributed training, fine-tuning, LLM, MLOps, Kubernetes, MLflow
Education: MS Computer Science, Stanford University""",
        "verification_status": "verified",
        "review_note": None,
        "scores": {
            "overall_score": 94,
            "skills_match": 96,
            "semantic_relevance": 93,
            "behavioral_signal": 91,
            "career_trajectory": 95,
            "why_rank": "Priya's resume demonstrates direct alignment with 9 of 11 required competencies. Her work on distributed training infrastructure at her last role matches our stack precisely. Career progression shows consistent upward trajectory with increasing scope.",
            "evidence": [
                '"Scaled ML inference pipeline from 10K to 2M req/day" — directly relevant to stated scale requirements.',
                '"Led 4-engineer team building real-time recommendation engine" — behavioral signal of leadership.',
            ],
            "debate": None,
            "borderline": False,
        }
    },
    {
        "name": "Marcus Webb",
        "email": "mwebb@devguild.io",
        "title": "Staff Software Engineer",
        "location": "Austin, TX",
        "skills": ["kubernetes", "distributed training", "python", "go", "kafka", "docker"],
        "experience_years": 7.0,
        "highlights": ["Kubernetes, distributed systems", "Open source contributor (12K GitHub stars)", "Ex-Stripe"],
        "resume_snippet": "Staff engineer on infrastructure team at Stripe…",
        "resume_text": """Marcus Webb
Staff Software Engineer | Austin, TX | mwebb@devguild.io

Staff engineer on infrastructure team at Stripe. Open source contributor with 12K GitHub stars.

Experience:
2019-2024: Staff Software Engineer, Stripe
• Designed fault-tolerant event streaming architecture handling 500K events/sec
• Mentored 8 engineers across 3 teams
• Kubernetes, distributed systems, Go, Python

2017-2019: Senior Engineer, TechCo
• Built core microservices infrastructure

Skills: Kubernetes, Python, Go, Kafka, Docker, distributed systems, open source
Education: BS Computer Science, UT Austin""",
        "verification_status": "verified",
        "review_note": None,
        "scores": {
            "overall_score": 89,
            "skills_match": 88,
            "semantic_relevance": 91,
            "behavioral_signal": 87,
            "career_trajectory": 90,
            "why_rank": "Marcus presents strong systems thinking and a track record of shipping at scale. His OSS contributions signal initiative beyond the day-job scope. Minor gap: limited direct ML experience, offset by strong systems fundamentals.",
            "evidence": [
                '"Designed fault-tolerant event streaming architecture handling 500K events/sec" — directly relevant.',
                '"Mentored 8 engineers across 3 teams" — strong behavioral signal.',
            ],
            "debate": None,
            "borderline": False,
        }
    },
    {
        "name": "Yuki Tanaka",
        "email": "yuki.t@mlresearch.jp",
        "title": "Research Scientist → Industry ML",
        "location": "Remote (Tokyo, JP)",
        "skills": ["python", "pytorch", "jax", "llm", "research", "published", "transformers"],
        "experience_years": 4.0,
        "highlights": ["NeurIPS 2023 paper author", "Strong research-to-production transition signals", "JAX, XLA expertise"],
        "resume_snippet": "Research scientist at RIKEN AI Lab…",
        "resume_text": """Yuki Tanaka
Research Scientist | Remote (Tokyo, JP) | yuki.t.research@university.jp

Research scientist at RIKEN AI Lab. First-authored paper at NeurIPS 2023 on efficient attention mechanisms.

Experience:
2022-2024: Research Scientist, RIKEN AI Lab
• First-authored paper on efficient attention mechanisms at NeurIPS 2023
• Implemented research prototype that shipped to 10M users
• JAX, XLA, PyTorch expertise

2020-2022: ML Engineer, Tech Startup
• Production ML deployment
• 18 months production experience

Skills: Python, PyTorch, JAX, XLA, LLM, transformers, research, NeurIPS
Education: PhD Computer Science, University of Tokyo""",
        "verification_status": "review",
        "review_note": "Email on resume differs from submission — consistency check flagged.",
        "scores": {
            "overall_score": 86,
            "skills_match": 82,
            "semantic_relevance": 89,
            "behavioral_signal": 83,
            "career_trajectory": 92,
            "why_rank": "Yuki's research background shows deep theoretical foundations. The trajectory from pure research toward applied ML is a strong fit signal. However, production deployment experience is thinner than top candidates.",
            "evidence": [
                '"First-authored paper on efficient attention mechanisms" — semantic relevance to LLM optimization work.',
                '"Implemented research prototype that shipped to 10M users" — bridging research/production gap.',
            ],
            "debate": {
                "pro": "The NeurIPS paper demonstrates cutting-edge LLM knowledge that's hard to hire for. Research-to-industry transitions often produce the most rigorous engineers. Trajectory suggests fast ramp.",
                "skeptic": "Only 18 months of production experience. The role needs someone who can ship on day 60, not day 180. Research habits may require adjustment to our velocity.",
            },
            "borderline": True,
        }
    },
    {
        "name": "Amara Okonkwo",
        "email": "amara@claratech.com",
        "title": "ML Platform Lead",
        "location": "London, UK",
        "skills": ["python", "mlops", "kubernetes", "gcp", "mlflow", "pytorch"],
        "experience_years": 5.5,
        "highlights": ["ML platform ownership end-to-end", "Cross-functional leadership", "GCP, Vertex AI certified"],
        "resume_snippet": "Head of ML Platform at ClaraTech (Series B)…",
        "resume_text": """Amara Okonkwo
ML Platform Lead | London, UK | amara@claratech.com

Head of ML Platform at ClaraTech (Series B). Built internal feature store used by 40 data scientists.

Experience:
2021-2024: ML Platform Lead, ClaraTech
• Built internal feature store used by 40 data scientists
• Reduced model deployment time from 3 weeks to 2 days
• GCP, Vertex AI, MLflow, Kubernetes

2019-2021: ML Engineer, BigTechCo
• ML infrastructure and deployment

Skills: Python, MLOps, GCP, Vertex AI, MLflow, Kubernetes, PyTorch, cross-functional leadership
Education: MS Machine Learning, UCL""",
        "verification_status": "verified",
        "review_note": None,
        "scores": {
            "overall_score": 85,
            "skills_match": 87,
            "semantic_relevance": 84,
            "behavioral_signal": 88,
            "career_trajectory": 82,
            "why_rank": "Amara demonstrates strong platform thinking — she's built the tooling that other ML engineers use, not just the models themselves. This multiplier effect is exactly what the role calls for.",
            "evidence": [
                '"Built internal feature store used by 40 data scientists" — platform-level impact.',
                '"Reduced model deployment time from 3 weeks to 2 days" — operational excellence signal.',
            ],
            "debate": None,
            "borderline": False,
        }
    },
    {
        "name": "Diego Herrera",
        "email": "diego.h@growthco.mx",
        "title": "Senior Data Scientist",
        "location": "Mexico City, MX",
        "skills": ["python", "sql", "scikit-learn", "pandas", "statistics"],
        "experience_years": 4.0,
        "highlights": ["Strong Python, SQL, experiment design", "Product analytics focus", "Limited LLM experience"],
        "resume_snippet": "Senior data scientist on growth team at GrowthCo…",
        "resume_text": """Diego Herrera
Senior Data Scientist | Mexico City, MX | diego.h@growthco.mx

Senior data scientist on growth team at GrowthCo. A/B testing framework, churn prediction.

Experience:
2021-2024: Senior Data Scientist, GrowthCo
• Designed A/B framework used across 8 product teams
• Built churn prediction model saving $2M ARR
• Python, SQL, experiment design

2019-2021: Data Analyst, StartupMX
• Product analytics

Skills: Python, SQL, scikit-learn, pandas, A/B testing, statistics
Education: BS Statistics, UNAM""",
        "verification_status": "verified",
        "review_note": None,
        "scores": {
            "overall_score": 78,
            "skills_match": 76,
            "semantic_relevance": 79,
            "behavioral_signal": 81,
            "career_trajectory": 76,
            "why_rank": "Diego's data science foundation is solid but the role skews toward ML engineering. His product analytics depth could be valuable in the longer term, but immediate fit is moderate.",
            "evidence": [
                '"Designed A/B framework used across 8 product teams" — analytical rigor signal.',
                '"Built churn prediction model saving $2M ARR" — business impact focus.',
            ],
            "debate": {
                "pro": "Strong analytical foundation, business impact orientation, and fast learner signals in resume. Could grow into the ML engineering aspects quickly.",
                "skeptic": "Zero production LLM experience and resume reads as product analytics, not ML engineering. The skill delta is significant for this role's requirements.",
            },
            "borderline": True,
        }
    },
    {
        "name": "Sarah Chen",
        "email": "schen@aiops.dev",
        "title": "MLOps Engineer",
        "location": "Seattle, WA",
        "skills": ["mlops", "mlflow", "kubeflow", "airflow", "kubernetes", "aws", "python"],
        "experience_years": 5.0,
        "highlights": ["MLflow, Kubeflow, Airflow expertise", "CI/CD for ML pipelines", "AWS SageMaker"],
        "resume_snippet": "MLOps engineer at AI-first infrastructure startup…",
        "resume_text": """Sarah Chen
MLOps Engineer | Seattle, WA | schen@aiops.dev

MLOps engineer at AI-first infrastructure startup. Expert in ML lifecycle tooling.

Experience:
2020-2024: MLOps Engineer, AI Infrastructure Inc
• Reduced model drift incidents by 70% through automated monitoring
• Managed 200+ model versions in production
• MLflow, Kubeflow, Airflow, AWS SageMaker

2019-2020: DevOps Engineer, TechCo
• CI/CD pipelines

Skills: MLOps, MLflow, Kubeflow, Airflow, Kubernetes, AWS, SageMaker, Python
Education: BS Computer Science, University of Washington""",
        "verification_status": "verified",
        "review_note": None,
        "scores": {
            "overall_score": 82,
            "skills_match": 85,
            "semantic_relevance": 80,
            "behavioral_signal": 82,
            "career_trajectory": 80,
            "why_rank": "Sarah brings deep operational expertise in the ML lifecycle. Her focus on deployment reliability and monitoring fills a specific gap in the stated role requirements.",
            "evidence": [
                '"Reduced model drift incidents by 70% through automated monitoring" — operational excellence.',
                '"Managed 200+ model versions in production" — scale and process maturity.',
            ],
            "debate": None,
            "borderline": False,
        }
    },
    {
        "name": "Raj Patel",
        "email": "rpatel@finml.co",
        "title": "Quantitative ML Engineer",
        "location": "New York, NY",
        "skills": ["python", "c++", "pytorch", "cuda", "statistics"],
        "experience_years": 5.0,
        "highlights": ["Financial ML, time-series forecasting", "C++ high-performance computing", "Strong math foundation"],
        "resume_snippet": "Quantitative ML engineer at Two Sigma…",
        "resume_text": """Raj Patel
Quantitative ML Engineer | New York, NY | raj.patel@twosigma.com

Quantitative ML engineer at Two Sigma. High-performance C++ and Python ML systems.

Experience:
2019-2024: Quantitative ML Engineer, Two Sigma
• Implemented low-latency feature computation in C++ at 1μs p99
• Published 2 internal papers on forecasting methods
• Python, C++, CUDA, time-series ML

2017-2019: Software Engineer, FinTechCo
• Algorithmic trading systems

Skills: Python, C++, CUDA, PyTorch, statistics, time-series, financial ML
Education: MS Financial Engineering, Cornell""",
        "verification_status": "review",
        "review_note": "Phone number on resume does not match form submission — flagged for review.",
        "scores": {
            "overall_score": 80,
            "skills_match": 79,
            "semantic_relevance": 78,
            "behavioral_signal": 84,
            "career_trajectory": 79,
            "why_rank": "Raj's quantitative background provides strong signal on numerical reasoning. The domain shift from fintech to general ML engineering is moderate — his optimization skills transfer well.",
            "evidence": [
                '"Implemented low-latency feature computation in C++ at 1μs p99" — performance engineering rigor.',
                '"Published 2 internal papers on forecasting methods" — intellectual initiative.',
            ],
            "debate": None,
            "borderline": False,
        }
    },
    {
        "name": "Elena Volkov",
        "email": "elena.v@deepnlp.eu",
        "title": "NLP Research Engineer",
        "location": "Berlin, DE",
        "skills": ["python", "pytorch", "llm", "fine-tuning", "rlhf", "nlp", "transformers"],
        "experience_years": 5.5,
        "highlights": ["Fine-tuning LLMs at scale", "RLHF implementation experience", "Multilingual NLP"],
        "resume_snippet": "NLP research engineer at EU AI lab → industry…",
        "resume_text": """Elena Volkov
NLP Research Engineer | Berlin, DE | elena.v@deepnlp.eu

NLP research engineer transitioning from EU AI lab to industry. RLHF implementation specialist.

Experience:
2022-2024: NLP Research Engineer, EU AI Lab Berlin
• Fine-tuned 70B parameter model for domain-specific tasks, achieving 18% improvement
• Implemented custom RLHF pipeline from scratch
• Multilingual NLP, PyTorch, transformers

2019-2022: ML Engineer, TechCo Germany
• Production NLP systems

Skills: Python, PyTorch, LLM, fine-tuning, RLHF, NLP, multilingual, transformers
Education: PhD Computational Linguistics, TU Berlin""",
        "verification_status": "verified",
        "review_note": None,
        "scores": {
            "overall_score": 88,
            "skills_match": 91,
            "semantic_relevance": 87,
            "behavioral_signal": 85,
            "career_trajectory": 89,
            "why_rank": "Elena's direct LLM fine-tuning and RLHF experience is rare and precisely targeted to the role. Her multilingual NLP work adds a differentiated dimension beyond the core requirements.",
            "evidence": [
                '"Fine-tuned 70B parameter model for domain-specific tasks, achieving 18% improvement" — direct LLM engineering.',
                '"Implemented custom RLHF pipeline from scratch" — rare, high-value signal.',
            ],
            "debate": None,
            "borderline": False,
        }
    },
    {
        "name": "Jordan Kim",
        "email": "jkim@productai.com",
        "title": "AI Product Engineer",
        "location": "Chicago, IL",
        "skills": ["python", "langchain", "llm", "rag", "javascript"],
        "experience_years": 3.0,
        "highlights": ["Product + engineering hybrid", "LangChain, LlamaIndex", "Startup founding team experience"],
        "resume_snippet": "Co-founder & AI engineer at early-stage startup…",
        "resume_text": """Jordan Kim
AI Product Engineer | Chicago, IL | jkim@productai.com

Co-founder and AI engineer at early-stage AI startup. Product-focused ML engineering.

Experience:
2022-2024: Co-founder & AI Engineer, ProductAI
• Shipped 3 AI-powered features used by 50K users
• Built RAG pipeline serving internal knowledge base
• LangChain, LlamaIndex, Python

2021-2022: Software Engineer, TechStartup
• Full-stack development

Skills: Python, LangChain, LlamaIndex, RAG, LLM, JavaScript, TypeScript
Education: BS Computer Science, University of Illinois""",
        "verification_status": "verified",
        "review_note": None,
        "scores": {
            "overall_score": 71,
            "skills_match": 68,
            "semantic_relevance": 73,
            "behavioral_signal": 75,
            "career_trajectory": 69,
            "why_rank": "Jordan sits at the product-engineering intersection. Strong on rapid prototyping and user-facing AI features, weaker on low-level ML infrastructure. Fits a product-facing variant of this role better than the core spec.",
            "evidence": [
                '"Shipped 3 AI-powered features used by 50K users" — product impact signal.',
                '"Built RAG pipeline serving internal knowledge base" — applied LLM experience.',
            ],
            "debate": None,
            "borderline": False,
        }
    },
    {
        "name": "Fatima Al-Rashid",
        "email": "fatima.ar@inference.ai",
        "title": "Inference Optimization Engineer",
        "location": "Dubai, UAE",
        "skills": ["python", "vllm", "tensorrt", "inference optimization", "cuda", "quantization", "c++"],
        "experience_years": 6.0,
        "highlights": ["vLLM, TensorRT-LLM expert", "GPU kernel optimization", "Cost reduction track record"],
        "resume_snippet": "Inference engineer at LLM inference startup…",
        "resume_text": """Fatima Al-Rashid
Inference Optimization Engineer | Dubai, UAE | fatima.ar@inference.ai

Inference optimization engineer at LLM inference startup. vLLM contributor, TensorRT-LLM expert.

Experience:
2021-2024: Inference Optimization Engineer, LLM Startup
• Reduced inference cost by 60% through quantization and batching
• Contributed merged PR to vLLM project
• GPU kernel optimization, TensorRT-LLM, CUDA

2018-2021: ML Engineer, Big Tech
• Production ML systems
• CUDA optimization

Skills: Python, vLLM, TensorRT, inference optimization, CUDA, quantization, C++, GPU
Education: MS Computer Engineering, AUB""",
        "verification_status": "verified",
        "review_note": None,
        "scores": {
            "overall_score": 91,
            "skills_match": 93,
            "semantic_relevance": 90,
            "behavioral_signal": 88,
            "career_trajectory": 93,
            "why_rank": "Fatima brings highly specialized inference optimization expertise that is directly addressable in the role spec. Her cost-per-token reduction achievements are quantified and significant. Strong trajectory.",
            "evidence": [
                '"Reduced inference cost by 60% through quantization and batching" — direct financial impact.',
                '"Contributed merged PR to vLLM project" — credibility in expert community.',
            ],
            "debate": None,
            "borderline": False,
        }
    },
]

FUNNEL_COUNTS = [
    {"label": "Fast Retrieval", "count": 10000, "description": "Keyword + embedding pre-filter"},
    {"label": "Enrichment", "count": 200, "description": "Profile enrichment + deduplication"},
    {"label": "Deep Reasoning", "count": 30, "description": "LLM semantic scoring + sub-scores"},
    {"label": "Ranked & Fairness-Checked", "count": 10, "description": "Final shortlist with bias audit"},
]

JD_REQUIREMENTS = {
    "hard_requirements": [
        "5+ years ML engineering experience",
        "Production LLM experience",
        "Distributed training expertise",
        "Strong Python",
        "MLOps tooling (MLflow/Kubeflow)",
    ],
    "nice_to_have": ["RLHF experience", "C++/CUDA", "Fine-tuning experience"],
    "negotiable": ["Specific MLOps tooling", "Domain specialization"],
    "implied_seniority": "senior",
    "key_skills": ["python", "llm", "pytorch", "distributed training", "mlops", "fine-tuning", "cuda"],
    "context": "Senior ML engineering role at the intersection of research and production, focused on LLM infrastructure.",
}


def seed():
    create_tables()
    db = SessionLocal()
    try:
        if db.query(Candidate).count() > 0:
            print("Database already seeded. Run with --force to re-seed.")
            return

        # Create job
        job = Job(
            title="Senior ML Engineer — LLM Infrastructure",
            description=JOB_DESCRIPTION,
            requirements=JD_REQUIREMENTS,
            funnel_counts=FUNNEL_COUNTS,
        )
        db.add(job)
        db.flush()
        job_id = job.id
        print(f"Created job: {job.title} (id={job_id})")

        # Sort by overall_score descending for rank assignment
        sorted_candidates = sorted(
            CANDIDATES,
            key=lambda c: c["scores"]["overall_score"],
            reverse=True,
        )

        for rank, cand_data in enumerate(sorted_candidates, start=1):
            scores_data = cand_data.pop("scores")

            candidate = Candidate(
                name=cand_data["name"],
                email=cand_data["email"],
                title=cand_data["title"],
                location=cand_data["location"],
                skills=cand_data["skills"],
                experience_years=cand_data["experience_years"],
                highlights=cand_data["highlights"],
                resume_snippet=cand_data["resume_snippet"],
                resume_text=cand_data["resume_text"],
                verification_status=cand_data["verification_status"],
                review_note=cand_data["review_note"],
                consistency_score=1.0 if cand_data["verification_status"] == "verified" else 0.7,
            )
            db.add(candidate)
            db.flush()

            score_row = CandidateScore(
                candidate_id=candidate.id,
                job_id=job_id,
                overall_score=scores_data["overall_score"],
                skills_match=scores_data["skills_match"],
                semantic_relevance=scores_data["semantic_relevance"],
                behavioral_signal=scores_data["behavioral_signal"],
                career_trajectory=scores_data["career_trajectory"],
                why_rank=scores_data["why_rank"],
                evidence=scores_data["evidence"],
                debate=scores_data.get("debate"),
                rank=rank,
                borderline=scores_data["borderline"],
                compute_path="deep" if scores_data["borderline"] else "fast",
                pipeline_timings={"total": 0.0, "seeded": True},
            )
            db.add(score_row)
            print(f"  [{rank:2d}] {cand_data['name']:28s} score={scores_data['overall_score']}")

        db.commit()
        print(f"\n✓ Seeded {len(CANDIDATES)} candidates with rankings for job_id={job_id}")
        print(f"✓ Ready at http://localhost:8000")
        print(f"  GET  /jobs/1/rank    → ranked shortlist")
        print(f"  GET  /jobs/1/funnel  → funnel counts")
        print(f"  GET  /candidates     → all candidates")

    except Exception as e:
        db.rollback()
        print(f"Seed failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    if "--force" in sys.argv:
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("DELETE FROM candidate_scores"))
        db.execute(__import__("sqlalchemy").text("DELETE FROM candidates"))
        db.execute(__import__("sqlalchemy").text("DELETE FROM jobs"))
        db.commit()
        db.close()
        print("Cleared existing data.")
    seed()

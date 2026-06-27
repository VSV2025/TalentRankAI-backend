"""Hackathon submission API — offline 7-layer pipeline with no LLM calls."""
import io
import json
import logging
import threading
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..services.offline_pipeline import (
    HACKATHON_JD,
    run_pipeline,
    results_to_csv,
)

# ── Embedded sample dataset (10 candidates) — used when use_sample=true ──────
SAMPLE_CANDIDATES = [
    {
        "candidate_id": "CAND001",
        "profile": {"headline": "Senior ML Engineer | LLM & Retrieval Systems", "summary": "7 years building production AI systems. Led retrieval-augmented generation and embedding pipelines shipped to 10M+ users.", "current_title": "Senior ML Engineer", "years_of_experience": 7.0, "location": "Pune, India", "country": "India"},
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 22}, {"name": "PyTorch", "proficiency": "expert", "duration_months": 60, "endorsements": 18}, {"name": "LLM", "proficiency": "advanced", "duration_months": 30, "endorsements": 14}, {"name": "RAG", "proficiency": "advanced", "duration_months": 24, "endorsements": 10}, {"name": "FAISS", "proficiency": "advanced", "duration_months": 18, "endorsements": 9}, {"name": "Elasticsearch", "proficiency": "advanced", "duration_months": 36, "endorsements": 12}, {"name": "fine-tuning", "proficiency": "advanced", "duration_months": 20, "endorsements": 8}, {"name": "Kubernetes", "proficiency": "intermediate", "duration_months": 24, "endorsements": 5}],
        "career_history": [{"title": "Senior ML Engineer", "company": "Ola AI Labs", "industry": "AI", "start_date": "2022-01-01", "end_date": None, "is_current": True, "description": "Built production RAG pipeline using FAISS and Elasticsearch serving 10M+ users with <50ms p99 latency. Fine-tuned LLaMA-2 for domain-specific ranking. Deployed embedding models via vLLM reducing inference cost by 60%."}, {"title": "ML Engineer", "company": "Haptik", "industry": "AI", "start_date": "2019-06-01", "end_date": "2021-12-31", "is_current": False, "description": "Built NLP classification and semantic search systems in production for enterprise chatbot platform."}],
        "education": [{"institution": "IIT Bombay", "degree": "B.Tech", "field_of_study": "Computer Science", "tier": "tier_1"}],
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.90, "avg_response_time_hours": 6, "profile_completeness_score": 95, "search_appearance_30d": 200, "profile_views_received_30d": 22, "applications_submitted_30d": 2, "connection_count": 520, "last_active_date": "2026-06-25", "interview_completion_rate": 0.95, "offer_acceptance_rate": 0.80, "github_activity_score": 82, "verified_email": True, "verified_phone": True, "linkedin_connected": True, "saved_by_recruiters_30d": 5, "notice_period_days": 30, "willing_to_relocate": True, "expected_salary_range_inr_lpa": {"min": 35, "max": 55}, "skill_assessment_scores": {"Python": 91, "LLM": 88, "RAG": 85}},
    },
    {
        "candidate_id": "CAND002",
        "profile": {"headline": "NLP Research Engineer | Production LLM Systems", "summary": "5 years in NLP and retrieval. Built vector search infra at scale and RLHF pipelines for instruction-tuned LLMs.", "current_title": "NLP Engineer", "years_of_experience": 5.5, "location": "Bangalore, India", "country": "India"},
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 60, "endorsements": 18}, {"name": "NLP", "proficiency": "expert", "duration_months": 54, "endorsements": 20}, {"name": "RLHF", "proficiency": "advanced", "duration_months": 18, "endorsements": 10}, {"name": "embeddings", "proficiency": "advanced", "duration_months": 36, "endorsements": 14}, {"name": "Qdrant", "proficiency": "advanced", "duration_months": 18, "endorsements": 8}, {"name": "transformers", "proficiency": "expert", "duration_months": 42, "endorsements": 16}, {"name": "MLflow", "proficiency": "intermediate", "duration_months": 24, "endorsements": 6}],
        "career_history": [{"title": "NLP Engineer", "company": "Sarvam AI", "industry": "AI", "start_date": "2023-01-01", "end_date": None, "is_current": True, "description": "Implemented RLHF pipeline for instruction-tuned Hindi-English LLM. Deployed Qdrant vector search for semantic retrieval over 50M documents. Achieved 22% MRR improvement over BM25 baseline."}, {"title": "ML Engineer", "company": "Juspay", "industry": "Fintech", "start_date": "2020-08-01", "end_date": "2022-12-31", "is_current": False, "description": "Built production fraud detection ML pipelines serving 2M transactions/day."}],
        "education": [{"institution": "NIT Trichy", "degree": "B.Tech", "field_of_study": "Computer Science", "tier": "tier_2"}],
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.85, "avg_response_time_hours": 10, "profile_completeness_score": 88, "search_appearance_30d": 160, "profile_views_received_30d": 18, "applications_submitted_30d": 3, "connection_count": 380, "last_active_date": "2026-06-24", "interview_completion_rate": 0.88, "offer_acceptance_rate": 0.75, "github_activity_score": 70, "verified_email": True, "verified_phone": True, "linkedin_connected": True, "saved_by_recruiters_30d": 4, "notice_period_days": 30, "willing_to_relocate": True, "expected_salary_range_inr_lpa": {"min": 28, "max": 45}, "skill_assessment_scores": {"NLP": 89, "Python": 87, "embeddings": 83}},
    },
    {
        "candidate_id": "CAND003",
        "profile": {"headline": "Staff ML Engineer | Search & Ranking Systems", "summary": "8 years building ranking and retrieval systems for e-commerce. Led ML platform team of 6.", "current_title": "Staff ML Engineer", "years_of_experience": 8.5, "location": "Noida, India", "country": "India"},
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 84, "endorsements": 28}, {"name": "ranking", "proficiency": "expert", "duration_months": 72, "endorsements": 24}, {"name": "retrieval", "proficiency": "expert", "duration_months": 72, "endorsements": 22}, {"name": "Elasticsearch", "proficiency": "expert", "duration_months": 60, "endorsements": 20}, {"name": "LLM", "proficiency": "intermediate", "duration_months": 18, "endorsements": 8}, {"name": "MLflow", "proficiency": "advanced", "duration_months": 36, "endorsements": 14}, {"name": "Kubernetes", "proficiency": "advanced", "duration_months": 36, "endorsements": 12}, {"name": "Spark", "proficiency": "advanced", "duration_months": 48, "endorsements": 16}],
        "career_history": [{"title": "Staff ML Engineer", "company": "Meesho", "industry": "E-Commerce", "start_date": "2021-03-01", "end_date": None, "is_current": True, "description": "Led ranking team for product search serving 200M+ users. Built hybrid BM25+embedding retrieval pipeline. Improved CTR by 18% via LambdaMART reranker. Managing 6-person ML team."}, {"title": "ML Engineer II", "company": "Flipkart", "industry": "E-Commerce", "start_date": "2018-06-01", "end_date": "2021-02-28", "is_current": False, "description": "Built production recommendation and search ranking systems. Deployed in production with 500K QPM."}],
        "education": [{"institution": "IIT Delhi", "degree": "M.Tech", "field_of_study": "Machine Learning", "tier": "tier_1"}],
        "redrob_signals": {"open_to_work_flag": False, "recruiter_response_rate": 0.70, "avg_response_time_hours": 24, "profile_completeness_score": 85, "search_appearance_30d": 180, "profile_views_received_30d": 20, "applications_submitted_30d": 1, "connection_count": 680, "last_active_date": "2026-06-20", "interview_completion_rate": 0.92, "offer_acceptance_rate": 0.85, "github_activity_score": 60, "verified_email": True, "verified_phone": True, "linkedin_connected": True, "saved_by_recruiters_30d": 6, "notice_period_days": 60, "willing_to_relocate": False, "expected_salary_range_inr_lpa": {"min": 55, "max": 80}, "skill_assessment_scores": {"ranking": 92, "retrieval": 90, "Python": 88}},
    },
    {
        "candidate_id": "CAND004",
        "profile": {"headline": "Data Scientist | ML & Analytics", "summary": "4 years in data science and analytics. Strong SQL, Python, and stats background. No production LLM experience.", "current_title": "Senior Data Scientist", "years_of_experience": 4.0, "location": "Hyderabad, India", "country": "India"},
        "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 48, "endorsements": 14}, {"name": "SQL", "proficiency": "expert", "duration_months": 48, "endorsements": 18}, {"name": "machine learning", "proficiency": "intermediate", "duration_months": 36, "endorsements": 10}, {"name": "pandas", "proficiency": "advanced", "duration_months": 48, "endorsements": 16}, {"name": "scikit-learn", "proficiency": "advanced", "duration_months": 36, "endorsements": 12}, {"name": "Tableau", "proficiency": "advanced", "duration_months": 36, "endorsements": 10}],
        "career_history": [{"title": "Senior Data Scientist", "company": "Razorpay", "industry": "Fintech", "start_date": "2022-07-01", "end_date": None, "is_current": True, "description": "A/B testing framework, churn prediction models, dashboards. No production LLM or embedding work."}, {"title": "Data Analyst", "company": "Swiggy", "industry": "E-Commerce", "start_date": "2020-05-01", "end_date": "2022-06-30", "is_current": False, "description": "Analytics and reporting. SQL-heavy role with some Python for data pipelines."}],
        "education": [{"institution": "BITS Pilani", "degree": "B.E.", "field_of_study": "Computer Science", "tier": "tier_2"}],
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.80, "avg_response_time_hours": 12, "profile_completeness_score": 78, "search_appearance_30d": 90, "profile_views_received_30d": 10, "applications_submitted_30d": 5, "connection_count": 310, "last_active_date": "2026-06-23", "interview_completion_rate": 0.80, "offer_acceptance_rate": 0.65, "github_activity_score": 30, "verified_email": True, "verified_phone": False, "linkedin_connected": True, "saved_by_recruiters_30d": 1, "notice_period_days": 45, "willing_to_relocate": True, "expected_salary_range_inr_lpa": {"min": 20, "max": 35}, "skill_assessment_scores": {"Python": 75, "SQL": 90}},
    },
    {
        "candidate_id": "CAND005",
        "profile": {"headline": "AI Engineer | LLM Applications & RAG Pipelines", "summary": "3 years building LLM-powered products from scratch. LangChain, OpenAI API, and a bit of fine-tuning.", "current_title": "AI Engineer", "years_of_experience": 3.0, "location": "Bangalore, India", "country": "India"},
        "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 36, "endorsements": 10}, {"name": "LLM", "proficiency": "intermediate", "duration_months": 24, "endorsements": 8}, {"name": "RAG", "proficiency": "intermediate", "duration_months": 18, "endorsements": 6}, {"name": "langchain", "proficiency": "intermediate", "duration_months": 18, "endorsements": 5}, {"name": "embeddings", "proficiency": "intermediate", "duration_months": 18, "endorsements": 4}, {"name": "FastAPI", "proficiency": "advanced", "duration_months": 24, "endorsements": 8}],
        "career_history": [{"title": "AI Engineer", "company": "Fractal Analytics", "industry": "AI", "start_date": "2023-04-01", "end_date": None, "is_current": True, "description": "Built RAG chatbot with LangChain for internal knowledge base. Deployed GPT-4 based workflows for enterprise clients. Limited scale — few hundred users."}, {"title": "Software Engineer", "company": "Infosys", "industry": "Consulting", "start_date": "2021-08-01", "end_date": "2023-03-31", "is_current": False, "description": "Backend web development in Python/Django. No ML work."}],
        "education": [{"institution": "VIT Vellore", "degree": "B.Tech", "field_of_study": "Information Technology", "tier": "tier_3"}],
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.90, "avg_response_time_hours": 4, "profile_completeness_score": 82, "search_appearance_30d": 110, "profile_views_received_30d": 12, "applications_submitted_30d": 8, "connection_count": 280, "last_active_date": "2026-06-26", "interview_completion_rate": 0.75, "offer_acceptance_rate": 0.60, "github_activity_score": 45, "verified_email": True, "verified_phone": True, "linkedin_connected": True, "saved_by_recruiters_30d": 2, "notice_period_days": 15, "willing_to_relocate": True, "expected_salary_range_inr_lpa": {"min": 18, "max": 30}, "skill_assessment_scores": {"Python": 78, "LLM": 62}},
    },
    {
        "candidate_id": "CAND006",
        "profile": {"headline": "ML Research Scientist | NLP & Information Retrieval", "summary": "PhD in NLP from IISc. 2 papers at ACL. Zero production deployments — all research.", "current_title": "Research Scientist", "years_of_experience": 5.0, "location": "Bangalore, India", "country": "India"},
        "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 60, "endorsements": 12}, {"name": "NLP", "proficiency": "expert", "duration_months": 60, "endorsements": 18}, {"name": "PyTorch", "proficiency": "expert", "duration_months": 48, "endorsements": 14}, {"name": "transformers", "proficiency": "expert", "duration_months": 48, "endorsements": 16}, {"name": "retrieval", "proficiency": "advanced", "duration_months": 36, "endorsements": 10}],
        "career_history": [{"title": "Research Scientist", "company": "IISc RBCDSAI", "industry": "Research", "start_date": "2021-01-01", "end_date": None, "is_current": True, "description": "Published NLP research on dense retrieval and re-ranking. ACL 2023 paper on efficient passage retrieval. No production deployments — academic research environment."}, {"title": "Research Intern", "company": "Microsoft Research India", "industry": "Research", "start_date": "2019-05-01", "end_date": "2020-12-31", "is_current": False, "description": "Research on question answering systems. No production work."}],
        "education": [{"institution": "IISc Bangalore", "degree": "PhD", "field_of_study": "Computational Linguistics", "tier": "tier_1"}],
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.60, "avg_response_time_hours": 48, "profile_completeness_score": 75, "search_appearance_30d": 70, "profile_views_received_30d": 8, "applications_submitted_30d": 2, "connection_count": 220, "last_active_date": "2026-06-10", "interview_completion_rate": 0.70, "offer_acceptance_rate": 0.50, "github_activity_score": 55, "verified_email": True, "verified_phone": False, "linkedin_connected": False, "saved_by_recruiters_30d": 1, "notice_period_days": 90, "willing_to_relocate": False, "expected_salary_range_inr_lpa": {"min": 30, "max": 50}, "skill_assessment_scores": {"NLP": 92, "PyTorch": 85, "retrieval": 80}},
    },
    {
        "candidate_id": "CAND007",
        "profile": {"headline": "Backend Engineer | Python & Distributed Systems", "summary": "6 years building scalable backend systems. No ML/AI experience. Interested in ML roles.", "current_title": "Senior Backend Engineer", "years_of_experience": 6.0, "location": "Pune, India", "country": "India"},
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 24}, {"name": "Go", "proficiency": "advanced", "duration_months": 36, "endorsements": 16}, {"name": "Kubernetes", "proficiency": "expert", "duration_months": 48, "endorsements": 20}, {"name": "Kafka", "proficiency": "advanced", "duration_months": 36, "endorsements": 14}, {"name": "PostgreSQL", "proficiency": "expert", "duration_months": 60, "endorsements": 18}, {"name": "Redis", "proficiency": "advanced", "duration_months": 48, "endorsements": 14}],
        "career_history": [{"title": "Senior Backend Engineer", "company": "Zerodha", "industry": "Fintech", "start_date": "2021-01-01", "end_date": None, "is_current": True, "description": "Scaled trading platform to 10M daily users. No ML work. Pure distributed systems and backend infrastructure."}, {"title": "Backend Engineer", "company": "ClearTax", "industry": "Fintech", "start_date": "2018-07-01", "end_date": "2020-12-31", "is_current": False, "description": "API development and microservices. No ML."}],
        "education": [{"institution": "Pune University", "degree": "B.E.", "field_of_study": "Computer Engineering", "tier": "tier_3"}],
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.75, "avg_response_time_hours": 18, "profile_completeness_score": 80, "search_appearance_30d": 80, "profile_views_received_30d": 9, "applications_submitted_30d": 4, "connection_count": 350, "last_active_date": "2026-06-22", "interview_completion_rate": 0.85, "offer_acceptance_rate": 0.70, "github_activity_score": 50, "verified_email": True, "verified_phone": True, "linkedin_connected": True, "saved_by_recruiters_30d": 2, "notice_period_days": 30, "willing_to_relocate": True, "expected_salary_range_inr_lpa": {"min": 30, "max": 50}, "skill_assessment_scores": {"Python": 88}},
    },
    {
        "candidate_id": "CAND008",
        "profile": {"headline": "Founding ML Engineer | Startup Builder", "summary": "Co-founded AI startup (Series A). Built production LLM pipelines, vector search, and fine-tuned models from scratch. 6 years total.", "current_title": "Founding ML Engineer", "years_of_experience": 6.0, "location": "Gurgaon, India", "country": "India"},
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 20}, {"name": "LLM", "proficiency": "expert", "duration_months": 30, "endorsements": 16}, {"name": "fine-tuning", "proficiency": "advanced", "duration_months": 24, "endorsements": 12}, {"name": "vector search", "proficiency": "expert", "duration_months": 24, "endorsements": 14}, {"name": "Pinecone", "proficiency": "advanced", "duration_months": 18, "endorsements": 10}, {"name": "PyTorch", "proficiency": "advanced", "duration_months": 48, "endorsements": 14}, {"name": "Kubernetes", "proficiency": "intermediate", "duration_months": 24, "endorsements": 8}, {"name": "lora", "proficiency": "advanced", "duration_months": 18, "endorsements": 8}],
        "career_history": [{"title": "Founding ML Engineer", "company": "VoiceAI (Series A)", "industry": "AI", "start_date": "2022-06-01", "end_date": None, "is_current": True, "description": "Built the entire AI stack from scratch. Fine-tuned Whisper + LLaMA for domain-specific voice AI. Production system serving 500K users. Vector search over 20M documents with Pinecone. LLM inference optimization cut cost by 45%."}, {"title": "ML Engineer", "company": "Amazon", "industry": "E-Commerce", "start_date": "2019-07-01", "end_date": "2022-05-31", "is_current": False, "description": "Search relevance ranking and recommendation systems at scale. Shipped ML models to 100M+ users."}],
        "education": [{"institution": "IIT Kharagpur", "degree": "B.Tech", "field_of_study": "Computer Science", "tier": "tier_1"}],
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.80, "avg_response_time_hours": 12, "profile_completeness_score": 90, "search_appearance_30d": 140, "profile_views_received_30d": 16, "applications_submitted_30d": 2, "connection_count": 420, "last_active_date": "2026-06-25", "interview_completion_rate": 0.90, "offer_acceptance_rate": 0.75, "github_activity_score": 78, "verified_email": True, "verified_phone": True, "linkedin_connected": True, "saved_by_recruiters_30d": 4, "notice_period_days": 30, "willing_to_relocate": False, "expected_salary_range_inr_lpa": {"min": 50, "max": 75}, "skill_assessment_scores": {"LLM": 88, "fine-tuning": 84, "Python": 90}},
    },
    {
        "candidate_id": "CAND009",
        "profile": {"headline": "Frontend Engineer | React & TypeScript", "summary": "5 years building UI and web applications. No backend or ML experience.", "current_title": "Senior Frontend Engineer", "years_of_experience": 5.0, "location": "Chennai, India", "country": "India"},
        "skills": [{"name": "React", "proficiency": "expert", "duration_months": 60, "endorsements": 22}, {"name": "TypeScript", "proficiency": "expert", "duration_months": 48, "endorsements": 18}, {"name": "JavaScript", "proficiency": "expert", "duration_months": 60, "endorsements": 24}, {"name": "CSS", "proficiency": "expert", "duration_months": 60, "endorsements": 20}, {"name": "Node.js", "proficiency": "advanced", "duration_months": 36, "endorsements": 14}],
        "career_history": [{"title": "Senior Frontend Engineer", "company": "Freshworks", "industry": "SaaS", "start_date": "2021-03-01", "end_date": None, "is_current": True, "description": "Built React component libraries and dashboards. No backend or ML work whatsoever."}, {"title": "Frontend Developer", "company": "Zoho", "industry": "SaaS", "start_date": "2019-07-01", "end_date": "2021-02-28", "is_current": False, "description": "UI development in JavaScript and React."}],
        "education": [{"institution": "Anna University", "degree": "B.E.", "field_of_study": "Computer Science", "tier": "tier_3"}],
        "redrob_signals": {"open_to_work_flag": True, "recruiter_response_rate": 0.85, "avg_response_time_hours": 8, "profile_completeness_score": 85, "search_appearance_30d": 40, "profile_views_received_30d": 5, "applications_submitted_30d": 6, "connection_count": 290, "last_active_date": "2026-06-24", "interview_completion_rate": 0.80, "offer_acceptance_rate": 0.70, "github_activity_score": 35, "verified_email": True, "verified_phone": True, "linkedin_connected": True, "saved_by_recruiters_30d": 0, "notice_period_days": 30, "willing_to_relocate": False, "expected_salary_range_inr_lpa": {"min": 25, "max": 40}, "skill_assessment_scores": {"React": 90, "TypeScript": 88}},
    },
    {
        "candidate_id": "CAND010",
        "profile": {"headline": "Applied ML Scientist | Recommendation & Personalization", "summary": "6 years building recommendation and ranking systems in production. Transitioning to LLM-heavy roles.", "current_title": "Applied ML Scientist", "years_of_experience": 6.5, "location": "Mumbai, India", "country": "India"},
        "skills": [{"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 22}, {"name": "machine learning", "proficiency": "expert", "duration_months": 66, "endorsements": 20}, {"name": "recommendation", "proficiency": "expert", "duration_months": 60, "endorsements": 18}, {"name": "ranking", "proficiency": "advanced", "duration_months": 48, "endorsements": 16}, {"name": "PyTorch", "proficiency": "advanced", "duration_months": 36, "endorsements": 12}, {"name": "embeddings", "proficiency": "intermediate", "duration_months": 18, "endorsements": 8}, {"name": "LLM", "proficiency": "beginner", "duration_months": 8, "endorsements": 2}, {"name": "Spark", "proficiency": "advanced", "duration_months": 48, "endorsements": 14}],
        "career_history": [{"title": "Applied ML Scientist", "company": "Netflix India", "industry": "Tech", "start_date": "2021-09-01", "end_date": None, "is_current": True, "description": "Built recommendation ranking models used by 10M+ India users. A/B tested 20+ model variants in production. Started exploring LLM-based personalization recently."}, {"title": "ML Engineer", "company": "Hotstar", "industry": "Media", "start_date": "2019-07-01", "end_date": "2021-08-31", "is_current": False, "description": "Video recommendation and content ranking systems. Deployed gradient boosting and neural ranking models."}],
        "education": [{"institution": "IIIT Hyderabad", "degree": "M.Tech", "field_of_study": "Machine Learning", "tier": "tier_2"}],
        "redrob_signals": {"open_to_work_flag": False, "recruiter_response_rate": 0.65, "avg_response_time_hours": 30, "profile_completeness_score": 83, "search_appearance_30d": 130, "profile_views_received_30d": 15, "applications_submitted_30d": 1, "connection_count": 460, "last_active_date": "2026-06-18", "interview_completion_rate": 0.88, "offer_acceptance_rate": 0.72, "github_activity_score": 58, "verified_email": True, "verified_phone": True, "linkedin_connected": True, "saved_by_recruiters_30d": 3, "notice_period_days": 60, "willing_to_relocate": False, "expected_salary_range_inr_lpa": {"min": 45, "max": 70}, "skill_assessment_scores": {"Python": 88, "machine learning": 85, "ranking": 82}},
    },
]

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/hackathon", tags=["hackathon"])

# In-memory job store  { job_id -> job_dict }
_jobs: dict = {}


def _make_job() -> dict:
    return {
        "status": "pending",
        "progress": {
            "current_layer": "L1 JD Parse",
            "layer_index": 0,
            "processed": 0,
            "total": 0,
            "message": "Starting pipeline…",
        },
        "results": None,
        "error": None,
    }


def _run_job(job_id: str, candidates: list) -> None:
    job = _jobs[job_id]
    job["status"] = "running"

    def progress_cb(p: dict):
        job["progress"] = p

    try:
        output = run_pipeline(candidates, top_k=min(2000, len(candidates)), progress_cb=progress_cb)
        job["status"] = "complete"
        job["results"] = output
    except Exception as exc:
        logger.exception("Pipeline job %s failed", job_id)
        job["status"] = "error"
        job["error"] = str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jd")
def get_jd():
    """Return the pre-parsed hackathon job description."""
    return HACKATHON_JD


@router.post("/rank")
async def start_ranking(
    file: Optional[UploadFile] = File(default=None),
    use_sample: bool = Form(default=False),
    sample_path: Optional[str] = Form(default=None),
):
    """
    Start the 7-layer offline ranking pipeline.

    Accepts one of:
    - file upload (.jsonl or .json array of candidate objects)
    - use_sample=true  → uses the 50-candidate sample from the challenge bundle
    - sample_path      → absolute path to a local candidates.jsonl on the server
    """
    candidates: list = []

    if file is not None:
        raw = await file.read()
        text = raw.decode("utf-8")
        # Support both JSONL and JSON array
        if text.strip().startswith("["):
            candidates = json.loads(text)
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

    elif sample_path:
        import os
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=400, detail=f"File not found: {sample_path}")
        with open(sample_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                candidates = json.load(f)
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        candidates.append(json.loads(line))

    elif use_sample:
        import os
        _CHALLENGE_PATHS = [
            r"C:\Users\drgsr\Downloads\[PUB] India_runs_data_and_ai_challenge"
            r"\[PUB] India_runs_data_and_ai_challenge"
            r"\India_runs_data_and_ai_challenge\sample_candidates.json",
            r"C:\Users\drgsr\Downloads\[PUB] India_runs_data_and_ai_challenge"
            r"\[PUB] India_runs_data_and_ai_challenge"
            r"\India_runs_data_and_ai_challenge\candidates.jsonl",
        ]
        loaded = False
        for p in _CHALLENGE_PATHS:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    first_char = f.read(1)
                    f.seek(0)
                    if first_char == "[":
                        candidates = json.load(f)
                    else:
                        for line in f:
                            line = line.strip()
                            if line:
                                candidates.append(json.loads(line))
                logger.info(f"Loaded {len(candidates)} candidates from {p}")
                loaded = True
                break
        if not loaded:
            candidates = SAMPLE_CANDIDATES
            logger.info("Challenge file not found — using embedded 10-candidate demo")
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide a file upload, use_sample=true, or sample_path.",
        )

    if not candidates:
        raise HTTPException(status_code=400, detail="No valid candidates found in input.")

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = _make_job()
    _jobs[job_id]["progress"]["total"] = len(candidates)

    t = threading.Thread(target=_run_job, args=(job_id, candidates), daemon=True)
    t.start()

    return {"job_id": job_id, "total_candidates": len(candidates), "status": "running"}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    """Poll pipeline status and progress."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "has_results": job["results"] is not None,
        "error": job["error"],
    }


@router.get("/results/{job_id}")
def get_results(job_id: str):
    """Return full results (top-100 ranked candidates + funnel counts)."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job["status"] == "running":
        raise HTTPException(status_code=202, detail="Pipeline still running.")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"])
    return job["results"]


@router.get("/download/{job_id}")
def download_csv(job_id: str):
    """Download the submission.csv for a completed job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job not complete yet.")

    csv_text = results_to_csv(job["results"]["results"])
    return StreamingResponse(
        io.BytesIO(csv_text.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=submission.csv"},
    )

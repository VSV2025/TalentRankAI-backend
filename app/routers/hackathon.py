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

def _rs(otw=True, rr=0.75, rt=12, pc=80, sa=100, pv=10, apps=3, cc=300,
        la="2026-06-20", icr=0.80, oar=0.70, gh=50, ve=True, vp=True,
        li=True, sbr=2, npd=30, wr=True, smin=20, smax=40, sc=None):
    return {
        "open_to_work_flag": otw, "recruiter_response_rate": rr,
        "avg_response_time_hours": rt, "profile_completeness_score": pc,
        "search_appearance_30d": sa, "profile_views_received_30d": pv,
        "applications_submitted_30d": apps, "connection_count": cc,
        "last_active_date": la, "interview_completion_rate": icr,
        "offer_acceptance_rate": oar, "github_activity_score": gh,
        "verified_email": ve, "verified_phone": vp, "linkedin_connected": li,
        "saved_by_recruiters_30d": sbr, "notice_period_days": npd,
        "willing_to_relocate": wr,
        "expected_salary_range_inr_lpa": {"min": smin, "max": smax},
        "skill_assessment_scores": sc or {},
    }

def _sk(name, prof, months, end):
    return {"name": name, "proficiency": prof, "duration_months": months, "endorsements": end}

def _ch(title, co, ind, start, end, curr, desc):
    return {"title": title, "company": co, "industry": ind,
            "start_date": start, "end_date": end, "is_current": curr, "description": desc}

def _ed(inst, deg, field, tier):
    return {"institution": inst, "degree": deg, "field_of_study": field, "tier": tier}

# ── Embedded sample dataset (50 candidates) — used when use_sample=true ───────
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
    # ── CAND011-020: Strong ML candidates ──────────────────────────────────────
    {"candidate_id":"CAND011","profile":{"headline":"ML Platform Engineer | MLOps & Infra","summary":"5 yrs building ML infra, model serving, feature stores and CI/CD for ML teams.","current_title":"ML Platform Engineer","years_of_experience":5.0,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("MLflow","expert",48,16),_sk("Kubernetes","expert",48,18),_sk("Airflow","advanced",36,12),_sk("feature store","advanced",30,10),_sk("PyTorch","intermediate",24,8),_sk("LLM","intermediate",18,6)],"career_history":[_ch("ML Platform Engineer","Swiggy","E-Commerce","2022-01-01",None,True,"Built feature store serving 500+ ML models. Reduced model deployment time from 2 weeks to 4 hours. Designed GPU-autoscaling infra for inference."),_ch("Data Engineer","PayTM","Fintech","2019-06-01","2021-12-31",False,"Built Spark pipelines for real-time fraud signals.")],"education":[_ed("IIIT Hyderabad","B.Tech","Computer Science","tier_2")],"redrob_signals":_rs(rr=0.82,gh=65,sa=120,pv=14,sbr=3,npd=30,smin=28,smax=45,sc={"Python":86,"MLflow":88,"Kubernetes":85})},
    {"candidate_id":"CAND012","profile":{"headline":"GenAI Engineer | LLM Fine-tuning & Serving","summary":"4 yrs. Specialised in fine-tuning open-source LLMs and deploying them at low latency.","current_title":"GenAI Engineer","years_of_experience":4.0,"location":"Hyderabad, India","country":"India"},"skills":[_sk("Python","expert",48,16),_sk("LLM","advanced",30,14),_sk("fine-tuning","advanced",24,12),_sk("vLLM","advanced",18,10),_sk("LoRA","advanced",18,10),_sk("Hugging Face","expert",36,16),_sk("RLHF","intermediate",12,6)],"career_history":[_ch("GenAI Engineer","PhonePe","Fintech","2022-08-01",None,True,"Fine-tuned LLaMA-3 for financial document QA. Built vLLM serving layer handling 10K RPM. Reduced inference cost 52% via quantisation + batching."),_ch("ML Engineer","Mu Sigma","Consulting","2020-06-01","2022-07-31",False,"Tabular ML models for client analytics projects.")],"education":[_ed("BITS Pilani","B.E.","Computer Science","tier_2")],"redrob_signals":_rs(rr=0.88,gh=72,sa=140,pv=16,sbr=4,npd=15,smin=24,smax=40,sc={"LLM":86,"fine-tuning":84,"Python":88})},
    {"candidate_id":"CAND013","profile":{"headline":"Computer Vision & ML Engineer","summary":"6 yrs in CV and multimodal AI. YOLO, diffusion models, OCR pipelines in production.","current_title":"Senior ML Engineer","years_of_experience":6.0,"location":"Noida, India","country":"India"},"skills":[_sk("Python","expert",72,20),_sk("computer vision","expert",60,22),_sk("PyTorch","expert",60,18),_sk("YOLO","advanced",36,14),_sk("diffusion models","advanced",18,10),_sk("OpenCV","expert",60,20),_sk("LLM","beginner",6,2)],"career_history":[_ch("Senior ML Engineer","Nykaa","E-Commerce","2021-04-01",None,True,"Built product image recognition pipeline processing 5M images/day. Deployed diffusion model for virtual try-on. OCR pipeline for invoice extraction."),_ch("ML Engineer","TCS Research","Consulting","2018-07-01","2021-03-31",False,"CV projects for manufacturing defect detection.")],"education":[_ed("DTU Delhi","B.Tech","Electronics","tier_2")],"redrob_signals":_rs(rr=0.78,gh=60,sa=110,pv=12,sbr=2,npd=45,smin=30,smax=50,sc={"Python":87,"computer vision":90,"PyTorch":85})},
    {"candidate_id":"CAND014","profile":{"headline":"ML Engineer | Time-Series & Forecasting","summary":"5 yrs. Expert in demand forecasting and anomaly detection at scale.","current_title":"ML Engineer","years_of_experience":5.0,"location":"Pune, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("time-series","expert",54,20),_sk("forecasting","expert",48,18),_sk("Prophet","advanced",36,12),_sk("PyTorch","intermediate",24,8),_sk("Spark","advanced",36,14),_sk("LLM","beginner",6,2)],"career_history":[_ch("ML Engineer","BigBasket","E-Commerce","2021-07-01",None,True,"Demand forecasting models saving ₹4Cr/month in waste. Anomaly detection for supply chain serving 50 cities."),_ch("Data Scientist","Mindtree","Consulting","2019-06-01","2021-06-30",False,"Statistical forecasting models for retail clients.")],"education":[_ed("Symbiosis Institute","MBA","Analytics","tier_3")],"redrob_signals":_rs(rr=0.80,gh=42,sa=95,pv=11,sbr=2,npd=30,smin=22,smax=38,sc={"Python":85,"time-series":88,"forecasting":86})},
    {"candidate_id":"CAND015","profile":{"headline":"ML Engineer | Recommender Systems","summary":"5 yrs building production recommendation engines. Matrix factorisation, two-tower, GNNs.","current_title":"ML Engineer II","years_of_experience":5.5,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("recommendation","expert",54,20),_sk("two-tower model","advanced",30,12),_sk("GNN","intermediate",18,8),_sk("embeddings","advanced",36,14),_sk("PyTorch","advanced",42,14),_sk("LLM","beginner",8,2)],"career_history":[_ch("ML Engineer II","Zomato","E-Commerce","2021-01-01",None,True,"Two-tower recommendation model serving 80M users. GNN-based collaborative filtering improved CTR 14%."),_ch("Software Engineer","ThoughtWorks","Consulting","2018-08-01","2020-12-31",False,"Backend development. No ML work.")],"education":[_ed("Manipal University","B.Tech","Computer Science","tier_3")],"redrob_signals":_rs(rr=0.85,gh=55,sa=105,pv=12,sbr=2,npd=30,smin=26,smax=42,sc={"Python":86,"recommendation":88,"embeddings":82})},
    {"candidate_id":"CAND016","profile":{"headline":"Data Scientist | NLP & Text Analytics","summary":"4 yrs NLP work: text classification, sentiment, topic modelling. No LLM production deployments yet.","current_title":"Data Scientist","years_of_experience":4.0,"location":"Chennai, India","country":"India"},"skills":[_sk("Python","advanced",48,14),_sk("NLP","advanced",42,16),_sk("scikit-learn","advanced",42,14),_sk("BERT","intermediate",24,8),_sk("SQL","advanced",48,16),_sk("Tableau","intermediate",36,10)],"career_history":[_ch("Data Scientist","HDFC Life","Insurance","2022-04-01",None,True,"Customer churn NLP model. Sentiment analysis for 5M support tickets. Topic modelling for policy documents."),_ch("Data Analyst","Wipro","Consulting","2020-06-01","2022-03-31",False,"BI dashboards and ad-hoc analysis. Limited ML.")],"education":[_ed("Anna University","B.E.","Computer Science","tier_3")],"redrob_signals":_rs(rr=0.82,gh=28,sa=75,pv=8,sbr=1,npd=30,smin=16,smax=28,sc={"Python":80,"NLP":76,"SQL":84})},
    {"candidate_id":"CAND017","profile":{"headline":"Applied Scientist | Search Relevance","summary":"7 yrs search engineering. BM25, learning-to-rank, hybrid retrieval. Joining LLM space now.","current_title":"Applied Scientist","years_of_experience":7.0,"location":"Gurgaon, India","country":"India"},"skills":[_sk("Python","expert",72,20),_sk("search","expert",72,24),_sk("Elasticsearch","expert",60,22),_sk("ranking","expert",60,20),_sk("LTR","expert",48,18),_sk("Solr","advanced",36,14),_sk("LLM","intermediate",14,6)],"career_history":[_ch("Applied Scientist","MakeMyTrip","Travel","2020-07-01",None,True,"Search relevance ranking for 10M+ monthly users. Built LTR models improving NDCG@10 by 28%. Hybrid BM25+vector search."),_ch("Search Engineer","Info Edge","Tech","2017-05-01","2020-06-30",False,"Search relevance and NLP for Naukri job search.")],"education":[_ed("IIT Roorkee","B.Tech","Computer Science","tier_1")],"redrob_signals":_rs(rr=0.72,gh=62,sa=125,pv=14,sbr=3,npd=60,smin=38,smax=58,sc={"search":90,"ranking":88,"Python":86})},
    {"candidate_id":"CAND018","profile":{"headline":"MLOps Engineer | Model Serving & Monitoring","summary":"4 yrs. Specialist in model deployment, drift detection, A/B infra. Strong DevOps background.","current_title":"MLOps Engineer","years_of_experience":4.5,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","advanced",48,14),_sk("MLflow","advanced",36,14),_sk("Docker","expert",48,18),_sk("Kubernetes","advanced",36,14),_sk("model monitoring","advanced",30,12),_sk("FastAPI","advanced",30,10),_sk("Terraform","intermediate",24,8)],"career_history":[_ch("MLOps Engineer","Razorpay","Fintech","2022-02-01",None,True,"Built model serving platform used by 30+ ML engineers. Drift detection and auto-retraining for fraud models. Reduced P99 latency 35%."),_ch("DevOps Engineer","Wipro","Consulting","2020-01-01","2022-01-31",False,"CI/CD pipelines and cloud infra. No ML.")],"education":[_ed("Manipal University","B.Tech","Information Technology","tier_3")],"redrob_signals":_rs(rr=0.85,gh=58,sa=100,pv=11,sbr=2,npd=30,smin=20,smax=35,sc={"Python":82,"Kubernetes":84,"MLflow":85})},
    {"candidate_id":"CAND019","profile":{"headline":"AI Research Engineer | Multimodal LLMs","summary":"3 yrs research + 2 yrs production. Built vision-language models and deployed them at startup scale.","current_title":"AI Research Engineer","years_of_experience":5.0,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("LLM","advanced",30,14),_sk("multimodal","advanced",24,12),_sk("fine-tuning","advanced",24,12),_sk("PyTorch","expert",48,16),_sk("CLIP","advanced",20,10),_sk("vLLM","intermediate",12,6)],"career_history":[_ch("AI Research Engineer","Intello (YC S23)","AI","2023-09-01",None,True,"Built document understanding LLM combining OCR + vision-language model. Production for 50 enterprise clients."),_ch("ML Researcher","IISc","Research","2020-08-01","2023-08-31",False,"Research on vision-language grounding. 1 EMNLP paper.")],"education":[_ed("IISc Bangalore","M.Tech","AI","tier_1")],"redrob_signals":_rs(rr=0.90,gh=75,sa=130,pv=15,sbr=4,npd=30,smin=25,smax=42,sc={"LLM":84,"PyTorch":88,"fine-tuning":82})},
    {"candidate_id":"CAND020","profile":{"headline":"ML Engineer | Fraud Detection & Risk","summary":"6 yrs fintech ML. Expert in real-time fraud detection. Python, XGBoost, streaming ML.","current_title":"Senior ML Engineer","years_of_experience":6.0,"location":"Mumbai, India","country":"India"},"skills":[_sk("Python","expert",72,20),_sk("XGBoost","expert",60,20),_sk("fraud detection","expert",60,22),_sk("Kafka","advanced",36,14),_sk("feature engineering","expert",60,20),_sk("scikit-learn","expert",60,18),_sk("LLM","beginner",6,2)],"career_history":[_ch("Senior ML Engineer","CRED","Fintech","2021-06-01",None,True,"Real-time fraud detection pipeline processing 2M transactions/day. Feature store with 200+ features. Model F1 improved from 0.82 to 0.94."),_ch("ML Engineer","Bajaj Finserv","Fintech","2018-07-01","2021-05-31",False,"Credit scoring and collections risk models.")],"education":[_ed("NIT Surathkal","B.Tech","Computer Science","tier_2")],"redrob_signals":_rs(rr=0.78,gh=50,sa=115,pv=13,sbr=3,npd=45,smin=32,smax=52,sc={"Python":88,"XGBoost":90,"fraud detection":88})},
    # ── CAND021-030: Mid-tier / mixed backgrounds ───────────────────────────────
    {"candidate_id":"CAND021","profile":{"headline":"Software Engineer | Python & ML Enthusiast","summary":"4 yrs software engineering. Learning ML on the side. Some Kaggle experience.","current_title":"Software Engineer","years_of_experience":4.0,"location":"Pune, India","country":"India"},"skills":[_sk("Python","advanced",48,14),_sk("Django","advanced",48,16),_sk("SQL","advanced",48,16),_sk("machine learning","beginner",12,4),_sk("scikit-learn","beginner",10,3)],"career_history":[_ch("Software Engineer","Persistent Systems","Tech","2020-07-01",None,True,"Backend API development. Recently completed an online ML course."),_ch("Junior Developer","TCS","Consulting","2018-07-01","2020-06-30",False,"Java web applications.")],"education":[_ed("Pune University","B.E.","Computer Engineering","tier_3")],"redrob_signals":_rs(rr=0.88,gh=35,sa=60,pv=6,sbr=1,npd=30,smin=12,smax=22,sc={"Python":80,"SQL":78})},
    {"candidate_id":"CAND022","profile":{"headline":"Data Engineer | ETL & Data Pipelines","summary":"5 yrs data engineering. Spark, Kafka, Airflow. No ML modelling experience.","current_title":"Senior Data Engineer","years_of_experience":5.0,"location":"Hyderabad, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("Spark","expert",54,20),_sk("Kafka","advanced",36,14),_sk("Airflow","expert",48,18),_sk("SQL","expert",60,22),_sk("dbt","advanced",24,10)],"career_history":[_ch("Senior Data Engineer","Ola Electric","Manufacturing","2021-08-01",None,True,"Built real-time data pipelines for 500K IoT devices. Kafka + Spark Streaming. No ML modelling."),_ch("Data Engineer","Mu Sigma","Consulting","2019-01-01","2021-07-31",False,"ETL pipelines and data warehouse work.")],"education":[_ed("JNTU Hyderabad","B.Tech","Computer Science","tier_3")],"redrob_signals":_rs(rr=0.80,gh=42,sa=85,pv=9,sbr=1,npd=30,smin=22,smax=38,sc={"Python":85,"Spark":88,"SQL":90})},
    {"candidate_id":"CAND023","profile":{"headline":"ML Engineer | Healthcare AI","summary":"4 yrs applying ML in healthcare domain. Medical image analysis and clinical NLP.","current_title":"ML Engineer","years_of_experience":4.5,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","advanced",48,14),_sk("medical imaging","advanced",42,16),_sk("PyTorch","advanced",36,12),_sk("NLP","intermediate",24,8),_sk("scikit-learn","advanced",36,12),_sk("DICOM","advanced",36,12)],"career_history":[_ch("ML Engineer","Niramai Health","Healthcare","2021-06-01",None,True,"ML models for breast cancer screening from thermal images. FDA submission support. 92% sensitivity in clinical trials."),_ch("Junior Data Scientist","Wipro GE Healthcare","Healthcare","2019-08-01","2021-05-31",False,"Medical device data analytics and reporting.")],"education":[_ed("Manipal University","B.Tech","Biomedical Engineering","tier_3")],"redrob_signals":_rs(rr=0.82,gh=38,sa=70,pv=8,sbr=1,npd=60,wr=False,smin=18,smax=30,sc={"Python":80,"PyTorch":78})},
    {"candidate_id":"CAND024","profile":{"headline":"Quant Researcher | ML for Finance","summary":"5 yrs in quantitative finance. Strong stats, Python, and ML for trading signals. No software engineering background.","current_title":"Quantitative Researcher","years_of_experience":5.0,"location":"Mumbai, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("statistics","expert",60,22),_sk("machine learning","advanced",36,14),_sk("pandas","expert",60,20),_sk("numpy","expert",60,18),_sk("time-series","advanced",48,16)],"career_history":[_ch("Quantitative Researcher","Edelweiss","Finance","2021-01-01",None,True,"ML-based alpha signals for equities. Backtesting framework. No production software engineering."),_ch("Analyst","ICICI Securities","Finance","2019-06-01","2020-12-31",False,"Quantitative analysis and financial modelling.")],"education":[_ed("IIM Ahmedabad","MBA","Finance","tier_1")],"redrob_signals":_rs(rr=0.65,gh=20,sa=60,pv=7,sbr=1,npd=90,wr=False,smin=40,smax=70,sc={"Python":82,"statistics":88})},
    {"candidate_id":"CAND025","profile":{"headline":"DevOps + ML Aspirant | Cloud & Containers","summary":"4 yrs DevOps. Wants to move into MLOps. Has done 2 ML courses but no production ML.","current_title":"DevOps Engineer","years_of_experience":4.0,"location":"Noida, India","country":"India"},"skills":[_sk("Kubernetes","expert",48,18),_sk("Docker","expert",48,20),_sk("Terraform","advanced",36,14),_sk("Python","intermediate",24,8),_sk("AWS","advanced",36,16),_sk("CI/CD","expert",48,18)],"career_history":[_ch("DevOps Engineer","Info Edge","Tech","2020-06-01",None,True,"Kubernetes cluster management for 200+ microservices. No ML workloads yet."),_ch("System Admin","HCL","Consulting","2018-07-01","2020-05-31",False,"Linux server administration and monitoring.")],"education":[_ed("AKTU Lucknow","B.Tech","Computer Science","tier_3")],"redrob_signals":_rs(rr=0.85,gh=30,sa=55,pv=6,sbr=1,npd=30,smin=14,smax=25,sc={"Kubernetes":88,"Docker":86})},
    {"candidate_id":"CAND026","profile":{"headline":"Product Manager | AI Products","summary":"6 yrs PM. Has shipped AI features. Non-technical but deep understanding of ML pipelines.","current_title":"Senior Product Manager","years_of_experience":6.0,"location":"Bangalore, India","country":"India"},"skills":[_sk("product management","expert",72,24),_sk("SQL","intermediate",36,10),_sk("Python","beginner",6,2),_sk("AI products","advanced",30,12)],"career_history":[_ch("Senior PM","Lenskart","E-Commerce","2021-05-01",None,True,"Led AI-powered virtual try-on product. Worked closely with ML team. 0 code written personally."),_ch("PM","InMobi","AdTech","2018-08-01","2021-04-30",False,"Mobile ad targeting products. Non-technical PM.")],"education":[_ed("IIM Bangalore","MBA","Marketing","tier_1")],"redrob_signals":_rs(rr=0.70,gh=5,sa=40,pv=5,sbr=0,npd=60,wr=False,smin=40,smax=65,sc={})},
    {"candidate_id":"CAND027","profile":{"headline":"ML Engineer | Robotics & Edge AI","summary":"4 yrs. TensorFlow Lite, ONNX, embedded ML. Robotics perception stack.","current_title":"ML Engineer","years_of_experience":4.0,"location":"Pune, India","country":"India"},"skills":[_sk("Python","advanced",48,14),_sk("TensorFlow","advanced",42,14),_sk("ONNX","advanced",36,12),_sk("C++","intermediate",30,8),_sk("edge AI","advanced",30,12),_sk("ROS","intermediate",24,8)],"career_history":[_ch("ML Engineer","Ather Energy","Manufacturing","2022-03-01",None,True,"Model compression and on-device inference for scooter perception stack. 3x inference speedup via ONNX quantisation."),_ch("Embedded Engineer","Bosch India","Manufacturing","2020-06-01","2022-02-28",False,"Embedded C++ for automotive ECU. Moved to ML.")],"education":[_ed("College of Engineering Pune","B.E.","Electronics","tier_3")],"redrob_signals":_rs(rr=0.78,gh=48,sa=80,pv=9,sbr=1,npd=30,smin=18,smax=32,sc={"Python":80,"TensorFlow":78,"ONNX":82})},
    {"candidate_id":"CAND028","profile":{"headline":"Data Scientist | Marketing Analytics","summary":"3 yrs. Attribution models, A/B testing, customer segmentation. Basic ML for business use cases.","current_title":"Data Scientist","years_of_experience":3.0,"location":"Mumbai, India","country":"India"},"skills":[_sk("Python","intermediate",36,10),_sk("SQL","advanced",36,14),_sk("scikit-learn","intermediate",24,8),_sk("R","intermediate",24,8),_sk("Tableau","advanced",30,12),_sk("A/B testing","advanced",30,12)],"career_history":[_ch("Data Scientist","Housing.com","Real Estate","2023-01-01",None,True,"User segmentation models and A/B testing platform. Basic sklearn classifiers for lead scoring."),_ch("Analyst","Deloitte","Consulting","2021-07-01","2022-12-31",False,"Data analysis and visualisation for clients.")],"education":[_ed("NMIMS Mumbai","MBA","Business Analytics","tier_3")],"redrob_signals":_rs(rr=0.88,gh=18,sa=50,pv=5,sbr=1,npd=30,smin=14,smax=24,sc={"SQL":82,"Python":72})},
    {"candidate_id":"CAND029","profile":{"headline":"Research Scientist | Reinforcement Learning","summary":"PhD in RL. Papers at NeurIPS and ICML. No industry production experience.","current_title":"Research Scientist","years_of_experience":6.0,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","expert",60,16),_sk("reinforcement learning","expert",60,20),_sk("PyTorch","expert",54,18),_sk("JAX","advanced",30,12),_sk("mathematics","expert",60,18)],"career_history":[_ch("Research Scientist","Google DeepMind India","Research","2022-01-01",None,True,"Published 3 papers on offline RL and decision transformers. No production ML code shipped."),_ch("PhD Student","IIT Madras","Research","2018-07-01","2021-12-31",False,"PhD thesis on model-based RL algorithms.")],"education":[_ed("IIT Madras","PhD","Computer Science","tier_1")],"redrob_signals":_rs(rr=0.55,gh=62,sa=50,pv=6,sbr=1,npd=90,wr=False,smin=45,smax=75,sc={"Python":90,"PyTorch":88,"RL":92})},
    {"candidate_id":"CAND030","profile":{"headline":"ML Engineer | E-Commerce & Personalisation","summary":"5 yrs building ML for e-commerce: search, personalisation, pricing. Transitioning to LLM.","current_title":"ML Engineer","years_of_experience":5.0,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("personalisation","expert",54,20),_sk("ranking","advanced",42,16),_sk("PyTorch","intermediate",24,8),_sk("A/B testing","advanced",36,14),_sk("Spark","advanced",36,12),_sk("LLM","beginner",8,2)],"career_history":[_ch("ML Engineer","Myntra","E-Commerce","2021-07-01",None,True,"Homepage personalisation for 50M users. Price elasticity models. Currently learning LLM integration."),_ch("Data Scientist","Jabong","E-Commerce","2019-06-01","2021-06-30",False,"Recommendation systems and demand forecasting.")],"education":[_ed("RV College","B.E.","Computer Science","tier_3")],"redrob_signals":_rs(rr=0.82,gh=45,sa=95,pv=10,sbr=2,npd=30,smin=25,smax=42,sc={"Python":85,"personalisation":84,"ranking":80})},
    # ── CAND031-040: Weak fits ──────────────────────────────────────────────────
    {"candidate_id":"CAND031","profile":{"headline":"Java Developer | Enterprise Applications","summary":"8 yrs Java enterprise developer. Spring Boot, microservices. Zero ML experience.","current_title":"Senior Java Developer","years_of_experience":8.0,"location":"Chennai, India","country":"India"},"skills":[_sk("Java","expert",96,28),_sk("Spring Boot","expert",72,24),_sk("SQL","expert",84,22),_sk("Maven","expert",84,20),_sk("REST APIs","expert",84,22)],"career_history":[_ch("Senior Java Developer","Cognizant","Consulting","2019-01-01",None,True,"Enterprise application development for banking client. Java, Spring Boot, Oracle DB. No ML."),_ch("Java Developer","Infosys","Consulting","2016-07-01","2018-12-31",False,"J2EE development for insurance applications.")],"education":[_ed("Anna University","B.E.","Computer Science","tier_3")],"redrob_signals":_rs(rr=0.75,gh=22,sa=35,pv=4,sbr=2,npd=60,smin=25,smax=45,sc={"Java":92,"SQL":88})},
    {"candidate_id":"CAND032","profile":{"headline":"Graphic Designer | UI/UX","summary":"5 yrs graphic and UI design. Adobe suite, Figma. No coding or ML background.","current_title":"Senior Designer","years_of_experience":5.0,"location":"Mumbai, India","country":"India"},"skills":[_sk("Figma","expert",60,22),_sk("Adobe Illustrator","expert",60,20),_sk("Photoshop","expert",60,20),_sk("UI design","expert",60,24),_sk("HTML","beginner",12,3)],"career_history":[_ch("Senior Designer","Byju's","EdTech","2020-06-01",None,True,"Led visual design for learning app. No technical or ML skills."),_ch("Graphic Designer","Ogilvy","Agency","2019-01-01","2020-05-31",False,"Brand design and marketing collateral.")],"education":[_ed("NID Ahmedabad","B.Des","Communication Design","tier_2")],"redrob_signals":_rs(rr=0.80,gh=5,sa=20,pv=2,sbr=0,npd=30,smin=12,smax=22,sc={})},
    {"candidate_id":"CAND033","profile":{"headline":"Business Analyst | Process & Requirements","summary":"4 yrs BA. Requirements gathering and process modelling. No technical skills.","current_title":"Business Analyst","years_of_experience":4.0,"location":"Hyderabad, India","country":"India"},"skills":[_sk("requirements","advanced",48,14),_sk("JIRA","expert",48,18),_sk("Excel","expert",48,18),_sk("process modelling","advanced",42,14),_sk("SQL","beginner",12,3)],"career_history":[_ch("Business Analyst","Capgemini","Consulting","2020-07-01",None,True,"Process automation requirements for banking client. BPMN diagrams and UAT. No ML."),_ch("Junior BA","Accenture","Consulting","2018-06-01","2020-06-30",False,"Requirements analysis and documentation.")],"education":[_ed("Osmania University","BCA","Computer Applications","tier_3")],"redrob_signals":_rs(rr=0.82,gh=2,sa=15,pv=1,sbr=0,npd=30,smin=10,smax=18,sc={})},
    {"candidate_id":"CAND034","profile":{"headline":"Sales Engineer | SaaS & Cloud","summary":"5 yrs pre-sales in cloud SaaS. Customer-facing technical role. No ML or data background.","current_title":"Senior Sales Engineer","years_of_experience":5.0,"location":"Gurgaon, India","country":"India"},"skills":[_sk("AWS","intermediate",36,10),_sk("Salesforce","advanced",48,16),_sk("SQL","beginner",12,3),_sk("presentation","expert",60,20),_sk("Python","beginner",6,2)],"career_history":[_ch("Senior Sales Engineer","Salesforce","SaaS","2021-03-01",None,True,"Technical demos and PoC builds for enterprise CRM sales. No ML work."),_ch("Sales Engineer","Oracle","Tech","2019-01-01","2021-02-28",False,"Pre-sales for Oracle Cloud. Customer-facing.")],"education":[_ed("SRM University","B.Tech","Mechanical Engineering","tier_3")],"redrob_signals":_rs(rr=0.70,gh=4,sa=12,pv=1,sbr=0,npd=30,wr=True,smin=18,smax=35,sc={})},
    {"candidate_id":"CAND035","profile":{"headline":"Content Writer | Tech & AI Topics","summary":"3 yrs writing about AI and ML. Strong communicator but no hands-on ML skills.","current_title":"Technical Content Writer","years_of_experience":3.0,"location":"Bangalore, India","country":"India"},"skills":[_sk("content writing","expert",36,16),_sk("SEO","advanced",30,12),_sk("AI knowledge","intermediate",24,8),_sk("Python","beginner",6,1)],"career_history":[_ch("Technical Content Writer","Analytics Vidhya","EdTech","2021-04-01",None,True,"Write tutorials and articles about ML and AI. No practical ML engineering."),_ch("Content Writer","Edureka","EdTech","2019-09-01","2021-03-31",False,"Course content for online learning platform.")],"education":[_ed("Christ University","B.Com","Economics","tier_3")],"redrob_signals":_rs(rr=0.90,gh=8,sa=22,pv=2,sbr=0,npd=15,smin=8,smax=15,sc={})},
    {"candidate_id":"CAND036","profile":{"headline":"iOS Developer | Swift & Mobile","summary":"6 yrs iOS development. Core ML used once for camera filter. No production ML.","current_title":"Senior iOS Developer","years_of_experience":6.0,"location":"Bangalore, India","country":"India"},"skills":[_sk("Swift","expert",72,24),_sk("iOS","expert",72,26),_sk("Xcode","expert",72,22),_sk("Core ML","beginner",8,2),_sk("Python","beginner",6,1)],"career_history":[_ch("Senior iOS Developer","Dream11","Gaming","2020-06-01",None,True,"Fantasy sports iOS app for 100M users. Core ML used once for image filter. No ML engineering."),_ch("iOS Developer","Urban Company","Services","2018-07-01","2020-05-31",False,"iOS app development for home services marketplace.")],"education":[_ed("DAIICT Gandhinagar","B.Tech","ICT","tier_2")],"redrob_signals":_rs(rr=0.78,gh=30,sa=28,pv=3,sbr=1,npd=45,smin=28,smax=48,sc={"Swift":92,"iOS":90})},
    {"candidate_id":"CAND037","profile":{"headline":"Cloud Solutions Architect | AWS & Azure","summary":"7 yrs cloud architecture. Deep AWS/Azure but no ML systems experience.","current_title":"Senior Solutions Architect","years_of_experience":7.0,"location":"Hyderabad, India","country":"India"},"skills":[_sk("AWS","expert",72,24),_sk("Azure","expert",60,20),_sk("Terraform","expert",48,18),_sk("Python","intermediate",24,8),_sk("cloud architecture","expert",72,26)],"career_history":[_ch("Senior Solutions Architect","Accenture","Consulting","2020-01-01",None,True,"Cloud migration projects for Fortune 500 clients. AWS/Azure. No ML workloads."),_ch("Cloud Engineer","IBM","Consulting","2017-07-01","2019-12-31",False,"Cloud infrastructure setup and migration.")],"education":[_ed("JNTU Hyderabad","B.Tech","Computer Science","tier_3")],"redrob_signals":_rs(rr=0.72,gh=25,sa=30,pv=3,sbr=1,npd=60,smin=35,smax=55,sc={"AWS":90,"Azure":88})},
    {"candidate_id":"CAND038","profile":{"headline":"Fresh Graduate | CS & AI Coursework","summary":"Final year B.Tech. Completed ML courses online. No work experience. Internship at startup.","current_title":"ML Intern","years_of_experience":0.5,"location":"Delhi, India","country":"India"},"skills":[_sk("Python","intermediate",18,6),_sk("scikit-learn","beginner",12,4),_sk("machine learning","beginner",12,4),_sk("SQL","beginner",12,3),_sk("TensorFlow","beginner",8,2)],"career_history":[_ch("ML Intern","AI Startup","AI","2025-12-01",None,True,"Built a sentiment classifier for 6 months. Kaggle competitions. No production deployments.")],"education":[_ed("DTU Delhi","B.Tech","Computer Science","tier_2")],"redrob_signals":_rs(rr=0.95,gh=20,sa=15,pv=1,sbr=0,npd=0,smin=5,smax=10,sc={"Python":65,"scikit-learn":55})},
    {"candidate_id":"CAND039","profile":{"headline":"HR Technology Specialist | HRMS & ATS","summary":"5 yrs in HR tech. Implemented ATS and HRMS systems. No ML or data background.","current_title":"HR Technology Specialist","years_of_experience":5.0,"location":"Pune, India","country":"India"},"skills":[_sk("SAP SuccessFactors","expert",60,20),_sk("Workday","advanced",36,14),_sk("Excel","expert",60,22),_sk("process automation","advanced",36,12)],"career_history":[_ch("HR Technology Specialist","Mahindra","Manufacturing","2019-07-01",None,True,"HRMS implementation and ATS configuration. No ML."),_ch("HR Executive","Infosys BPM","Consulting","2018-06-01","2019-06-30",False,"HR operations and payroll processing.")],"education":[_ed("Symbiosis Pune","MBA","HR","tier_3")],"redrob_signals":_rs(rr=0.80,gh=2,sa=10,pv=1,sbr=0,npd=30,smin=10,smax=18,sc={})},
    {"candidate_id":"CAND040","profile":{"headline":"Python Developer | Automation & Scripting","summary":"3 yrs writing Python scripts and automation tools. Some pandas usage. No ML or AI work.","current_title":"Python Developer","years_of_experience":3.0,"location":"Noida, India","country":"India"},"skills":[_sk("Python","advanced",36,12),_sk("Selenium","advanced",30,10),_sk("pandas","intermediate",24,8),_sk("REST APIs","advanced",30,10),_sk("SQL","intermediate",24,8)],"career_history":[_ch("Python Developer","HCL","Consulting","2021-08-01",None,True,"Test automation scripts in Selenium+Python. Data extraction with pandas. No ML."),_ch("Junior Developer","Wipro","Consulting","2021-01-01","2021-07-31",False,"Python scripting for ETL automation.")],"education":[_ed("Amity University","B.Tech","Computer Science","tier_3")],"redrob_signals":_rs(rr=0.85,gh=28,sa=45,pv=4,sbr=1,npd=30,smin=12,smax=20,sc={"Python":78,"SQL":72})},
    # ── CAND041-050: More strong / Elite candidates ─────────────────────────────
    {"candidate_id":"CAND041","profile":{"headline":"Principal AI Engineer | LLM Infrastructure","summary":"9 yrs. Built LLM inference infra used by 5M+ users. vLLM, TGI, model sharding expert.","current_title":"Principal AI Engineer","years_of_experience":9.0,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","expert",96,28),_sk("LLM","expert",36,22),_sk("vLLM","expert",24,18),_sk("model serving","expert",36,22),_sk("CUDA","advanced",30,14),_sk("PyTorch","expert",72,22),_sk("Kubernetes","expert",60,20)],"career_history":[_ch("Principal AI Engineer","Sarvam AI","AI","2023-01-01",None,True,"Built multi-tenant LLM serving infra handling 50K RPM. Model sharding across 8×A100s. Reduced cost per token 70%."),_ch("Staff Engineer","Flipkart","E-Commerce","2018-07-01","2022-12-31",False,"ML platform and model serving for product search serving 1B+ queries/day.")],"education":[_ed("IIT Bombay","M.Tech","Computer Science","tier_1")],"redrob_signals":_rs(otw=False,rr=0.65,gh=88,sa=220,pv=25,sbr=7,npd=90,wr=False,smin=80,smax=120,sc={"LLM":94,"Python":92,"vLLM":92})},
    {"candidate_id":"CAND042","profile":{"headline":"ML Engineer | Conversational AI & Dialogue","summary":"5 yrs. End-to-end voice AI: ASR, NLU, dialogue management, TTS in production.","current_title":"Senior ML Engineer","years_of_experience":5.0,"location":"Hyderabad, India","country":"India"},"skills":[_sk("Python","expert",60,20),_sk("ASR","advanced",42,16),_sk("NLU","advanced",42,16),_sk("dialogue systems","advanced",36,14),_sk("LLM","advanced",24,12),_sk("PyTorch","advanced",42,14),_sk("TTS","intermediate",24,8)],"career_history":[_ch("Senior ML Engineer","Vernacular.ai","AI","2021-06-01",None,True,"Production voice bot for telecom serving 10M+ calls/day. ASR+NLU pipeline. Improved task success rate from 72% to 88%."),_ch("ML Engineer","Uniphore","AI","2019-07-01","2021-05-31",False,"Dialogue management and ASR for enterprise voice AI.")],"education":[_ed("IIIT Bangalore","M.Tech","Data Science","tier_2")],"redrob_signals":_rs(rr=0.85,gh=65,sa=130,pv=14,sbr=3,npd=30,smin=30,smax=50,sc={"Python":88,"NLU":86,"LLM":82})},
    {"candidate_id":"CAND043","profile":{"headline":"ML Research Engineer | Efficient LLMs","summary":"4 yrs. Specialised in model compression: quantisation, pruning, distillation. Research + production.","current_title":"ML Research Engineer","years_of_experience":4.0,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","expert",48,18),_sk("quantisation","expert",36,16),_sk("model compression","expert",36,16),_sk("knowledge distillation","advanced",30,14),_sk("PyTorch","expert",42,16),_sk("ONNX","advanced",30,12),_sk("LLM","advanced",24,12)],"career_history":[_ch("ML Research Engineer","Qualcomm India","Tech","2022-04-01",None,True,"4-bit quantisation of LLMs for on-device inference. 8x memory reduction with <2% accuracy loss. Shipped in Snapdragon AI SDK."),_ch("Research Intern","Samsung Research India","Tech","2020-06-01","2022-03-31",False,"Model compression research for mobile devices.")],"education":[_ed("IIT Madras","M.Tech","AI","tier_1")],"redrob_signals":_rs(rr=0.88,gh=80,sa=140,pv=16,sbr=4,npd=30,smin=32,smax=52,sc={"Python":90,"quantisation":92,"PyTorch":88})},
    {"candidate_id":"CAND044","profile":{"headline":"ML Engineer | Graph ML & Knowledge Graphs","summary":"5 yrs. GNNs, knowledge graphs, entity resolution for fintech and e-commerce at scale.","current_title":"ML Engineer","years_of_experience":5.0,"location":"Pune, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("GNN","expert",48,18),_sk("knowledge graph","expert",48,18),_sk("PyTorch Geometric","advanced",36,14),_sk("Neo4j","advanced",42,16),_sk("entity resolution","advanced",36,14),_sk("LLM","intermediate",18,8)],"career_history":[_ch("ML Engineer","ICICI Bank","Finance","2021-08-01",None,True,"Knowledge graph for fraud ring detection. GNN model with 94% precision. Entity resolution at 10M+ customer scale."),_ch("Data Scientist","Tiger Analytics","Consulting","2019-07-01","2021-07-31",False,"Graph analytics for telecom churn prediction.")],"education":[_ed("COEP Pune","B.Tech","Computer Engineering","tier_2")],"redrob_signals":_rs(rr=0.82,gh=68,sa=110,pv=12,sbr=2,npd=45,smin=28,smax=48,sc={"Python":86,"GNN":88,"knowledge graph":86})},
    {"candidate_id":"CAND045","profile":{"headline":"ML Engineer | Speech & Audio AI","summary":"5 yrs. Production speech recognition and audio ML. Whisper fine-tuning, speaker diarisation.","current_title":"ML Engineer","years_of_experience":5.0,"location":"Chennai, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("speech recognition","expert",54,20),_sk("Whisper","advanced",24,12),_sk("audio ML","expert",48,18),_sk("speaker diarisation","advanced",30,12),_sk("PyTorch","advanced",42,14),_sk("LLM","intermediate",14,6)],"career_history":[_ch("ML Engineer","Koo App","Social","2021-09-01",None,True,"Fine-tuned Whisper for 10 Indian languages. Speaker diarisation for meeting transcription. 4% WER on benchmark."),_ch("AI Engineer","Slang Labs","AI","2019-06-01","2021-08-31",False,"Voice search in Indian languages for e-commerce apps.")],"education":[_ed("IIT Madras","B.Tech","Electrical Engineering","tier_1")],"redrob_signals":_rs(rr=0.85,gh=62,sa=105,pv=11,sbr=2,npd=30,smin=26,smax=44,sc={"Python":88,"speech recognition":90,"audio ML":88})},
    {"candidate_id":"CAND046","profile":{"headline":"Senior ML Engineer | Ranking & Ads","summary":"6 yrs in ads ML: CTR prediction, bid optimisation, contextual bandits. Production at massive scale.","current_title":"Senior ML Engineer","years_of_experience":6.0,"location":"Gurgaon, India","country":"India"},"skills":[_sk("Python","expert",72,22),_sk("CTR prediction","expert",60,22),_sk("ads ML","expert",60,20),_sk("contextual bandits","advanced",36,14),_sk("XGBoost","expert",60,20),_sk("PyTorch","advanced",42,14),_sk("Spark","expert",54,18)],"career_history":[_ch("Senior ML Engineer","InMobi","AdTech","2020-07-01",None,True,"CTR prediction model serving 50B impressions/day. Contextual bandit for real-time bid optimisation. 22% RPM improvement."),_ch("ML Engineer","Verizon Media","AdTech","2018-07-01","2020-06-30",False,"Ad click prediction and audience segmentation for DSP.")],"education":[_ed("IIT Kanpur","B.Tech","Computer Science","tier_1")],"redrob_signals":_rs(rr=0.75,gh=70,sa=145,pv=16,sbr=4,npd=60,wr=False,smin=45,smax=70,sc={"Python":90,"XGBoost":88,"CTR prediction":90})},
    {"candidate_id":"CAND047","profile":{"headline":"ML Engineer | Drug Discovery & BioML","summary":"5 yrs applying ML in computational biology. Protein structure, molecular property prediction.","current_title":"ML Engineer","years_of_experience":5.0,"location":"Hyderabad, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("PyTorch","expert",54,18),_sk("bioinformatics","expert",54,20),_sk("molecular ML","expert",48,18),_sk("transformers","advanced",36,14),_sk("RDKit","advanced",42,16)],"career_history":[_ch("ML Engineer","Aganitha Cognitive","BioTech","2021-01-01",None,True,"Molecular property prediction models for drug candidate screening. GNN over molecular graphs. Deployed for pharma client."),_ch("Computational Biologist","Strand Life Sciences","BioTech","2019-06-01","2020-12-31",False,"Bioinformatics pipelines and genomic data analysis.")],"education":[_ed("IISc Bangalore","M.Tech","Computational Biology","tier_1")],"redrob_signals":_rs(rr=0.82,gh=72,sa=80,pv=9,sbr=2,npd=60,wr=True,smin=28,smax=48,sc={"Python":88,"PyTorch":86,"bioinformatics":90})},
    {"candidate_id":"CAND048","profile":{"headline":"Senior Data Scientist | LTV & Growth ML","summary":"5 yrs. Built LTV prediction, pricing ML, and growth models for consumer apps.","current_title":"Senior Data Scientist","years_of_experience":5.0,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","expert",60,18),_sk("LTV modelling","expert",48,18),_sk("causal inference","advanced",36,14),_sk("PyTorch","intermediate",24,8),_sk("SQL","expert",60,22),_sk("experimentation","expert",48,18),_sk("LLM","beginner",6,2)],"career_history":[_ch("Senior Data Scientist","Meesho","E-Commerce","2021-03-01",None,True,"LTV model for 150M users. Dynamic pricing ML reducing CAC 18%. Causal inference for promotion attribution."),_ch("Data Scientist","OYO","Travel","2019-06-01","2021-02-28",False,"Demand forecasting and price optimisation for hotel inventory.")],"education":[_ed("IIIT Hyderabad","M.Tech","Data Science","tier_2")],"redrob_signals":_rs(rr=0.80,gh=50,sa=105,pv=11,sbr=2,npd=30,smin=28,smax=48,sc={"Python":86,"SQL":88,"causal inference":82})},
    {"candidate_id":"CAND049","profile":{"headline":"ML Engineer | NLP & Legal Tech","summary":"4 yrs applying NLP to legal documents. Contract analysis, clause extraction, LLM for legal QA.","current_title":"ML Engineer","years_of_experience":4.0,"location":"Mumbai, India","country":"India"},"skills":[_sk("Python","expert",48,16),_sk("NLP","expert",48,20),_sk("LLM","advanced",24,12),_sk("RAG","advanced",18,10),_sk("transformers","expert",42,16),_sk("spaCy","expert",36,14),_sk("contract analysis","expert",36,16)],"career_history":[_ch("ML Engineer","SpotDraft","LegalTech","2022-03-01",None,True,"Built LLM-powered contract analysis system for Fortune 500 legal teams. RAG over 10M+ legal documents. 40% faster contract review."),_ch("NLP Engineer","Keka HR","HRTech","2020-06-01","2022-02-28",False,"NLP for job description parsing and candidate matching.")],"education":[_ed("BITS Pilani","B.E.","Computer Science","tier_2")],"redrob_signals":_rs(rr=0.88,gh=58,sa=100,pv=11,sbr=2,npd=30,smin=22,smax=38,sc={"NLP":88,"LLM":82,"Python":86})},
    {"candidate_id":"CAND050","profile":{"headline":"AI Engineer | Production RAG & Agents","summary":"3 yrs. Specialist in production RAG systems and LLM agents. Shipped to 1M+ users.","current_title":"AI Engineer","years_of_experience":3.5,"location":"Bangalore, India","country":"India"},"skills":[_sk("Python","expert",42,16),_sk("RAG","expert",30,16),_sk("LLM agents","advanced",24,12),_sk("LangChain","expert",30,14),_sk("vector search","advanced",24,12),_sk("FastAPI","advanced",30,12),_sk("LLM","advanced",30,14)],"career_history":[_ch("AI Engineer","Keka HR","HRTech","2023-01-01",None,True,"Production RAG for HR knowledge base serving 1M+ employees. LLM agent for automated HR workflows. Reduced support tickets 45%."),_ch("Software Engineer","Freshworks","SaaS","2021-08-01","2022-12-31",False,"Backend API development. Moved to AI team.")],"education":[_ed("Vellore Institute of Technology","B.Tech","Computer Science","tier_3")],"redrob_signals":_rs(rr=0.92,gh=68,sa=130,pv=15,sbr=3,npd=15,smin=20,smax=35,sc={"RAG":88,"LLM":84,"Python":86})},
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


def _run_job_from_file(job_id: str, file_path: str) -> None:
    """Memory-efficient variant for large uploads — streams line-by-line, never loads all into RAM."""
    import os
    from ..services.offline_pipeline import run_pipeline_from_file
    job = _jobs[job_id]
    job["status"] = "running"

    def progress_cb(p: dict):
        job["progress"] = p

    try:
        output = run_pipeline_from_file(file_path, top_k=2000, progress_cb=progress_cb)
        job["status"] = "complete"
        job["results"] = output
    except Exception as exc:
        logger.exception("Pipeline job %s (file) failed", job_id)
        job["status"] = "error"
        job["error"] = str(exc)
    finally:
        try:
            os.unlink(file_path)
        except Exception:
            pass


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
        import os, tempfile
        # Stream upload directly to /tmp — never loads the whole file into RAM.
        # This allows large files (100K JSONL, 465 MB) without OOM on the 512 MB container.
        fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", dir="/tmp")
        try:
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = await file.read(65536)   # 64 KB chunks
                    if not chunk:
                        break
                    out.write(chunk)
        except Exception:
            try: os.unlink(tmp_path)
            except Exception: pass
            raise

        # Count candidates for progress reporting (fast sequential read)
        line_count = 0
        with open(tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    line_count += 1

        if line_count == 0:
            try: os.unlink(tmp_path)
            except Exception: pass
            raise HTTPException(status_code=400, detail="No valid candidates found in uploaded file.")

        job_id = str(uuid.uuid4())[:8]
        _jobs[job_id] = _make_job()
        _jobs[job_id]["progress"]["total"] = line_count
        t = threading.Thread(target=_run_job_from_file, args=(job_id, tmp_path), daemon=True)
        t.start()
        return {"job_id": job_id, "total_candidates": line_count, "status": "running"}

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
            logger.info("Challenge file not found — using embedded 50-candidate demo")
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

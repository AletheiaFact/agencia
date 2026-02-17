# PRD: Agencia Stack Modernization Initiative

**Document Version:** 1.0  
**Date:** August 30, 2025  
**Project:** Aletheia Fact-Checking System Modernization  
**Priority:** P0 (Critical - Security & Production Readiness)

## Executive Summary

Transform the Agencia fact-checking system from a development prototype to a production-ready, observable, and cost-optimized platform through comprehensive stack modernization. Address critical security vulnerabilities, implement modern architectural patterns, and establish enterprise-grade reliability.

**Timeline:** 8 weeks  
**ROI:** 40-70% cost reduction, 3-5x performance improvement  
**Risk Mitigation:** Eliminates critical security vulnerabilities, establishes production monitoring

## Current State Analysis

### Critical Issues
- **Security Risk:** `typing==3.7.4.3` deprecated package with vulnerabilities
- **Outdated Stack:** LangChain 0.1.20 (2+ major versions behind)
- **No Observability:** Zero monitoring, tracing, or debugging capabilities
- **Production Gaps:** Limited error handling, no caching, hardcoded configurations

### Performance Baseline
- **Response Time:** 10-30 seconds per fact-check
- **Cost per Query:** $0.15-0.30 (estimated)
- **Reliability:** 70-80% success rate
- **Observability:** 0% (no metrics)

## Goals & Success Metrics

### Primary Goals
1. **Security Compliance:** Eliminate all known vulnerabilities
2. **Production Readiness:** 99.5% uptime with comprehensive monitoring
3. **Cost Optimization:** 40-70% reduction in operational costs
4. **Performance:** 3-5x improvement in response times
5. **Developer Experience:** Modern tooling with debugging capabilities

### Success Metrics
- **Security:** Zero critical/high vulnerabilities in dependencies
- **Performance:** <5 second average response time (vs current 10-30s)
- **Reliability:** 99.5% uptime, <0.1% error rate
- **Cost:** 40-70% reduction in LLM API costs through optimization
- **Observability:** 100% request tracing, error tracking, performance monitoring

## Technical Strategy

### Phase 1: Critical Security & Foundation (Weeks 1-2)

#### 1.1 Python & Typing Modernization
**Problem:** `typing==3.7.4.3` security vulnerability
**Solution:** Migrate to Python 3.11+ with native typing

```python
# Current (vulnerable)
from typing import Union, List, Optional, Dict

# Modern (secure)
def process_claim(items: list[str | int]) -> dict[str, str] | None:
    pass
```

**Tasks:**
- Remove `typing==3.7.4.3` dependency
- Upgrade Python to 3.11+
- Update all type hints to modern syntax
- Add `from __future__ import annotations` for compatibility

#### 1.2 LangChain Critical Upgrade
**Problem:** LangChain 0.1.20 missing security patches and features
**Solution:** Migrate to LangChain 0.3.x using official migration tools

**Migration Command:**
```bash
pip install -U langchain-cli>=0.0.31
langchain-cli migrate --interactive ./app
```

**Breaking Changes Impact:**
- Agent creation patterns updated
- Memory management deprecated (migrate to LangGraph persistence)
- Tool integration improved
- Pydantic 2 required

### Phase 2: Architecture Decision (Weeks 3-4)

#### 2.1 Multi-Agent Orchestration Evaluation

**Option A: LangGraph 0.6+ (Recommended)**
- **Pros:** Production-ready, native LangChain integration, advanced state management
- **Cons:** Requires workflow redesign
- **Migration Effort:** Moderate
- **Long-term Value:** High

**Option B: Enhanced CrewAI 0.63.6**
- **Pros:** Minimal migration, familiar patterns
- **Cons:** Limited production features, debugging challenges
- **Migration Effort:** Low
- **Long-term Value:** Medium

**Option C: Microsoft AutoGen 0.4.0**
- **Pros:** Enterprise-grade, advanced conversation patterns
- **Cons:** High complexity, major architecture change
- **Migration Effort:** High
- **Long-term Value:** High (enterprise scenarios)

**Recommendation:** **LangGraph 0.6+** for optimal balance of production readiness and migration effort.

#### 2.2 LangGraph Architecture Design

```python
# Modernized workflow structure
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

class ModernFactCheckWorkflow:
    def __init__(self):
        # Persistent state management
        memory = SqliteSaver.from_conn_string("./checkpoints.db")
        
        workflow = StateGraph(EnhancedAgentState)
        
        # Enhanced nodes with error handling
        workflow.add_node("validate_claim", self.validate_claim)
        workflow.add_node("generate_questions", self.generate_questions)  
        workflow.add_node("research_gazettes", self.research_gazettes)
        workflow.add_node("research_online", self.research_online)
        workflow.add_node("cross_verify", self.cross_verify)
        workflow.add_node("generate_report", self.generate_report)
        
        # Intelligent routing with fallbacks
        workflow.add_conditional_edges(
            "validate_claim",
            self.route_research_strategy,
            {
                "gazette_search": "research_gazettes",
                "online_search": "research_online", 
                "both": "research_gazettes",
                "invalid": END
            }
        )
        
        self.app = workflow.compile(checkpointer=memory)
```

### Phase 3: Performance & Observability (Weeks 5-6)

#### 3.1 Observability Implementation

**LangSmith Integration (Recommended):**
```python
import os
from langsmith import Client

# Environment setup
os.environ["LANGSMITH_API_KEY"] = "your-key"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "agencia-fact-checking"

# Automatic tracing - no code changes needed!
```

**Custom Metrics:**
- Fact-check completion rate
- Average processing time by claim type
- Cost per fact-check
- Agent performance analytics
- Error rate by component

#### 3.2 Performance Optimization

**Caching Strategy:**
```python
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

# Semantic caching for similar claims
set_llm_cache(SQLiteCache(database_path="./cache.db"))
```

**Model Selection Strategy:**
```python
class IntelligentModelRouter:
    def select_model(self, task_type: str, complexity: float):
        if task_type == "subject_creation" and complexity > 0.8:
            return "gpt-4-turbo"  # High accuracy needs
        elif task_type == "data_analysis":
            return "gpt-3.5-turbo"  # Sufficient capability
        else:
            return "claude-3-haiku"  # Cost-effective default
```

### Phase 4: Production Hardening (Weeks 7-8)

#### 4.1 Infrastructure as Code
```python
# Enhanced server.py with production features
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog

app = FastAPI(
    title="Aletheia Agencia API",
    version="2.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None
)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(CORSMiddleware, allow_origins=["https://your-domain.com"])

# Structured logging
logger = structlog.get_logger()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        "request_processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    return response
```

#### 4.2 Enhanced Error Handling
```python
class EnhancedErrorHandler:
    def __init__(self):
        self.error_patterns = {
            "city_not_found": CityNotFoundError,
            "no_gazettes": NoGazettesFoundError,
            "api_timeout": APITimeoutError,
            "rate_limit": RateLimitError
        }
    
    async def handle_with_retry(self, operation, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## Technical Specifications

### Updated Dependencies

#### Core Stack
```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.116.1"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
pydantic = "^2.9.0"

# LangChain ecosystem (latest)
langchain = "^0.3.0"
langchain-community = "^0.3.0"
langchain-core = "^0.3.0"
langchain-openai = "^0.2.0"

# Orchestration - choose one:
langgraph = "^0.6.6"  # Recommended
# crewai = "^0.63.6"   # Alternative

# Observability - choose one:
langsmith = "^0.1.0"    # Commercial, LangChain native
# langfuse = "^2.0.0"   # Open source alternative

# Performance & Caching
redis = "^5.1.0"
aioredis = "^2.0.1"
asyncio-throttle = "^1.0.2"

# Security & Production
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
structlog = "^24.4.0"
prometheus-client = "^0.20.0"
```

#### Development Dependencies
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.3.0"
pytest-asyncio = "^0.24.0"
pytest-cov = "^5.0.0"
black = "^24.8.0"
isort = "^5.13.2"
mypy = "^1.11.0"
ruff = "^0.6.0"
pre-commit = "^3.8.0"
```

### Environment Configuration
```bash
# .env template
# Core APIs
OPENAI_API_KEY=your_openai_key
SERPAPI_API_KEY=your_serpapi_key

# Observability
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=agencia-production

# Caching
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# Production
ENVIRONMENT=production
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8080
WORKERS=4
```

## Implementation Roadmap

### Week 1-2: Foundation & Security
- [ ] Remove `typing==3.7.4.3` dependency
- [ ] Upgrade Python to 3.11+
- [ ] Migrate LangChain to 0.3.x
- [ ] Update FastAPI to 0.116.1
- [ ] Comprehensive testing of existing functionality

### Week 3-4: Architecture Modernization
- [ ] **Decision Point:** Choose orchestration framework
- [ ] Implement chosen framework (LangGraph recommended)
- [ ] Add observability (LangSmith recommended)
- [ ] Migrate agent configurations

### Week 5-6: Performance Optimization
- [ ] Implement caching layer (Redis + semantic caching)
- [ ] Add intelligent model routing
- [ ] Optimize prompt templates
- [ ] Implement parallel processing

### Week 7-8: Production Readiness
- [ ] Add comprehensive monitoring
- [ ] Implement structured logging
- [ ] Load testing and optimization
- [ ] Documentation and deployment updates

## Resource Requirements

### Engineering Resources
- **Full-Stack Developer:** 6-8 weeks (lead)
- **DevOps Engineer:** 2-3 weeks (infrastructure)
- **QA Engineer:** 1-2 weeks (testing validation)

### Infrastructure Requirements
- **Development Environment:** Enhanced with debugging tools
- **Staging Environment:** Production-like for testing
- **Monitoring Infrastructure:** LangSmith or self-hosted Langfuse
- **Caching Layer:** Redis cluster for production

### Budget Considerations
- **Development Time:** $15,000-25,000 (depending on team rates)
- **Infrastructure:** $200-500/month additional
- **Tooling:** $100-300/month (observability platforms)
- **Training:** $2,000-5,000 (team upskilling)

## Risk Mitigation

### Technical Risks
- **Migration Complexity:** Use LangChain CLI migration tools
- **Breaking Changes:** Comprehensive testing at each phase
- **Performance Regression:** Benchmark before/after each change

### Business Risks
- **Downtime:** Staged rollout with feature flags
- **Cost Increase:** Monitor costs closely during optimization
- **Team Learning Curve:** Dedicated training and documentation

### Contingency Plans
- **Rollback Strategy:** Git-based rollback with database migrations
- **Gradual Migration:** Feature-flag controlled gradual rollout
- **Support Plan:** 24/7 monitoring during initial deployment

## Success Definition

### Technical Success
- Zero critical/high security vulnerabilities
- 99.5% uptime in production
- <5 second average response time
- 100% request observability

### Business Success
- 40-70% cost reduction achieved
- Developer productivity increased 2-3x
- Customer satisfaction maintained/improved
- Future architectural foundation established

## Next Steps

### Immediate Actions (Next 48 Hours)
1. **Security Fix:** Remove vulnerable typing dependency
2. **Environment Audit:** Assess current development environment
3. **Stakeholder Alignment:** Confirm modernization timeline and resources

### Week 1 Planning
1. **Architecture Decision:** Finalize orchestration framework choice
2. **Team Training:** Schedule LangGraph/LangChain 0.3 training
3. **Environment Setup:** Prepare development/staging environments

### Go/No-Go Decision Points
- **Week 2:** Security fixes validated, LangChain migration successful
- **Week 4:** New architecture functional, observability implemented
- **Week 6:** Performance targets met, production testing successful

---

**Stakeholders:**
- **Engineering Team:** Implementation and maintenance
- **Product Team:** Feature requirements and user impact
- **DevOps Team:** Infrastructure and deployment
- **Security Team:** Vulnerability assessment and compliance

**Contact:** [Your Name] - [Your Email]  
**Review Cycle:** Weekly during implementation, monthly post-deployment
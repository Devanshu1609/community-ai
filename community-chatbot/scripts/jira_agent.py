from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import os
import re
from fastapi.middleware.cors import CORSMiddleware
import json
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from jira_pipeline import JiraPipeline  # <-- Import JiraPipeline

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# Pydantic models
class JiraQueryRequest(BaseModel):
    query: str
    use_fallback: bool = True 

class JiraQueryResponse(BaseModel):
    response: str
    query_used: str
    method_used: str 
    success: bool

# Create the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_jira_components()
    yield

# Define app with lifespan
app = FastAPI(title="Jira Agent API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
jira_agent = None
jira_wrapper = None
llm = None
jql_generation_prompt = None
summarization_prompt = None
jira_pipeline = None  # <-- New global for JiraPipeline

def initialize_jira_components():
    """Initialize all Jira and LangChain components"""
    global jira_agent, jira_wrapper, llm, jql_generation_prompt, summarization_prompt, jira_pipeline
    
    print("Initializing Jira and LangChain components...")
    
    # Initialize LangChain Jira components
    jira_wrapper = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(jira_wrapper)
    tools = toolkit.get_tools()
    
    llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")
    
    agent_kwargs = {
        "prefix": """You are a specialized Jira assistant.
You MUST use the provided tools to answer questions about Jira.
Do NOT answer any questions from your own knowledge.
If a user's query seems like a general knowledge question, you MUST assume it refers to data within Jira.

*** CRITICAL JQL RULE ***
When filtering by a field with a string value that contains spaces (like a person's name, a project name, or a summary), you MUST enclose the value in single or double quotes.
CORRECT: assignee = 'Aru Sharma'
INCORRECT: assignee = Aru Sharma
CORRECT: summary ~ '"New Login Button"'
INCORRECT: summary ~ 'New Login Button'

Always format your response as a Thought, an Action, and an Action Input.
Begin!"""
    }
    
    jira_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs=agent_kwargs,
    )
    
    jql_generation_prompt = PromptTemplate.from_template(
    """You are an expert in Jira Query Language (JQL). Your sole task is to convert a user's natural language request into a valid JQL query.
You must only respond with the JQL query string and nothing else.

--- Important JQL Syntax Rules ---
1.  **Quoting:** Any string value containing spaces or special characters MUST be enclosed in single ('') or double ("") quotes.
    -   Example: `assignee = 'Aru Sharma'`
    -   Example: `summary ~ '"Detailed new feature"'`
2.  **Usernames:** When searching for an assignee, it is best to use their name in quotes.
3.  **Linked Issues:** Use `issue in linkedIssues('KEY')`, not `issueLink = KEY`
4.  **Operators:** Use `=` for single values, `IN` for multiple.
5.  **Dates:** Example: `created < '2019-01-01'`
5. **Case:** Fields lowercase, keywords uppercase
   - Example: status = 'Open' ORDER BY created DESC
6. **Allowed fields:** project, status, assignee, reporter, issuetype,
   priority, created, updated, resolution, labels, summary, description

--- Examples ---
User Request: "find all tickets in the 'PROJ' project"
JQL Query: project = 'PROJ'

User Request: "show me all open bugs in the 'Mobile' project assigned to Aru Sharma"
JQL Query: project = 'Mobile' AND issuetype = 'Bug' AND status = 'Open' AND assignee = 'Aru Sharma'

User Request: "what were the top 5 highest priority issues created last week?"
JQL Query: created >= -7d ORDER BY priority DESC

Now, convert the following user request into a JQL query.

User Request: "{user_query}"
JQL Query:"""
)
    
    summarization_prompt = PromptTemplate.from_template(
        """You are a helpful assistant. The user asked the following question:

"{user_query}"

An AI agent attempted to answer this but failed. As a fallback, we ran a JQL query and got the following raw Jira issue data.
Please analyze this data and provide a clear, concise, and helpful answer to the user's original question. If the data seems irrelevant or empty, state that you couldn't find relevant information.

JSON Data:
{json_data}

Based on the data, answer the user's question.
"""
    )

    # Initialize JiraPipeline
    jira_pipeline = JiraPipeline(
        server_url=os.getenv("JIRA_INSTANCE_URL"),
        username=os.getenv("JIRA_USERNAME"),
        token=os.getenv("JIRA_API_TOKEN")
    )


def validate_and_fix_jql(jql: str) -> str:
    """
    Validates and fixes common JQL mistakes:
    - Ensures string values with spaces are quoted
    - Fixes linked issue syntax
    - Cleans up extra quotes
    - Normalizes ORDER BY
    """
    fixed_jql = jql.strip()

    # Fix linkedIssues syntax
    fixed_jql = re.sub(r"issueLink\s*=\s*([A-Z]+-\d+)", r"issue in linkedIssues('\1')", fixed_jql)

    # Quote assignee with spaces
    fixed_jql = re.sub(
        r"assignee\s*=\s*([A-Za-z]+\s+[A-Za-z]+)",
        lambda m: f"assignee = '{m.group(1)}'",
        fixed_jql
    )

    # Quote project with spaces
    fixed_jql = re.sub(
        r"project\s*=\s*([A-Za-z]+\s+[A-Za-z]+)",
        lambda m: f"project = '{m.group(1)}'",
        fixed_jql
    )

    # Quote summary search with spaces
    fixed_jql = re.sub(
        r"summary\s*~\s*([^'\"]\S+)",
        lambda m: f"summary ~ '\"{m.group(1)}\"'",
        fixed_jql
    )

    # ✅ Normalize IN clauses
    fixed_jql = re.sub(r"\(\s*", "(", fixed_jql)
    fixed_jql = re.sub(r"\s*\)", ")", fixed_jql)
    fixed_jql = re.sub(r",\s*", ", ", fixed_jql)

    # ✅ Uppercase operators
    keywords = ["order by", "and", "or", "not", "in", "is", "empty"]
    for kw in keywords:
        fixed_jql = re.sub(rf"\b{kw}\b", kw.upper(), fixed_jql, flags=re.IGNORECASE)


    # Remove double quotes errors
    fixed_jql = fixed_jql.replace("''", "'").replace('""', '"')
    fixed_jql = re.sub(r"[;.,]+$", "", fixed_jql)

    # Normalize ORDER BY
    fixed_jql = re.sub(r"order by", "ORDER BY", fixed_jql, flags=re.IGNORECASE)

    return fixed_jql


def intelligent_agent_run(query: str):
    """
    Tries to run the main agent. If it fails, it uses an LLM to generate
    a JQL query from the user's input and executes that instead using JiraPipeline.
    """
    try:
        print("--- Attempting main agent execution ---")
        response = jira_agent.run(query)
        return {
            "response": response,
            "method_used": "agent",
            "query_used": query,
            "success": True
        }
    except Exception as e:
        print("\n--- Agent failed, switching to intelligent fallback mode ---")
        print(f"Error: {e}\n")

        # Use the LLM to generate a JQL query from the user's original query
        print("Generating JQL from natural language...")
        jql_generation_chain = jql_generation_prompt | llm
        generated_jql = jql_generation_chain.invoke({"user_query": query}).content
        print(f"Dynamically Generated JQL: '{generated_jql}'")

        fixed_jql = validate_and_fix_jql(generated_jql)
        print(f"Validated & Fixed JQL: {fixed_jql}")

        # Execute the generated JQL query using JiraPipeline
        print("Fetching and normalizing Jira data via pipeline...")
        try:
            df = jira_pipeline.fetch_tickets(fixed_jql)
            df = jira_pipeline.normalize_data(df)
            fallback_data = df.to_dict(orient="records")

            if not fallback_data:
                return {
                    "response": "The generated JQL query ran successfully but returned no issues. Please try rephrasing your request or be more specific.",
                    "method_used": "fallback",
                    "query_used": fixed_jql,
                    "success": True
                }

            # Summarize results for the user
            print("Summarizing JiraPipeline results for the user...")
            summarization_chain = summarization_prompt | llm
            final_response = summarization_chain.invoke({
                "user_query": query,
                "json_data": fallback_data 
            }).content
            
            return {
                "response": final_response,
                "method_used": "fallback",
                "query_used": fixed_jql,
                "success": True
            }

        except Exception as fallback_e:
            print(f"Fallback JQL execution also failed: {fallback_e}")
            return {
                "response": f"I'm sorry, I couldn't process your request. Both the primary agent and the fallback query failed. The last error was: {fallback_e}",
                "method_used": "failed",
                "query_used": query,
                "success": False
            }

@app.options("/jira/query")
async def options_jira_query():
    return {}

@app.post("/jira/query", response_model=JiraQueryResponse)
async def query_jira(request: JiraQueryRequest):
    """Main endpoint to query Jira using natural language"""
    try:
        if not jira_agent:
            raise HTTPException(status_code=500, detail="Jira agent not initialized")
        
        if request.use_fallback:
            result = intelligent_agent_run(request.query)
        else:
            try:
                response = jira_agent.run(request.query)
                result = {
                    "response": response,
                    "method_used": "agent",
                    "query_used": request.query,
                    "success": True
                }
            except Exception as e:
                result = {
                    "response": f"Agent failed: {str(e)}",
                    "method_used": "agent",
                    "query_used": request.query,
                    "success": False
                }
        
        return JiraQueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jira/direct-jql")
async def direct_jql_query(jql_query: str):
    """Direct JQL query endpoint"""
    try:
        if not jira_wrapper:
            raise HTTPException(status_code=500, detail="Jira wrapper not initialized")
        
        result = jira_wrapper.run(mode="jql", query=jql_query)
        return {
            "jql_query": jql_query,
            "result": result,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jira/generate-jql")
async def generate_jql(natural_query: str):
    """Generate JQL from natural language"""
    try:
        if not llm or not jql_generation_prompt:
            raise HTTPException(status_code=500, detail="Components not initialized")
        
        jql_generation_chain = jql_generation_prompt | llm
        generated_jql = jql_generation_chain.invoke({"user_query": natural_query}).content
        generated_jql = generated_jql.strip().strip("'\"")
        
        return {
            "natural_query": natural_query,
            "generated_jql": generated_jql,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jira/fetch-pipeline")
async def fetch_jira_pipeline(project: str, max_results: int = 50):
    """Fetch, clean, and normalize Jira tickets using the JiraPipeline."""
    try:
        if not jira_pipeline:
            raise HTTPException(status_code=500, detail="Jira pipeline not initialized")
        
        df = jira_pipeline.fetch_tickets(f"project = {project} ORDER BY created DESC", max_results=max_results)
        df = jira_pipeline.normalize_data(df)
        
        return {
            "project": project,
            "tickets": df.to_dict(orient="records"),
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Jira tickets via pipeline: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "jira_initialized": jira_agent is not None,
        "wrapper_initialized": jira_wrapper is not None,
        "pipeline_initialized": jira_pipeline is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

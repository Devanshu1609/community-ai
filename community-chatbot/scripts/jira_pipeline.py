# jira_pipeline.py
from jira import JIRA
import pandas as pd
import re
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JiraPipeline:
    def __init__(self, server_url: str, username: str, token: str):
        """
        Initialize connection to Jira.
        """
        try:
            self.jira = JIRA(server=server_url, basic_auth=(username, token))
            logger.info("Connected to Jira successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Jira: {e}")
            raise e

    def fetch_tickets(self, jql: str, max_results: int = 100) -> pd.DataFrame:
        """
        Fetch Jira tickets using a JQL query.
        Returns a pandas DataFrame.
        """
        try:
            issues = self.jira.search_issues(jql, maxResults=max_results)
            data = []
            for issue in issues:
                data.append({
                    "key": issue.key,
                    "summary": issue.fields.summary,
                    "description": issue.fields.description,
                    "status": issue.fields.status.name,
                    "priority": issue.fields.priority.name if issue.fields.priority else None,
                    "assignee": issue.fields.assignee.displayName if issue.fields.assignee else None,
                    "created": issue.fields.created,
                    "updated": issue.fields.updated,
                })
            logger.info(f"Fetched {len(data)} tickets from Jira.")
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching Jira tickets: {e}")
            return pd.DataFrame()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text fields.
        Removes HTML, extra spaces, and normalizes whitespace.
        """
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning and normalization to all text fields.
        Converts dates to UTC datetime and standardizes status/priority.
        """
        if df.empty:
            return df
        df["summary"] = df["summary"].apply(self.clean_text)
        df["description"] = df["description"].apply(self.clean_text)
        df["created"] = pd.to_datetime(df["created"], utc=True)  # parse as UTC
        df["updated"] = pd.to_datetime(df["updated"], utc=True)  # parse as UTC
        df["status"] = df["status"].str.lower()
        df["priority"] = df["priority"].str.lower()
        return df

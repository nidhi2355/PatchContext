import os
import json
from git import Repo
from github import Github
from datetime import datetime
from dotenv import load_dotenv

# Load API keys from the .env file
load_dotenv()

class RepositoryExtractor:
    def __init__(self, repo_url: str, clone_dir: str):
        self.repo_url = repo_url
        self.clone_dir = clone_dir
        
        # Initialize GitHub API client. Uses token if available to prevent strict rate limits.
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.gh = Github(self.github_token) if self.github_token else Github()
        
        # Extract "fastapi/fastapi" from the URL
        self.repo_name = repo_url.split("github.com/")[-1].replace(".git", "")

    def clone_or_pull_repo(self):
        """Clones the repository if it doesn't exist, otherwise pulls latest changes."""
        if not os.path.exists(self.clone_dir):
            print(f"Cloning {self.repo_url} into {self.clone_dir}...")
            Repo.clone_from(self.repo_url, self.clone_dir)
        else:
            print(f"Repository already exists at {self.clone_dir}. Pulling latest changes...")
            repo = Repo(self.clone_dir)
            repo.remotes.origin.pull()
            
    def extract_commits(self, max_commits=1000):
        """Extracts the git commit history."""
        repo = Repo(self.clone_dir)
        commits_data = []
        
        print(f"Extracting up to {max_commits} commits...")
        # FastAPI's primary branch is 'master'
        for commit in repo.iter_commits('master', max_count=max_commits): 
            commits_data.append({
                "sha": commit.hexsha,
                "author": commit.author.name,
                "date": commit.committed_datetime.isoformat(),
                "message": commit.message.strip(),
                "type": "commit"
            })
            
        return commits_data

    def extract_issues_and_prs(self, max_items=200):
        """Extracts recent issues and pull requests via GitHub REST API."""
        print(f"Extracting up to {max_items} issues and PRs...")
        gh_repo = self.gh.get_repo(self.repo_name)
        
        items_data = []
        # PyGithub's get_issues() returns both Issues and PRs combined
        issues = gh_repo.get_issues(state='all')
        
        count = 0
        for item in issues:
            if count >= max_items:
                break
            
            is_pr = item.pull_request is not None
            
            items_data.append({
                "number": item.number,
                "title": item.title,
                "body": item.body or "",
                "state": item.state,
                "created_at": item.created_at.isoformat(),
                "type": "pull_request" if is_pr else "issue",
                "html_url": item.html_url
            })
            count += 1
            
        return items_data

    def save_data(self, data, filename):
        """Saves extracted data to a JSON file in the raw data folder."""
        filepath = os.path.join("data", "raw", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Data saved successfully to {filepath}")


if __name__ == "__main__":
    # Configuration for the FastAPI repository
    FASTAPI_URL = "https://github.com/fastapi/fastapi.git"
    CLONE_DIR = "data/raw/fastapi_repo"
    
    extractor = RepositoryExtractor(FASTAPI_URL, CLONE_DIR)
    
    # 1. Clone the codebase locally
    extractor.clone_or_pull_repo()
    
    # 2. Extract and save commits
    commits = extractor.extract_commits(max_commits=1000)
    extractor.save_data(commits, "commits.json")
    
    # 3. Extract and save Issues & PRs
    issues_prs = extractor.extract_issues_and_prs(max_items=200)
    extractor.save_data(issues_prs, "issues_prs.json")
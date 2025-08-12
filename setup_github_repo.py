#!/usr/bin/env python3
"""
GitHub Repository Setup Script for EdunSight
============================================

This script automates the process of:
1. Initializing a Git repository
2. Creating a .gitignore file
3. Adding all project files
4. Making initial commit
5. Creating GitHub repository (requires GitHub CLI)
6. Pushing to GitHub

Prerequisites:
- Git installed and configured
- GitHub CLI (gh) installed and authenticated
- Internet connection

Usage:
    python setup_github_repo.py

Author: EdunSight Team
Date: August 12, 2025
"""

import os
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


class GitHubRepoSetup:
    """Automates GitHub repository setup and push"""
    
    def __init__(self, repo_name="EdunSight", description="Student Performance Prediction ML Pipeline"):
        self.repo_name = repo_name
        self.description = description
        self.project_dir = Path.cwd()
        self.success_steps = []
        self.failed_steps = []
        
    def run_command(self, command, description="", check=True):
        """Run a shell command and handle errors"""
        print(f"🔄 {description}")
        print(f"   Command: {command}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=self.project_dir
            )
            
            if result.returncode == 0:
                print(f"✅ {description} - SUCCESS")
                if result.stdout.strip():
                    print(f"   Output: {result.stdout.strip()}")
                self.success_steps.append(description)
                return result
            else:
                print(f"❌ {description} - FAILED")
                print(f"   Error: {result.stderr.strip()}")
                self.failed_steps.append(description)
                if check:
                    raise subprocess.CalledProcessError(result.returncode, command, result.stderr)
                return result
                
        except Exception as e:
            print(f"❌ {description} - EXCEPTION: {e}")
            self.failed_steps.append(description)
            if check:
                raise
            return None
    
    def check_prerequisites(self):
        """Check if required tools are installed"""
        print("🔍 Checking prerequisites...")
        
        # Check Git
        try:
            result = self.run_command("git --version", "Checking Git installation")
            git_version = result.stdout.strip()
            print(f"   Git: {git_version}")
        except:
            print("❌ Git is not installed or not in PATH")
            return False
        
        # Check GitHub CLI
        try:
            result = self.run_command("gh --version", "Checking GitHub CLI installation")
            gh_version = result.stdout.strip().split('\n')[0]
            print(f"   GitHub CLI: {gh_version}")
        except:
            print("❌ GitHub CLI is not installed or not in PATH")
            print("   Install from: https://cli.github.com/")
            return False
        
        # Check GitHub CLI authentication
        try:
            result = self.run_command("gh auth status", "Checking GitHub CLI authentication")
            print("   GitHub CLI: Authenticated ✅")
        except:
            print("❌ GitHub CLI is not authenticated")
            print("   Run: gh auth login")
            return False
        
        return True
    
    def create_gitignore(self):
        """Create comprehensive .gitignore file"""
        gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
coverage_html/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# ML/Data Science specific
*.joblib
*.pkl
*.pickle
models/*.onnx
data/raw/*.zip
data/processed/
logs/*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/

# Local development
local_config.yaml
*.local.*
"""
        
        gitignore_path = self.project_dir / ".gitignore"
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        
        print("✅ Created comprehensive .gitignore file")
        return True
    
    def setup_git_repo(self):
        """Initialize Git repository and make initial commit"""
        
        # Initialize Git repository
        self.run_command("git init", "Initializing Git repository")
        
        # Configure Git (if not already configured)
        try:
            self.run_command("git config user.name", "Checking Git user name", check=False)
        except:
            self.run_command('git config user.name "EdunSight Developer"', "Setting Git user name")
        
        try:
            self.run_command("git config user.email", "Checking Git user email", check=False)
        except:
            self.run_command('git config user.email "developer@edusight.ai"', "Setting Git user email")
        
        # Create .gitignore
        self.create_gitignore()
        
        # Add all files
        self.run_command("git add .", "Adding all project files to Git")
        
        # Create initial commit
        commit_message = f"🎉 Initial commit - EdunSight ML Pipeline\\n\\nFeatures:\\n- Student performance prediction\\n- MVVM architecture\\n- Streamlit web interface\\n- LightGBM model\\n- Comprehensive test suite\\n\\nCommitted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.run_command(f'git commit -m "{commit_message}"', "Creating initial commit")
        
        return True
    
    def create_github_repo(self):
        """Create GitHub repository using GitHub CLI"""
        
        # Create repository
        create_cmd = f'gh repo create {self.repo_name} --description "{self.description}" --public'
        self.run_command(create_cmd, f"Creating GitHub repository '{self.repo_name}'")
        
        return True
    
    def push_to_github(self):
        """Push local repository to GitHub"""
        
        # Add remote origin
        remote_url = f"https://github.com/$(gh api user --jq .login)/{self.repo_name}.git"
        
        # Get GitHub username
        result = self.run_command("gh api user --jq .login", "Getting GitHub username")
        username = result.stdout.strip()
        
        remote_url = f"https://github.com/{username}/{self.repo_name}.git"
        self.run_command(f"git remote add origin {remote_url}", "Adding GitHub remote origin")
        
        # Rename branch to main (if needed)
        self.run_command("git branch -M main", "Setting main branch")
        
        # Push to GitHub
        self.run_command("git push -u origin main", "Pushing to GitHub repository")
        
        return True, remote_url
    
    def create_readme_badge(self, repo_url):
        """Add badges and repository info to README"""
        readme_path = self.project_dir / "README.md"
        
        if readme_path.exists():
            # Read existing README
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add badges at the top
            badges = f"""# EdunSight - Student Performance Prediction ML Pipeline

[![GitHub Repository]({repo_url}/badge.svg)]({repo_url})
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🚀 **Repository**: [{repo_url}]({repo_url})

"""
            
            # Replace the existing title and add badges
            lines = content.split('\n')
            new_lines = []
            title_found = False
            
            for line in lines:
                if line.startswith('# EdunSight') and not title_found:
                    new_lines.extend(badges.strip().split('\n'))
                    title_found = True
                elif not (line.startswith('# EdunSight') and title_found):
                    new_lines.append(line)
            
            # Write updated README
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            
            print("✅ Updated README.md with repository badges")
    
    def generate_report(self):
        """Generate setup report"""
        print("\n" + "="*60)
        print("🎯 GITHUB REPOSITORY SETUP COMPLETE")
        print("="*60)
        
        print(f"\n📊 **Setup Summary:**")
        print(f"   Repository Name: {self.repo_name}")
        print(f"   Description: {self.description}")
        print(f"   Project Directory: {self.project_dir}")
        
        print(f"\n✅ **Successful Steps ({len(self.success_steps)}):**")
        for step in self.success_steps:
            print(f"   ✓ {step}")
        
        if self.failed_steps:
            print(f"\n❌ **Failed Steps ({len(self.failed_steps)}):**")
            for step in self.failed_steps:
                print(f"   ✗ {step}")
        
        print(f"\n🔗 **Repository URL:**")
        try:
            result = subprocess.run("gh repo view --web --json url -q .url", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                repo_url = result.stdout.strip()
                print(f"   {repo_url}")
            else:
                result = subprocess.run("gh api user --jq .login", shell=True, capture_output=True, text=True)
                username = result.stdout.strip()
                print(f"   https://github.com/{username}/{self.repo_name}")
        except:
            print(f"   https://github.com/YOUR_USERNAME/{self.repo_name}")
        
        print(f"\n🚀 **Next Steps:**")
        print(f"   1. Visit your repository on GitHub")
        print(f"   2. Configure repository settings")
        print(f"   3. Set up GitHub Actions (optional)")
        print(f"   4. Add collaborators (if needed)")
        print(f"   5. Start developing!")
    
    def run(self):
        """Execute the complete GitHub setup process"""
        print("🚀 EdunSight GitHub Repository Setup")
        print("="*50)
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                print("\n❌ Prerequisites not met. Please install required tools.")
                return False
            
            print(f"\n📁 Project Directory: {self.project_dir}")
            print(f"📝 Repository Name: {self.repo_name}")
            print(f"📋 Description: {self.description}")
            
            # Confirm setup
            response = input(f"\n🤔 Proceed with GitHub repository setup? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Setup cancelled by user.")
                return False
            
            print(f"\n🎬 Starting repository setup...")
            
            # Setup Git repository
            self.setup_git_repo()
            
            # Create GitHub repository
            self.create_github_repo()
            
            # Push to GitHub
            success, repo_url = self.push_to_github()
            
            if success:
                # Update README with badges
                self.create_readme_badge(repo_url)
                
                # Generate final report
                self.generate_report()
                
                print(f"\n🎉 SUCCESS! EdunSight repository is now on GitHub!")
                return True
            else:
                print(f"\n❌ Failed to complete repository setup.")
                return False
                
        except KeyboardInterrupt:
            print(f"\n⚠️ Setup interrupted by user.")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            self.failed_steps.append(f"Unexpected error: {e}")
            return False


def main():
    """Main execution function"""
    
    # Customize repository details here
    repo_name = "EdunSight"
    description = "🎓 Student Performance Prediction ML Pipeline with MVVM Architecture and Streamlit Web Interface"
    
    # Create and run setup
    setup = GitHubRepoSetup(repo_name, description)
    success = setup.run()
    
    if success:
        print(f"\n🎊 Repository setup completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 Repository setup failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

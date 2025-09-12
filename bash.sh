# 2. Git start karo
git init

# 3. .gitignore banao (jo extra files ko ignore karega)
cat <<EOL > .gitignore
__pycache__/
*.py[cod]
*.pyo
*.pyd
venv/
ENV/
env/
.venv/
.ipynb_checkpoints
*.log
.DS_Store
Thumbs.db
faiss_index/
.streamlit/
.vscode/
.idea/
EOL

# 4. Files add aur commit karo
git add .
git commit -m "Initial commit - RAG Chatbot Project"

# 5. Apne GitHub repo ka URL add karo
git branch -M main
git remote add origin https://github.com/ducklab268/raj_bot.git

# 6. Push karo GitHub pe
git push -u origin main


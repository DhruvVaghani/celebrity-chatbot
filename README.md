# 🎬 Celebrity Knowledge Chatbot

An interactive AI-powered chatbot built with OpenAI's GPT-4, LangChain, and Streamlit that allows users to search for celebrities and discover major global events around their birthdate.

---

## 🚀 Live Demo

👉 [Try the app here](https://celebrity-chatbot-8nqi9y5jqpb7n79nkz7qdv.streamlit.app/)

---

## 🧠 Project Overview

- **OpenAI GPT-4 Integration:** Utilizes powerful LLM capabilities to generate detailed celebrity profiles and historical event summaries based on user input.
- **LangChain Framework:** Implements prompt chaining and memory management to create context-aware, conversational flows.
- **Wikipedia Search Automation:** Dynamically retrieves suggestions and information for over 1 million+ Wikipedia articles.
- **Streamlit Frontend:** Provides a clean, real-time interactive web interface for users to easily query and explore.
- **Secure API Handling:** OpenAI API keys are managed securely using Streamlit Secrets.

---

## 🛠️ Tech Stack

- **Large Language Models (LLMs):** GPT-4 via OpenAI API
- **Framework:** LangChain
- **Frontend:** Streamlit
- **Data Source:** Wikipedia
- **Security:** Streamlit Secrets for API key management
- **Deployment:** Streamlit Community Cloud

---

## 📸 Screenshot

![image](https://github.com/user-attachments/assets/d4162d29-cc0c-4a76-bbb6-005d94a32619)

![image](https://github.com/user-attachments/assets/7a9f3bfa-603a-4f1d-b51d-69f1fbc46446)


---

## 📦 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/celebrity-chatbot.git
   cd celebrity-chatbot

2. **Create a virtual environment and activate it:**
python -m venv myenv
source myenv/bin/activate   # For Windows: myenv\Scripts\activate

3. **Install dependencies:**
pip install -r requirements.txt

4 **Set your OpenAI API key:**
OPENAI_API_KEY = "your-openai-api-key-here"

5 **Run the application:**
streamlit run main.py

🌟 Features
🔎 Celebrity search with autocomplete suggestions

🎬 Real-time YouTube video link generation for celebrity interviews

📅 Historical events around the celebrity’s birthdate

🧠 Context memory management using LangChain ConversationBuffer

🚀 Secure and scalable deployment on Streamlit Cloud

📚 Key Learnings
Integration of LLMs in real-world applications

Orchestration of multi-step reasoning workflows using LangChain

Prompt engineering for optimal AI responses

Handling API key security during public deployments

Building and deploying full-stack AI apps with minimal backend overhead

🤝 Contributing
Pull requests are welcome! Feel free to open an issue to discuss improvements or suggest new features.

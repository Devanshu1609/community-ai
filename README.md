# Mifos Community AI Chatbot

As the Mifos community around its core products (Mifos X, Payment Hub EE, Mobile Applications and Mifos Gazelle) grows rapidly, this chatbot aims to provide a quick self-service support methods for implementors, developers and new users. It aims to simplify finding relevant content across the multiple sources through Generative AI tools.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Run Locally (Jupyter Notebook)](#run-locally-jupyter-notebook)
  - [Use Deployed Chatbots (No-Setup Needed!)](#use-deployed-chatbots-no-setup-needed)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Features](#features)
- [Contributing](#contributing)
- [Links and Resources](#links-and-resources)

## Installation

To set up and run the Mifos Chatbot locally:

### 1. Clone the Repository:

```bash
git clone https://github.com/openMF/community-ai.git
cd Web-App
```

### 2. Install Dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have **Python 3.8+** installed.

### 3. Set up your API Key:

Create a `.env` file in the project root with the following:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```
**OR**
```plaintext
GEMINI_API_KEY=your-gemini-api-key
```
   

## Usage

You can run the Mifos Chatbot in two ways:

### Run Locally (Jupyter Notebook)

1. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

2. Open the `web-app_bot.ipynb` file.

3. Run the notebook cells one by one.

4. Interact with the chatbot via the Gradio interface:
   - Enter your query in the input box.
   - Click **Submit** or press **Enter**.
   - Get instant responses based on the Mifos/Fineract codebase and docs!

---

### Use Deployed Chatbots (No Setup Needed!)

Instantly access the bots hosted on Hugging Face Spaces:

- [Web App Bot](https://huggingface.co/spaces/MifosBot/Web-App)
- [Mifos Mobile Bot](https://huggingface.co/spaces/MifosBot/Mifos-Mobile)
- [Mobile Wallet Bot](https://huggingface.co/spaces/MifosBot/Mobile-Wallet)
- [Android Client Bot](https://huggingface.co/spaces/MifosBot/Android-Client)

No installation, just open the link and start asking questions! 

---

## Project Structure

Here's how the repository is organized:

| Path/Folder                        | Description |
| :---------------------------------- | :---------- |
| `web-app/`                          | Contains Mifos Web App files analyzed by the chatbot. |
| `web-app_bot.ipynb`                 | Jupyter Notebook to run the chatbot locally. |
| `web_app_vector_storage_metadata/`  | Stores vector embeddings for fast information retrieval. |
| `tools/translation-helper/`         | (New!) Gemini-powered Translation Helper Tool. |
| `requirements.txt`                  | Python dependencies needed for running the chatbot and translation tool. |
| `CodeCommentingScript.py`           | Script for pre-processing the codebase and generating vector embeddings. |

---

## How It Works

- The chatbot processes and indexes the Mifos codebase, creating vector embeddings for efficient retrieval.
- When a user asks a question, the system finds the most relevant code snippets or documentation.
- The chatbot then uses GPT models to generate a human-like response based on the retrieved information.
- Responses focus on file organization, key components, and project structure.
  
---

### Features

- Multilingual support for 35+ global languages  
- Formal/informal tone switch for contextual translations  
- Powered by Gemini 2.0 Flash via Google Generative AI  
- Trigger translation via Enter key or button  
- Built using Gradio for a minimal UI  

## Contributing

We welcome your ideas, bug reports, and contributions!
- Fork the repo
- Create a feature branch
- Submit a Pull Request

---

## Links and Resources

For more information about Mifos and Fineract, 
visit [Mifos](https://mifos.org/) or join our [AI working group](https://join.slack.com/share/enQtOTY2NTIzNDI0MjI5MC02YzhjMzc1NzhhMjM2OTZhNWQ4YWZkYWY0MWFmNTQxNmE4Yjg2ZDM4YWI4YzJmYzczYTQwNzk2NjAzNDgxMTc5)

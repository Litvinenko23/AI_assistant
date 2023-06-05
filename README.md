# AI_assistant
AI assistant with memory. The application allows the user to communicate with the GPT-3.5-turbo.  The application stores the previous conversation history in memory (Faiss), and when a new transformation is started, it retrieves previous conversations from memory and uses them in the tooltip to set the context.

### Start project
- docker pull andreyl23/ai_assistant
- docker run -p 80:8501 -e OPENAI_API_KEY="your openai_api_key" andreyl23/ai_assistant

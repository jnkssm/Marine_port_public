import os
from openai import OpenAI

# Set API key directly
api_key = "sk-fca2f07099954f289e5a3efd3a48e681"

# Initialize client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

print("Testing DeepSeek API connection...")

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Respond with exactly 'API_CONNECTION_SUCCESSFUL'"}
        ],
        temperature=0.1,
        max_tokens=50
    )
    
    result = response.choices[0].message.content.strip()
    print(f"Response: {result}")
    
    if "SUCCESSFUL" in result:
        print("✅ DeepSeek API is working correctly!")
    else:
        print(f"⚠️ Unexpected response: {result}")
        
except Exception as e:
    print(f"❌ Error: {e}")
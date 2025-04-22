import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Simple test to check if we can make a call to the OpenAI API
try:
    # Make a simple completion request
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello from OpenAI API' if you receive this message."}
        ]
    )
    
    # Extract and print the response
    assistant_message = response.choices[0].message.content
    print("API call successful!")
    print(f"Response: {assistant_message}")
    
except Exception as e:
    print(f"Error when calling OpenAI API: {e}")
    
print("\nAPI key information:")
# Don't print the actual key for security, just check if it exists
if openai.api_key:
    masked_key = openai.api_key[:4] + "..." + openai.api_key[-4:] if len(openai.api_key) > 8 else "***"
    print(f"API key found (masked): {masked_key}")
else:
    print("No API key found in environment variables!")
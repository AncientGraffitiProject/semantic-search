#!/usr/bin/env python3
"""
Simple test to verify Gemini integration is working
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

def test_gemini_connection():
    """Test basic Gemini connectivity"""
    try:
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test simple query
        response = model.generate_content("Hello! Can you help me rank Latin inscriptions?")
        print("✅ Gemini connection successful!")
        print(f"Response: {response.text[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Gemini Integration...")
    test_gemini_connection()

import asyncio
import re
from googletrans import Translator

# Enhanced translator that detects and handles both Latin and Greek
async def translate_text(phrase):
    async with Translator() as translator:
        # Check for Greek characters
        has_greek = any(ord(char) >= 0x0370 and ord(char) <= 0x03FF for char in phrase)
        
        # Set source language based on content
        if has_greek:
            src_lang = 'el'  # Greek
        else:
            src_lang = 'la'  # Latin
        
        try:
            result = await translator.translate(phrase, src=src_lang, dest='en')
            # Clean up the output
            output = re.sub(r'\s+', ' ', result.text).strip()
            print("    " + output)
        except Exception as e:
            # Fallback: try auto-detection if specific language fails
            try:
                result = await translator.translate(phrase, dest='en')
                output = re.sub(r'\s+', ' ', result.text).strip()
                print("    " + output)
            except:
                print("    Translation unavailable")


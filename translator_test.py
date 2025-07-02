import asyncio
import re
from googletrans import Translator

# Pulled the example from the googletrans documentation then changed it to fit my needs.
async def translate_text(phrase):
    async with Translator() as translator:
        result = await translator.translate(phrase, src='la', dest='en')
        # Clean up the output by removing excessive whitespace
        output = re.sub(r'\s+', ' ', result.text).strip()
        print("    " + output)


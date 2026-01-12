import os
import json
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

class Translator:
    def __init__(self):
        """
        Initializes the Google Gemini Translator.
        Requires GEMINI_API_KEY in .env file.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found. Please add it to your .env file.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def translate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Translates a list of dialogue segments from English to Hindi using Gemini.
        Also detects emotion (happy, sad, angry, fearful, neutral).
        Preserves speaker context and structure.
        """
        if not segments:
            return []

        print(f"Translating {len(segments)} segments with Gemini (Context + Emotion)...")
        
        # Prepare the prompt structure
        simplified_segments = []
        for seg in segments:
            simplified_segments.append({
                "id": seg.get("start"),
                "speaker": seg.get("speaker", 0),
                "text": seg.get("transcript", "")
            })

        prompt = f"""
        You are a professional dubbing translator.
        Translate the English segments to Hindi and detect the emotion.
        
        Rules:
        1. Return a JSON list of objects.
        2. Preserve 'id' and 'speaker' exactly.
        3. Translate 'text' to natural conversational Hindi.
        4. Add an 'emotion' field: "neutral", "happy", "sad", "angry", "fearful", "surprised".
        5. Use context to determine the best translation and emotion.

        Input:
        {json.dumps(simplified_segments, indent=2)}
        """

        try:
            response = self.model.generate_content(prompt)
            
            # Clean up response text (remove markdown code blocks if present)
            result_text = response.text.strip()
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            translated_data = json.loads(result_text)
            
            # Map back to original segments structure
            # Create a map of start_time -> {text, emotion}
            translation_map = {item["id"]: {"text": item["text"], "emotion": item.get("emotion", "neutral")} for item in translated_data}
            
            final_segments = []
            for seg in segments:
                original_id = seg.get("start")
                new_seg = seg.copy()
                if original_id in translation_map:
                    new_seg["transcript"] = translation_map[original_id]["text"]
                    new_seg["emotion"] = translation_map[original_id]["emotion"]
                else:
                    print(f"⚠️ Warning: Missing translation for segment starting at {original_id}")
                final_segments.append(new_seg)
                
            return final_segments

        except Exception as e:
            print(f"❌ Gemini Translation Failed: {e}")
            # Fallback: Return original segments if failure
            print("Fallback: Returning original English segments.")
            return segments

    def translate(self, text: str) -> str:
        """Single text translation (Legacy support)"""
        if not text:
            return ""
        try:
            response = self.model.generate_content(f"Translate this to Hindi: {text}")
            return response.text.strip()
        except:
            return text

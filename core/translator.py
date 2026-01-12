import os
import json
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

# Supported languages for dubbing
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
}

class Translator:
    def __init__(self, target_language: str = "hi"):
        """
        Initializes the Google Gemini Translator.
        
        Args:
            target_language: Language code (hi, ta, te, kn, ml, etc.)
                           Defaults to Hindi for backward compatibility.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found. Please add it to your .env file.")
        
        if target_language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {target_language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
        
        self.target_language = target_language
        self.language_name = SUPPORTED_LANGUAGES[target_language]
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        print(f"🌐 Translator initialized for: {self.language_name} ({target_language})")

    def translate_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Translates dialogue segments from English to target language using Gemini.
        Passes full timestamp, duration, and speaker context to help LLM:
        - Pick appropriate words that fit within the time window
        - Understand conversation flow and speaker context
        - Detect emotion for TTS voice modulation
        """
        if not segments:
            return []

        print(f"Translating {len(segments)} segments to {self.language_name} with Gemini...")
        print(f"  → Passing timestamps and speaker info for context")
        
        # Prepare detailed segment info with duration for duration-aware translation
        detailed_segments = []
        for seg in segments:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            duration = end - start
            # Estimate max words: ~2.5 words/second is typical for Indian languages
            max_words = max(3, int(duration * 2.5))
            
            # Structure with dialogue text first for better LLM context
            detailed_segments.append({
                "id": start,
                "english_dialogue": seg.get("transcript", ""),  # Original dialogue - shown first for context
                "speaker": seg.get("speaker", 0),
                "timestamp": f"{round(start, 2)}s - {round(end, 2)}s",  # Human-readable timestamp
                "duration_sec": round(duration, 2),
                "max_words_allowed": max_words,
            })

        prompt = f"""You are a professional dubbing translator for video/film content.
Translate the English dialogues to {self.language_name} and detect emotion.

CRITICAL RULES FOR DUBBING:
1. **DURATION CONSTRAINT**: Each segment has 'duration_sec' and 'max_words_allowed'.
   - Your translation MUST fit within the 'max_words_allowed' limit.
   - This prevents dialogue overlap in the final video.
   - Prefer shorter synonyms and natural contractions.
   - If needed, summarize while preserving core meaning.

2. **DIALOGUE CONTEXT**: 
   - 'english_dialogue' shows what was spoken at that timestamp.
   - 'timestamp' shows when the dialogue occurs (e.g., "2.5s - 5.0s").
   - Use this context to understand conversation flow and pacing.

3. **SPEAKER CONTEXT**:
   - 'speaker' identifies different speakers (0, 1, 2, etc.)
   - Maintain consistent voice/style for each speaker throughout.
   - Use appropriate formality based on speaker relationships.

4. **OUTPUT FORMAT**:
   - Return a JSON list of objects.
   - Preserve 'id' and 'speaker' exactly as given.
   - Add 'text' field with your {self.language_name} translation.
   - Add 'emotion' field: "neutral", "happy", "sad", "angry", "fearful", "surprised".

5. **QUALITY**:
   - Use natural, conversational {self.language_name} (not formal/bookish).
   - Preserve the tone, intent, and emotion of the original dialogue.
   - Make it sound like native {self.language_name} speakers would say it.

Input Segments (with English dialogue and timing):
{json.dumps(detailed_segments, indent=2, ensure_ascii=False)}

Return ONLY the JSON array, no other text."""

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
            
            translated_data = json.loads(result_text.strip())
            
            # Map back to original segments structure
            translation_map = {
                item["id"]: {
                    "text": item["text"], 
                    "emotion": item.get("emotion", "neutral")
                } 
                for item in translated_data
            }
            
            final_segments = []
            for seg in segments:
                original_id = seg.get("start")
                new_seg = seg.copy()
                if original_id in translation_map:
                    new_seg["transcript"] = translation_map[original_id]["text"]
                    new_seg["emotion"] = translation_map[original_id]["emotion"]
                    print(f"  ✅ [{original_id:.1f}s] Speaker {seg.get('speaker', 0)}: {new_seg['transcript'][:40]}...")
                else:
                    print(f"  ⚠️ Missing translation for segment at {original_id}s")
                final_segments.append(new_seg)
                
            print(f"✅ Translation complete: {len(final_segments)} segments in {self.language_name}")
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
            response = self.model.generate_content(
                f"Translate this to {self.language_name} (natural, conversational): {text}"
            )
            return response.text.strip()
        except:
            return text

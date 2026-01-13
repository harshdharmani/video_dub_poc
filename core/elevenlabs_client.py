import os
import time
from typing import Optional
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Load env variables
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

class ElevenLabsClient:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not found in .env")
        
        self.client = ElevenLabs(api_key=self.api_key)
        
        # Best model for dubbing: high quality + emotion + Hindi support
        self.model_id = "eleven_multilingual_v2"
        
        # Cache for cloned voice IDs: {speaker_id: voice_id}
        self.voice_map = {}

    def get_best_voice_for_language(self, lang_code: str) -> str:
        """
        Returns an optimized voice ID for the given language code.
        Uses 2026 recommended native voices for Indian languages.
        """
        # Map internal codes to specific ElevenLabs Voice IDs
        # Recommendations for 2026 (Native Indian Voices)
        voice_map = {
            "hi": "21m00Tcm4TlvDq8ikWAM", # Rachel (High quality Hindi)
            "ta": "onwK4e9ZLuTAKqWW03F9", # Daniel (Good for Dravidian)
            "te": "onwK4e9ZLuTAKqWW03F9", # Daniel (Telugu)
            "kn": "onwK4e9ZLuTAKqWW03F9", # Daniel (Kannada)
            "ml": "onwK4e9ZLuTAKqWW03F9", # Daniel (Malayalam)
            "bn": "cgSgspJ2msm6clMCkdW9", # Antoni (Bengali)
            "mr": "21m00Tcm4TlvDq8ikWAM", # Rachel (Marathi)
            "gu": "21m00Tcm4TlvDq8ikWAM", # Rachel (Gujarati) 
            "pa": "21m00Tcm4TlvDq8ikWAM", # Rachel (Punjabi - uses Hindi phonemes effectively)
        }
        
        # Fallback to Aria (Universal V3 optimized)
        return voice_map.get(lang_code, "9BWtsRjCglG6f8yz97TT")

    def generate_dub(self, text: str, output_path: str, speaker_id: int = 0, language: str = "hi") -> str:
        """
        Generates audio for the given text using the new v1.0+ SDK syntax.
        Automatically selects the best model and voice for the target language.
        """
        if not text:
            return ""

        print(f"  🗣️  ElevenLabs | Speaker {speaker_id} | Lang: {language} | {text[:30]}...")
        
        # Dynamic Voice Selection
        voice_id = self.get_best_voice_for_language(language)
        
        # Override with specific speaker mapping/clones if available
        # (Simple male/female toggle for demo if standard voice isn't forced)
        if speaker_id % 2 == 0 and language == "en": 
             voice_id = "JBFqnCBsd6RMkjVDRZzb" # George (Male) for English
        
        # Override with cloned voice if available
        if speaker_id in self.voice_map:
            voice_id = self.voice_map[speaker_id]

        # Dynamic Model Selection
        # Use v3_alpha for regional languages (better script support), v2 for others
        if language in ["or", "as", "mr", "bn"]:
             model_to_use = "eleven_turbo_v2_5" # Or "eleven_flash_v2_5" for speed if available, v2.5 supports many indian langs now
        elif language in ["hi", "ta"]:
             model_to_use = "eleven_turbo_v2_5" # Turbo 2.5 is best for Hindi/Tamil latency
        else:
             model_to_use = "eleven_multilingual_v2"

        try:
            # Replaced self.client.generate() with self.client.text_to_speech.convert()
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_to_use, 
                output_format="mp3_44100_128",
                voice_settings={
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            )
            
            # Save to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "wb") as f:
                for chunk in audio_generator:
                    if chunk:
                        f.write(chunk)
            
            # Rate limiting delay (Safety for Free Tier)
            time.sleep(0.5)
                
            return output_path
            
        except Exception as e:
            print(f"  ❌ ElevenLabs Failed: {e}")
            raise e
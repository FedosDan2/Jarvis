#!/usr/bin/env python3
import gc
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from scripts.integration_speeking.text_to_speech import TTSservice

import warnings
warnings.filterwarnings("ignore", message=".*quantized tensor creation.*")

DEFAULT_SYSTEM_PROMPT = (
    "Ты — голосовой эмпатичный ассистент Джарвис. "
    "Отвечай на русском языке. "
    "Не используй markdown, списки, эмодзи, звёздочки, решётки, скобки и технические символы. "
    "Пиши так, как будто текст будет произнесён вслух."
    "Старайся всегда закончить мысль. "
)


class JarvisLLMVoice:
    """
    Голосовой агент:
      user text -> LLM answer -> TTS wav bytes.

    По умолчанию:
      - LLM грузится лениво;
      - TTS грузится лениво;
      - после запроса обе модели можно выгрузить.
    """

    def __init__(
        self,
        llm_repo_id: str = "Qwen/Qwen2.5-3B-Instruct-GGUF",
        llm_filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf",
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        chat_format: str = "chatml",
        temperature: float = 0.5,
        max_tokens: int = 300,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tts_backend: str = "silero",
        tts_sample_rate: int = 48000,
        tts_speaker: str = "baya",
        tts_cache_dir: str = "models/tts_cache",
    ):
        self.llm_repo_id = llm_repo_id
        self.llm_filename = llm_filename
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        self.llm: Optional[Llama] = None
        self.llm_path: Optional[str] = None

        self.tts = TTSservice(
            backend=tts_backend,
            sample_rate=tts_sample_rate,
            speaker=tts_speaker,
            cache_dir=tts_cache_dir,
        )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    def _download_llm(self) -> str:
        if self.llm_path is not None and Path(self.llm_path).exists():
            return self.llm_path

        self.llm_path = hf_hub_download(
            repo_id=self.llm_repo_id,
            filename=self.llm_filename,
        )

        return self.llm_path

    def ensure_llm(self):
        if self.llm is not None:
            return

        model_path = self._download_llm()

        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            chat_format=self.chat_format,
            verbose=False,
        )

    def unload_llm(self):
        if self.llm is not None:
            try:
                self.llm.close()
            except Exception:
                pass

            self.llm = None

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    def unload_tts(self):
        self.tts.unload()

    def unload_all(self):
        self.unload_llm()
        self.unload_tts()

    # ------------------------------------------------------------------
    # Основной запрос
    # ------------------------------------------------------------------
    def ask(
        self,
        user_query: str,
        speak: bool = True,
        keep_llm: bool = False,
        keep_tts: bool = False,
    ) -> tuple[str, bytes]:
        """
        Возвращает:
            (answer_text, audio_wav_bytes)

        Если speak=False, audio_wav_bytes будет пустым.
        Если keep_llm=False, LLM выгружается после генерации текста.
        Если keep_tts=False, TTS выгружается после озвучки.
        """
        self.ensure_llm()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]

        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            repeat_penalty=1.15,
            top_k=40,
            top_p=0.9,
        )

        answer = output["choices"][0]["message"]["content"].strip()

        if not keep_llm:
            self.unload_llm()

        audio_bytes = b""

        if speak:
            audio_bytes = self.tts.speak(answer, apply_fx=True)

            if not keep_tts:
                self.unload_tts()

        return answer, audio_bytes


if __name__ == "__main__":
    agent = JarvisLLMVoice()

    query = "Привет. Расскажи мне о современных открытиях в сфере Искусственного Интелекта. "

    answer, audio = agent.ask(
        query,
        speak=True,
        keep_llm=False,
        keep_tts=False,
    )
    
    print(answer)

    if audio:
        out_dir = Path("out")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / "jarvis_reply.wav"
        out_file.write_bytes(audio)

        print(f"\n🔊 Аудио сохранено: {out_file}")
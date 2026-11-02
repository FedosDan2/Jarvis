#!/usr/bin/env python3
"""
Jarvis-агент на LangChain + LangGraph.

Архитектура:
  - LazyLlamaChatModel : кастомный BaseChatModel поверх llama-cpp-python.
                         Грузит веса ЛЕНИВО (при первом _generate) и умеет
                         выгружаться -> модели не висят в VRAM постоянно.
  - TTSservice         : твоя существующая озвучка (Silero), вызывается в узле.
  - StateGraph         : llm -> tts, с чекпоинтером = память диалога.

Ключевая идея: ИСТОРИЯ ДИАЛОГА живёт в чекпоинтере (CPU/RAM),
а ВЕСА МОДЕЛИ — только на время генерации. Поэтому unload() модели
НЕ теряет разговор. Это и есть load-on-demand без потери контекста.
"""
from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Annotated, Optional, TypedDict

import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from llama_cpp import Llama
from pydantic import PrivateAttr

# устойчивый импорт чекпоинтера: в новых langgraph он InMemorySaver,
# в старых — MemorySaver. Не падаем на дрейфе версии.
try:
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver as _Saver

from scripts.integration_speeking.text_to_speech import TTSservice

warnings.filterwarnings("ignore", message=".*quantized tensor creation.*")


DEFAULT_SYSTEM_PROMPT = (
    "Ты — голосовой эмпатичный ассистент. "
    "Отвечай на русском языке. "
    "Не используй markdown, списки, эмодзи, звёздочки, решётки, скобки и технические символы. "
    "Пиши так, как будто текст будет произнесён вслух. "
    "Старайся всегда закончить мысль. "
    "Отвечая на мои вопросы, начинай с обращения ко мне. "
)

# соответствие типов сообщений langchain -> роли llama.cpp
_ROLE = {
    SystemMessage: "system",
    HumanMessage: "user",
    AIMessage: "assistant",
}


def _msg_to_dict(m: BaseMessage) -> dict:
    return {"role": _ROLE.get(type(m), "user"), "content": m.content}


# ==========================================================================
# 1. Ленивая чат-модель поверх llama-cpp-python (это и есть LangChain-слой)
# ==========================================================================
class LazyLlamaChatModel(BaseChatModel):
    """
    Точный порт твоих параметров Llama(...) в интерфейс LangChain.
    Веса создаются только когда реально нужен _generate -> load-on-demand.
    """

    model_path: str
    n_ctx: int = 2048
    n_gpu_layers: int = -1
    chat_format: str = "chatml"
    temperature: float = 0.6
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.15
    max_tokens: int = 300
    verbose: bool = False

    # PrivateAttr НЕ валидируется pydantic-полем -> можно хранить тяжёлый объект
    _llm: Optional[Llama] = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "lazy-llama-cpp"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "chat_format": self.chat_format,
        }

    # ---------- жизненный цикл весов ----------
    def ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            chat_format=self.chat_format,
            verbose=self.verbose,
        )

    def unload(self) -> None:
        """Выгрузить веса. История диалога при этом НЕ теряется (она в чекпоинтере)."""
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception:
                pass
            self._llm = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ---------- генерация ----------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        self.ensure_loaded()

        dicts = [_msg_to_dict(m) for m in messages]
        out = self._llm.create_chat_completion(
            messages=dicts,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_k=kwargs.get("top_k", self.top_k),
            top_p=kwargs.get("top_p", self.top_p),
            repeat_penalty=kwargs.get("repeat_penalty", self.repeat_penalty),
            stop=stop,
        )

        text = out["choices"][0]["message"]["content"].strip()
        usage = out.get("usage")

        gen = ChatGeneration(
            message=AIMessage(content=text),
            generation_info={"raw": out},
        )
        return ChatResult(
            generations=[gen],
            llm_output={"usage": usage} if usage else None,
        )


# ==========================================================================
# 2. Состояние графа
# ==========================================================================
class AgentState(TypedDict, total=False):
    # add_messages = редьюсер: новые сообщения ДОБАВЛЯЮТСЯ к истории,
    # а не затирают её. Это и есть память диалога внутри state.
    messages: Annotated[list, add_messages]
    answer: str
    audio: bytes


# ==========================================================================
# 3. Узлы
# ==========================================================================
def make_llm_node(llm: LazyLlamaChatModel, system_prompt: str):
    def call_model(state: AgentState, config) -> AgentState:
        # системный промпт инжектим КАЖДЫЙ вызов, но НЕ кладём в state,
        # чтобы он не копился в чекпоинтере как отдельное сообщение истории.
        msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
        ai = llm.invoke(msgs)
        return {"messages": [ai], "answer": ai.content}

    return call_model


def make_tts_node(tts: TTSservice):
    def tts_step(state: AgentState, config) -> AgentState:
        audio = tts.speak(state.get("answer", ""), apply_fx=True)
        return {"audio": audio}

    return tts_step


# ==========================================================================
# 4. Сборка графа
# ==========================================================================
def build_graph(llm: LazyLlamaChatModel, tts: TTSservice, system_prompt: str):
    b = StateGraph(AgentState)
    b.add_node("llm", make_llm_node(llm, system_prompt))
    b.add_node("tts", make_tts_node(tts))
    b.add_edge(START, "llm")
    b.add_edge("llm", "tts")
    b.add_edge("tts", END)
    return b.compile(checkpointer=_Saver())


# ==========================================================================
# 5. Фасад-агент (заменяет твой JarvisLLMVoice, но с памятью и графом)
# ==========================================================================
class JarvisAgent:
    def __init__(
        self,
        llm_repo_id: str = "Qwen/Qwen2.5-3B-Instruct-GGUF",
        llm_filename: str = "qwen2.5-3b-instruct-q4_k_m.gguf",
        *,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        temperature: float = 0.95,
        max_tokens: int = 300,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tts_speaker: str = "xenia",
        tts_sample_rate: int = 48000,
        tts_cache_dir: str = "models/tts_cache",
        keep_model: bool = False,   # True = держать веса между запросами (быстро, но VRAM)
        keep_tts: bool = False,
    ):
        from huggingface_hub import hf_hub_download

        self.model_path = hf_hub_download(repo_id=llm_repo_id, filename=llm_filename)

        self.llm = LazyLlamaChatModel(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.tts = TTSservice(
            backend="silero",
            sample_rate=tts_sample_rate,
            speaker=tts_speaker,
            cache_dir=tts_cache_dir,
        )
        self.keep_model = keep_model
        self.keep_tts = keep_tts
        self.graph = build_graph(self.llm, self.tts, system_prompt)
        self.default_thread = "jarvis-main"

    def ask(self, user_query: str, thread_id: str | None = None) -> tuple[str, bytes]:
        """
        Один ход диалога. thread_id = идентификатор беседы.
        Новый thread_id => чистый разговор (это и есть «сброс памяти» в langgraph).
        """
        cfg = {"configurable": {"thread_id": thread_id or self.default_thread}}
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=cfg,
        )
        answer = result.get("answer", "")
        audio = result.get("audio", b"")

        if not self.keep_model:
            self.llm.unload()
        if not self.keep_tts:
            self.tts.unload()

        return answer, audio

    def history(self, thread_id: str | None = None) -> list[BaseMessage]:
        """Посмотреть накопленную историю диалога (живёт в чекпоинтере)."""
        cfg = {"configurable": {"thread_id": thread_id or self.default_thread}}
        snap = self.graph.get_state(cfg)
        return list(snap.values.get("messages", []))


if __name__ == "__main__":
    agent = JarvisAgent(keep_model=False, keep_tts=False)

    # Ход 1
    a1, _ = agent.ask(
        "Привет. Меня зовут Дарья. Запомни: я хочу писать дипломную работу по теории статистических решений с преподавателем Петром Меньшановым. "
    )
    print("\n[1] ", a1)

    a2, aud2 = agent.ask("Последнее время я слишком сильно загоняюсь о том, что Петр Меньшанов не возьмется курировать меня на дипл омную работу. Напиши, пожалуйста, мотивирующее, успокаивающее сообщение. И еще накидай мне коплиментов и скажи, что я молодец и у меня все получится.")
    print("\n[2] ", a2)
    print("\n[2] ", a2)

    if aud2:
        out = Path("out"); out.mkdir(exist_ok=True)
        (out / "jarvis_turn2.wav").write_bytes(aud2)
        print(f"\n🔊 {out/'jarvis_turn2.wav'}")

    print("\n--- история диалога ---")
    for m in agent.history():
        print(f"{type(m).__name__:12s}: {m.content[:80]}")
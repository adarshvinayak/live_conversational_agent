from __future__ import annotations

import logging
from dotenv import load_dotenv
import os
load_dotenv('.env.local')

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import openai
from livekit.plugins import tavus
from livekit.plugins import deepgram, cartesia, silero

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.DEBUG)

# Load API keys from environment
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

missing_keys = []
if not DEEPGRAM_API_KEY:
    missing_keys.append("DEEPGRAM_API_KEY")
if not CARTESIA_API_KEY:
    missing_keys.append("CARTESIA_API_KEY")
if not OPENAI_API_KEY:
    missing_keys.append("OPENAI_API_KEY")
if missing_keys:
    raise EnvironmentError(f"Missing required API keys: {', '.join(missing_keys)}. Please set them in your .env.local file.")

async def entrypoint(ctx: JobContext):
    try:
        logger.info(f"connecting to room {ctx.room.name}")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # --- Tavus AvatarSession Integration ---
        avatar = tavus.AvatarSession(
            replica_id="r6ae5b6efc9d",
            persona_id="pc55154f229a",
        )
        
        # Set up AgentSession with Deepgram STT, OpenAI LLM, Cartesia TTS, and VAD
        session = AgentSession(
            stt=deepgram.STT(
                model="nova-3",
                api_key=DEEPGRAM_API_KEY,
            ),
            llm=openai.LLM(api_key=OPENAI_API_KEY),
            tts=cartesia.TTS(
                model="sonic-2",
                api_key=CARTESIA_API_KEY,
                voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
            ),
            vad=silero.VAD.load(),
        )

        # Create a proper Agent instance
        agent = Agent(
            instructions="You are a helpful AI assistant. Respond naturally and conversationally to user questions.",
        )

        logger.info("starting avatar session")
        await avatar.start(session, room=ctx.room)
        
        logger.info("starting agent session")
        await session.start(agent=agent, room=ctx.room)
        
        logger.info("agent started successfully")
        
    except Exception as e:
        logger.error(f"Error in entrypoint: {e}")
        raise


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )

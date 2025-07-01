<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Python Multimodal Voice Agent

<p>
  <a href="https://cloud.livekit.io/projects/p_/sandbox"><strong>Deploy a sandbox app</strong></a>
  •
  <a href="https://docs.livekit.io/agents/overview/">LiveKit Agents Docs</a>
  •
  <a href="https://livekit.io/cloud">LiveKit Cloud</a>
  •
  <a href="https://blog.livekit.io/">Blog</a>
</p>

# LiveKit Agent with Tavus Avatar Integration

A voice AI agent built with LiveKit Agents framework, featuring:
- 🎤 **Speech-to-Text**: Deepgram Nova-3 model
- 🧠 **LLM**: OpenAI GPT models  
- 🔊 **Text-to-Speech**: Cartesia Sonic-2
- 👤 **Avatar**: Tavus realistic avatar
- 🎯 **Voice Activity Detection**: Silero VAD

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- API keys for required services

## Setup Instructions

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env.local` file in the project root and add your API keys:

```env
# Required API Keys
DEEPGRAM_API_KEY=your_deepgram_api_key_here
CARTESIA_API_KEY=your_cartesia_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LiveKit Server Configuration (for production)
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

### 3. Get API Keys

You'll need to sign up for these services and get API keys:

- **Deepgram**: [console.deepgram.com](https://console.deepgram.com/)
- **Cartesia**: [play.cartesia.ai](https://play.cartesia.ai/)
- **OpenAI**: [platform.openai.com](https://platform.openai.com/)

## Running the Agent

### Console Mode (Testing)
For local testing with microphone input:

```bash
python agent.py console
```

**Console Controls:**
- Press `Ctrl+B` to toggle between Text/Audio mode
- Press `Q` to quit
- Speak into your microphone to interact with the agent

### Development Mode
For development with hot reloading:

```bash
python agent.py dev
```

### Production Mode
For production deployment:

```bash
python agent.py start
```

## Agent Features

### Current Configuration
- **Replica ID**: `r6ae5b6efc9d` (Tavus avatar)
- **Persona ID**: `pc55154f229a` (Tavus persona)
- **Voice**: Cartesia Sonic-2 with voice ID `f786b574-daa5-4673-aa0c-cbe3e8534c02`
- **STT Model**: Deepgram Nova-3
- **LLM**: OpenAI GPT (default model)

### Customization
You can modify the agent behavior in `agent.py`:

- Change avatar/persona IDs in the Tavus configuration
- Update voice settings in Cartesia TTS
- Modify agent instructions
- Switch LLM models

## Frontend Integration

This agent can work with:
- LiveKit's [Agents Playground](https://agents.livekit.io/)
- Custom web/mobile frontends using LiveKit SDKs
- Telephony integration via LiveKit SIP

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're using LiveKit Agents 1.1.5+
2. **Missing API Keys**: Verify all required keys are set in `.env.local`
3. **Audio Issues**: Check microphone permissions and device selection
4. **Network Issues**: Ensure firewall allows WebRTC connections

### Debug Mode
The agent includes debug logging. Check console output for detailed information.

## Project Structure

```
lk-agent-avatar/
├── agent.py           # Main agent code
├── requirements.txt   # Python dependencies
├── .env.local        # API keys (create this)
├── README.md         # This file
└── venv/             # Virtual environment
```

## Documentation

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [LiveKit Python SDK](https://docs.livekit.io/realtime/client/python/)
- [Tavus API Documentation](https://docs.tavus.io/)

## Support

For issues and questions:
- [LiveKit Community Slack](https://livekit.io/join-slack)
- [GitHub Issues](https://github.com/livekit/agents/issues)

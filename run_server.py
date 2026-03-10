import os
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from app.main import app

NGROK_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")
if NGROK_TOKEN:
    ngrok.set_auth_token(NGROK_TOKEN)

ngrok_tunnel = ngrok.connect(8000)
print("Public URL:", ngrok_tunnel.public_url)
nest_asyncio.apply()

uvicorn.run(app, host="0.0.0.0", port=8000)
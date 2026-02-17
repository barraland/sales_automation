   
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
import requests
import uvicorn

api = FastAPI()

VERIFY_TOKEN = "my_verify_token_123"
TOKEN = "EAAedUO2ZA8XQBQpTOAeZC9wZA8l1JPTzPsRTDqWcKJ07o2ku5WYaTqAhiNGeAgxId2jtNx5nRGJpXkGXUhNTbnXyunmDVviIop5jkEJfOGrZBNhHQC2dcWKTmC5F2ZBBoEXdjFfUxF8KfOVGXFDQzgtTXZBQkLz85T6ZAjmtZAIZCE5jvsbInLjajjSG60VqxwxmeCzZAvZA9voN99V880pboZAdBIZB1pCOGRimJEH2SRZB1Kpz60MmmaJAhC6S0lofxbwn3ZAFHlVdrfpjFMBEgpXN6wZD"  
PHONE_NUMBER_ID = "1043497188838302"

@api.get("/whatsapp")
async def verify(request: Request):
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(content=params.get("hub.challenge"))
    return Response(content="Forbidden", status_code=403)

@api.post("/whatsapp")
async def webhook(request: Request):
    data = await request.json()
    print("Payload ricevuto:", data)

    # Estrazione messaggio
    try:
        if "messages" in data["entry"][0]["changes"][0]["value"]:
            msg = data["entry"][0]["changes"][0]["value"]["messages"][0]
            sender = msg["from"]
            if msg["type"] == "text":
                text = msg["text"]["body"]
                print(f"Messaggio da {sender}: {text}")
                # Qui chiamerai LangGraph! Per ora testiamo il ritorno:
                send_message(sender, f"Ricevuto su WSL2: {text}")
    except:
        pass

    return {"status": "ok"}

def send_message(to, text):
    url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    requests.post(url, json=payload, headers=headers)

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=9999)
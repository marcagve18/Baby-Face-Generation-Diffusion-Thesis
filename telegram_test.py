# telegram_lib.py
import urllib.parse
import requests
import json
import os

class TelegramBot:
    def __init__(self, bot_token, chat_id=None):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def get_updates(self):
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        return json.loads(requests.get(url).content)

    def send_message(self, message):
        encoded_message = urllib.parse.quote_plus(message)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage?chat_id={self.chat_id}&parse_mode=Markdown&text={encoded_message}"
        return requests.get(url).status_code == 200
    
token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID")
bot = TelegramBot(token, chat_id)

print(bot.send_message("Hi!"))
import telegram
import asyncio
from settings import TOKEN_BOT, CHAT_ID

async def send_telegram(photo_path="alert.png"):
    try:
        my_token = TOKEN_BOT
        bot = telegram.Bot(token=my_token)
        await bot.sendPhoto(chat_id=CHAT_ID, photo=open("alert.png", "rb"), caption="Có xâm nhập, nguy hiểm!")

    except Exception as ex:
        print("Can not send message telegram ", ex)

    print("Send success")

def flow_send_telegram():
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(send_telegram())
    except:
        pass
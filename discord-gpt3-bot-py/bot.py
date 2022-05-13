import discord
import os
from discord.ext import commands
import OpenAi
intents = discord.Intents().all()
client = commands.Bot(command_prefix="!")

BOT_TOKEN = os.getenv("BOT_TOKEN")

@client.event
async def on_ready():
    print("Bot is ready")

@client.command()
async def hello(ctx):
    await ctx.send("Hi")

responses = 0
list_user = []

@client.event
async def on_message(message):
    if message.channel.id == message.author.dm_channel.id: # dm only
        chat_log = ""
        list_user.append(message.author.id)
        inputs = message.content
        answer = OpenAi.ask(inputs)
        chat_log = OpenAi.append_interaction_to_chat_log(inputs, answer,chat_log)
        await message.channel.send(answer)
        f = open("log_user.txt", "w")
        for it in list_user:
            f.write("%i\n" % it)
        f.close()


@client.command()
@commands.is_owner()
async def shutdown(context):
    exit()


client.run(BOT_TOKEN)

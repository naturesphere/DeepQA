# -*- coding: utf-8 -*-
from chatterbot import ChatBot


# Uncomment the following lines to enable verbose logging
# import logging
# logging.basicConfig(level=logging.INFO)

# Create a new instance of a ChatBot
bot = ChatBot(
    "Terminal",
    storage_adapter="chatterbot.storage.MongoDatabaseAdapter",
    logic_adapters=[
        "chatterbot.logic.MathematicalEvaluation",
        #"chatterbot.logic.TimeLogicAdapter",
        "chatterbot.logic.BestMatch",
        {
            'import_path':'chatterbot.logic.LowConfidenceAdapter',
            'threshold':0.8,
            'default_response':'I am sorry, but I do not understand.'
        }
    ],
    filters=['chatterbot.filters.RepetitiveResponseFilter'],
    input_adapter="chatterbot.input.TerminalAdapter",
    output_adapter="chatterbot.output.TerminalAdapter",
    trainer='chatterbot.trainers.ListTrainer',
    database="chatterbot-database"
)
'''
bot.train([
    "Hi, can I help you?",
    "Sure, I'd to book a flight to Iceland.",
    "Your flight has been booked."
])
'''
'''
bot.train([
    "Show your ticket, please",
    "Here you are",
    "Ok, you are free to go!"
])
'''
bot.read_only=True

print("Type something to begin...")

# The following loop will execute each time the user enters input
while True:
    try:
        # We pass None to this method because the parameter
        # is not used by the TerminalAdapter
        bot_input = bot.get_response(None)

    # Press ctrl-c or ctrl-d on the keyboard to exit
    except (KeyboardInterrupt, EOFError, SystemExit):
        break

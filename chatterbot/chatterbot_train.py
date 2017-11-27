# -*- coding: utf-8 -*-
from chatterbot import ChatBot


# Uncomment the following lines to enable verbose logging
import logging
logging.basicConfig(level=logging.INFO)

# Create a new instance of a ChatBot

# langlist = ['chinese','english']
langlist = ['bangla','chinese','custom','english','french','german','hebrew','hindi','indonesia','italian',
            'marathi','portuguese','russian','spanish','swedish','tchinese','telugu','turkish']
 
Cbots = dict().fromkeys(langlist)
swflag = '>_<'

for lang in langlist:
    bot = ChatBot(
        lang,
        storage_adapter="chatterbot.storage.MongoDatabaseAdapter",
        logic_adapters=[
            "chatterbot.logic.MathematicalEvaluation",
            #"chatterbot.logic.TimeLogicAdapter",
            "chatterbot.logic.BestMatch",
            {
                'import_path':'chatterbot.logic.LowConfidenceAdapter',
                'threshold':0.6,
                'default_response':'I am sorry, but I do not understand.'
            }
        ],
        # filters=['chatterbot.filters.RepetitiveResponseFilter'],
        input_adapter="chatterbot.input.TerminalAdapter",
        output_adapter="chatterbot.output.TerminalAdapter",
        trainer='chatterbot.trainers.ChatterBotCorpusTrainer',
        database=lang+"_db"
    )
    bot.train("chatterbot.corpus."+lang)

#     bot.read_only=True

# print("Type something to begin...")

# # The following loop will execute each time the user enters input
# while True:
#     try:
#         # We pass None to this method because the parameter
#         # is not used by the TerminalAdapter
#         bot_input = bot.get_response(None)

#     # Press ctrl-c or ctrl-d on the keyboard to exit
#     except (KeyboardInterrupt, EOFError, SystemExit):
#         break

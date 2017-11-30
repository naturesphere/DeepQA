#!/usr/bin/python
# -*- coding:utf-8 -*-
# http://127.0.0.1:8000/CHATBOT?apikey=b1275afe-39f6-39c4-77b4-e5328dddba7&lang=en&kw=hi 



from tornado.ioloop import IOLoop
import tornado.web
from tornado.websocket import websocket_connect
import logging
import json
import time
import threading
from chatbot import chatbot 
import motor.motor_tornado
import time
import sys
import argparse
from chatterbot import ChatBot

logging.basicConfig(level=logging.INFO)

class WSClient(object):

    def __init__(self):
        self.msg = None
        self.conn = None

    def on_connected(self, f):
        try:
            self.conn = f.result()
        except Exception as e:
            print(str(e))
        
    def on_message(self, msgs):
        self.msg = msgs
        
    def sentMsg(self,keys,lang):
        print({"message":str(keys),"language":str(lang)})
        jsonStr = {"message":str(keys),"language":str(lang)}
        self.conn.write_message(json.dumps(jsonStr))
    
    def getMsg(self):
        return self.msg

class MongDBConnector():
    
    def __init__(self,server_site,server_port,database,colletion):
        self.server_site = server_site
        self.server_port = server_port
        try:
            self.colletion = MongoClient(server_site,server_port)[database][colletion]
        except Exception as msg:
            print(msg)
            self.colletion = None

    def insert_record(self, ip, question, answer):
        day = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        moment = time.strftime('%H:%M:%S',time.localtime(time.time()))
        record = {  'day':day,
                    'moment':moment,
                    'ip':ip,
                    'question':question,
                    'answer':answer
                }
        self.colletion.insert_one(record)

class MainHandler(tornado.web.RequestHandler):

    def retriveInput(self):
        reString = '{ "result": 0, "response": ":-)"}'

        global BOT, DB, Cbots, swflag, S2C
        try:
            inputKeys = self.get_argument('kw').strip()
            langKeys = self.get_argument('lang').strip()
            apiKeys = self.get_argument('apikey').strip()
            if apiKeys == "b1275afe-39f6-39c4-77b4-e5328dddba7" and inputKeys:
                Cbot = Cbots[S2C[langKeys]] # 
                response = swflag
                if Cbot != None:
                    response = Cbot.get_response(inputKeys)
                if response == swflag and langKeys=='en':
                    response = BOT.daemonPredict(inputKeys)
                
                # reString = '{ "result": 100, "response": \''+str(response)+'\'}'
                reString = '{"response":"'+str(response)+'"' + ',"id":88888888,"result":100,"msg":"OK."}'
                # reString = '{"response":"{}","id":88888888,"result":100,"msg":"OK."}'.format(str(response))
                self.write(reString)
                try:    # write to mongdb
                    # day = time.strftime('%Y-%m-%d',time.localtime(time.time()))
                    moment = time.strftime('%H:%M:%S',time.localtime(time.time()))
                    record = {  
                                'moment':moment,
                                'ip':self.request.remote_ip,
                                'question':inputKeys,
                                'answer':response
                            }
                    colletion = time.strftime('day_%Y%m%d',time.localtime(time.time())) 
                    DB[colletion].insert_one(record)
                except Exception as msg:
                    print(msg)
            else:
                self.write(reString)
        except Exception as e:
            self.write("Err: %s" % e)

    def get(self):   
       self.retriveInput()

    def post(self):
        self.retriveInput()

def cbots_init(swflag):
    langlist = ['bangla','chinese','custom','english','french','german','hebrew','hindi','indonesia','italian',
                'marathi','portuguese','russian','spanish','swedish','tchinese','telugu','turkish','testlang']
    # langlist = ['chinese','english']
    Cbots = dict().fromkeys(langlist)
    for lang in langlist:
        Cbots[lang] = ChatBot(
        lang,
        storage_adapter="chatterbot.storage.MongoDatabaseAdapter",
        logic_adapters=[
            "chatterbot.logic.MathematicalEvaluation",
            # "chatterbot.logic.TimeLogicAdapter",
            {
                "import_path":"chatterbot.logic.BestMatch",
                "response_selection_method":"chatterbot.response_selection.get_random_response"
            },
            {
                'import_path':'chatterbot.logic.LowConfidenceAdapter',
                'threshold':0.6,
                'default_response':swflag
            }
        ],
        # filters=['chatterbot.filters.RepetitiveResponseFilter'],
        # input_adapter="chatterbot.input.TerminalAdapter",
        # output_adapter="chatterbot.output.TerminalAdapter",
        trainer='chatterbot.trainers.ChatterBotCorpusTrainer',
        database = lang + "_db",
        # read_only=True,
        )
    return Cbots

def make_app(db):
    return tornado.web.Application([
        (r"/CHATBOT", MainHandler),
    ],db=db)

if __name__ == "__main__":
    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000, help='port')
    parser.add_argument('--modelTag', type=str, default='server', help='model suffix')
    args = parser.parse_args(sys.argv[1:])
    # wsc = WSClient()
    # websocket_connect('ws://192.168.1.109:1234/chat',ioloop,callback=wsc.on_connected,on_message_callback=wsc.on_message)
    # chat bot
    BOT = chatbot.Chatbot()
    BOT.main(['--modelTag', args.modelTag, '--test', 'daemon', '--rootDir','.'])
    # BOT2 = chatbot.Chatbot()
    # BOT2.main(['--modelTag', '1114', '--test', 'daemon', '--rootDir','.'])

    #chatterbot
    swflag = '>_<'
    S2C = { 'bn':'bangla',    
            'zh':'chinese',    
            'custom':'custom',    
            'en':'english',    
            'fr':'french',
            'de':'german',    
            'he':'hebrew',     
            'in':'hindi',         
            'id':'indonesia',   
            'it':'italian',
            'mr':'marathi',            
            'pt':'portuguese',
            'ru':'russian',            
            'es':'spanish',            
            'sv':'swedish',            
            'zh_TW':'tchinese',
            'te':'telugu',           
            'tr':'turkish'
        }
    Cbots = cbots_init(swflag)

    # mongdb client
    server_site = '127.0.0.1'
    server_port = 27017
    database = 'chatbotDB'
    DB = motor.motor_tornado.MotorClient(server_site,server_port)[database]

    #system arguments
    app = make_app(DB)
    app.listen(args.port)
    IOLoop.current().start()
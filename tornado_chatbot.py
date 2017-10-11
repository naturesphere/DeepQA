#!/usr/bin/python
# -*- coding:utf-8 -*-

import tornado.ioloop
import tornado.web
from tornado.websocket import websocket_connect
import logging
import json
import time
import threading
from chatbot import chatbot 

langs = ["bangla","chinese","french","german","hebrew","hindi","indonesia",
         "italian","marathi","portuguese","russian","spanish","tchinese","telugu","turkish"]

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


class MainHandler(tornado.web.RequestHandler):
    def get(self):   
        reString = '{ "result": 0, "response": ":-)"}'
#        global wsc
        global BOT
        try:
            inputKeys = self.get_argument('kw').strip()
            langKeys = self.get_argument('lang').strip()
            apiKeys = self.get_argument('apikey').strip()

            if apiKeys == "b1275afe-39f6-39c4-77b4-e5328dddba7" and inputKeys:
#                wsc.sentMsg(inputKeys, langKeys)
#                response = wsc.getMsg()
#                response = ChatbotManager.callBot(inputKeys)
                response = BOT.daemonPredict(inputKeys)
                reString = '{ "result": 100, "response": \''+str(response)+'\'}'
                self.write(reString)
            else:
                self.write(reString)
        except Exception as e:
            self.write("Err: %s" % e)
         

def make_app():
    return tornado.web.Application([
        (r"/CHATBOT", MainHandler),
    ])

if __name__ == "__main__":
    ioloop = tornado.ioloop.IOLoop.current()
#    wsc = WSClient()
#    websocket_connect('ws://192.168.1.109:1234/chat',ioloop,callback=wsc.on_connected,on_message_callback=wsc.on_message)
    BOT = chatbot.Chatbot()
    BOT.main(['--modelTag', 'server', '--test', 'daemon', '--rootDir','.'])
    app = make_app()
    app.listen(8891)
    ioloop.start()
    




    


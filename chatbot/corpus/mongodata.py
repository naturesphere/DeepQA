from pymongo import MongoClient
import pprint

# host = '192.168.1.253'
# port = 27017
# dbname = 'chatbotDB'
# clname = 'day_20171021'
# client = MongoClient(host, port)
# cln = client[dbname][clname]
# # 读取独立ip
# unique_ips = cln.distinct('ip',{})
# print("unique ip count:",len(unique_ips))
# conversations = []
# # 读取每个ip下的对话
# for ip in unique_ips:
#     linesBuffer = []
#     ctxs = cln.find({'ip':ip})
#     for ctx in ctxs:
#         linesBuffer.append({"text":ctx['answer']})
#         linesBuffer.append({"text":ctx['question']})
#         # pprint.pprint(ctx)
#     # print("===")
#     conversations.append({"lines":linesBuffer}) # 转化为conversation格式
# print(conversations)
# results = cln.find({'ip':'127.0.0.1',
#                     'moment':'15:27:23'})
# print(type(results))
# cnt = 0
# for r in results:
#     pprint.pprint(r)
#     cnt +=1

# print(len(results))

class MongoData:

    def __init__(self, collectionNames=[], databaseName='chatbotDB', host='127.0.0.1', port=27017):
        self.host = host
        self.port = port
        self.databaseName = databaseName
        self.collectionNames = collectionNames
        self.client = MongoClient(self.host, self.port)
        self.conversations = []
        try:
            self.loadDatabaseConversatins()
        except:
            self.conversations = []
            raise ValueError('can not connect to mongodb.')

    def loadDatabaseConversatins(self):
        if self.collectionNames==[]:# default [] means all collections
            self.collectionNames = self.client[self.databaseName].collection_names()

        for clt in self.collectionNames:
            self.conversations += self.loadCollectionConversations(clt)

        return self.conversations

    def loadCollectionConversations(self, collectionName):
        clt = self.client[self.databaseName][collectionName]
        # 读取独立ip
        unique_ips = clt.distinct('ip',{})
        # print("unique ip count:",len(unique_ips))
        conversations = []
        # 读取每个ip下的对话
        for ip in unique_ips:
            linesBuffer = []
            ctxs = clt.find({'ip':ip})
            for ctx in ctxs:
                linesBuffer.append({"text":ctx['question']})
                linesBuffer.append({"text":ctx['answer']})
                # pprint.pprint(ctx)
            # print("===")
            conversations.append({"lines":linesBuffer}) # 转化为conversation格式
        # print(conversations)    
        return conversations

    def getConversations(self):
        return self.conversations

if __name__=="__main__":
    print("in main")
    md = MongoData(host='192.168.1.253',collectionNames=[])
    cns = md.getConversations()
    print(cns)

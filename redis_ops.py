import redis
import pickle
import age_gender_service_settings as settings

class redisOperation():
    def __init__(self, host = settings.REDIS_HOST, port = settings.REDIS_PORT,
                 db = settings.REDIS_DB, password = ""):
        self.database = redis.Redis(
            host = host, port = port, db = db, password = password
        )
        print("Successfully connect to Redis Server.")

    def setData(self, key, value):
        self.database.set(key, pickle.dumps(value))

    def getData(self, key):
        data = self.database.get(key)
        if data is None:
            return None
        else:
            return pickle.loads(data)

    def getKeys(self):
        byteKeys = self.database.keys()

        rawKeys = []
        for key in byteKeys:
            rawKeys.append(key.decode())
        return rawKeys

if __name__ == "__main__":
    r = redisOperation()
    r.setData('test-for', [19,2,31,35,6])
    data = r.getData('test-fo')
    print(data)

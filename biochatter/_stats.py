# BioChatter usage statistics (for community key usage)
# keep persistent statistics about community key usage in redis
# adapted from https://github.com/mobarski/ask-my-pdf
import os
from time import strftime

import redis
from retry import retry

DEFAULT_USER = "community"


class Stats:
    def __init__(self):
        self.config = {}

    def render(self, key):
        variables = dict(
            date=strftime("%Y-%m-%d"),
            hour=strftime("%H"),
        )
        variables.update(self.config)
        for k, v in variables.items():
            key = key.replace("[" + k + "]", v)
        return key


class RedisStats(Stats):
    def __init__(self):
        REDIS_PW = os.getenv("REDIS_PW")
        if not REDIS_PW:
            raise Exception("No Redis password in environment variables!")
        self.db = redis.Redis(
            host="redis-10494.c250.eu-central-1-1.ec2.cloud.redislabs.com",
            port=10494,
            password=REDIS_PW,
        )
        self.config = {}

    @retry(tries=5, delay=0.1)
    def increment(self, key, kv_dict):
        # TODO: non critical code -> safe exceptions
        key = self.render(key)
        p = self.db.pipeline()
        for member, val in kv_dict.items():
            member = self.render(member)
            self.db.zincrby(key, val, member)
        p.execute()

    @retry(tries=5, delay=0.1)
    def get(self, key):
        # TODO: non critical code -> safe exceptions
        key = self.render(key)
        items = self.db.zscan_iter(key)
        return {k.decode("utf8"): v for k, v in items}


stats_data_dict = {}


def get_stats(**kw):
    stats = RedisStats()
    stats.config.update(kw)
    return stats


def get_community_usage_cost():
    usage_stats = get_stats(user=DEFAULT_USER)
    data = usage_stats.get(f"usage:[date]:{DEFAULT_USER}")
    used = 0.0
    used += 0.04 * data.get("total_tokens:gpt-4", 0) / 1000  # prompt_price=0.03 but output_price=0.06
    used += 0.02 * data.get("total_tokens:text-davinci-003", 0) / 1000
    used += 0.002 * data.get("total_tokens:text-curie-001", 0) / 1000
    used += 0.002 * data.get("total_tokens:gpt-3.5-turbo", 0) / 1000
    used += 0.0004 * data.get("total_tokens:text-embedding-ada-002", 0) / 1000
    return used


if __name__ == "__main__":
    s1 = get_stats(user="sebastian")
    s1.increment("aaa:[date]:[user]", dict(a=1, b=2))
    s1.increment("aaa:[date]:[user]", dict(a=1, b=2))
    print(s1.get("aaa:[date]:[user]"))
    #
    s2 = get_stats(user="kerbal")
    s2.increment("aaa:[date]:[user]", dict(a=1, b=2))
    s2.increment("aaa:[date]:[user]", dict(a=1, b=2))
    print(s2.get("aaa:[date]:[user]"))

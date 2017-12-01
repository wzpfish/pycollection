# coding:utf-8
"""
Elasticsearch convenient functions.
"""
from elasticsearch import Elasticsearch
import elasticsearch.helpers
import json


class EsHelper(object):
    def __init__(self, uri, user, password, timeout=30000):
        self._user = user
        self._password = password
        self._uri = uri
        self._timeout = timeout
        self._client = Elasticsearch(uri, http_auth=(user, password), timeout=timeout)

    def reset(self, uri, user, password, timeout):
        """reset client.
        """
        if user:
            self._user = user
        if password:
            self._password = password
        if uri:
            self._uri = uri
        self._client = Elasticsearch(self._uri, http_auth=(self._user, self._password), timeout=timeout)

    def scan(self, index, query, need_id=False):
        """Crawl fresh data.
        """
        for e in elasticsearch.helpers.scan(self._client, index=index, query=query, size=100, request_timeout=30000):
            if not need_id:
                yield e['_source']
            else:
                yield e["_id"], e["_source"]

    @staticmethod
    def esclient(user, password, uri, timeout=30000):
        """Get the raw ES client.
        """
        return Elasticsearch(uri, http_auth=(user, password), timeout=timeout)

    def backup(self, src_index, des_index, des_client, _type):
        """Backup
        """
        def copy_settings(src_client, src_index):
            valids = {"number_of_shards", "number_of_replicas"}
            settings = src_client.indices.get_settings(index=src_index)[src_index]['settings']['index']
            settings_new = {}
            for i in valids:
                if i in settings:
                    settings_new[i] = settings[i]
            return settings_new

        def copy_mappings(src_client, src_index):
            return src_client.indices.get_mapping(index=src_index)[src_index]['mappings']

        des_client.indices.delete(des_index)
        r = des_client.indices.create(index=des_index,
                                      body={'settings': copy_settings(self._client, src_index),
                                            "mappings": copy_mappings(self._client, src_index)})
        
        # Reindex 
        body = {"query": {"term": {"_type": _type}}}
        elasticsearch.helpers.reindex(client=self._client,
                                      source_index=src_index,
                                      target_index=des_index,
                                      target_client=des_client,
                                      query=body,
                                      chunk_size=1024,
                                      scroll='5m')

    def search(self, index, query):
        """Search by a query in index.
        """
        return [doc['_source'] for doc in self._client.search(index=index, body=query)['hits']['hits']]

    def batch_upsert(self, index, _type, ids, docs):
        """return (success number, list of errors)
        """
        actions = [{"_source": doc, "_id": _id, "_type": _type, "_index": index}
                for _id, doc in zip(ids, docs)]
        return elasticsearch.helpers.bulk(
                self._client, actions, raise_on_error=False)

    def batch_update(self, index, _type, ids, docs):
        assert len(ids) == len(docs)
        actions = []
        for _id, doc in zip(ids, docs):
            action = {
                '_op_type': 'update',
                '_index': index,
                '_type': _type,
                '_id': _id,
                'doc': doc,
                # If not exists, it will insert a new doc.
                "doc_as_upsert": True
            }
            actions.append(action)
        print(elasticsearch.helpers.bulk(self._client, actions))

    def build_index(self, index, mappings):
        """Build the index given the index schema(mappings).
        """
        self._client.indices.delete(index)
        self._client.indices.create(index)
        for k in mappings:
            self._client.indices.put_mapping(index=index, doc_type=k, body=mappings[k])

    def put_mappings(self, index, doc_type, body):
        """
        """
        self._client.indices.put_mapping(index=index, doc_type=doc_type, body=body)


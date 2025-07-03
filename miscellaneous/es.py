# from elasticsearch import Elasticsearch
#
# # Connect to local ES
# es = Elasticsearch("http://localhost:9200")
#
# # Check connection
# if es.ping():
#     print("Connected to Elasticsearch!")
# else:
#     print("Connection failed.")

from elasticsearch import Elasticsearch

client = Elasticsearch(
    "https://my-elasticsearch-project-c5dc7b.es.us-central1.gcp.elastic.cloud:443",
    api_key="Zm5fdE81Y0IxajJydThhel91a3U6TVJiSWp6THR0MlNjV0tPQUJnWDZWQQ=="
)

info = client.info()
print(info)

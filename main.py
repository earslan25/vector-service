import grpc
from concurrent import futures
import logging

import torch

import vector_service_pb2_grpc
import vector_service_pb2

from datasets import load_dataset

from text_transformer.text_transformer import TextTransformer, clean_text

# Initialize text transformer
transformer = TextTransformer("justin871030/bert-base-uncased-goemotions-original-finetuned")


# VectorServiceServicer provides an implementation of the methods of the Vector service.
class VectorServiceServicer(vector_service_pb2_grpc.VectorServiceServicer):
    def GetEmbedding(self, request, context):
        """
        Retrieves the embedding for a given input.
        """
        print("Received request:", request)
        cleaned_input = clean_text(request.input_text)
        print("Cleaned input:", cleaned_input)
        embedding = transformer.embed_text([cleaned_input]).squeeze(0)
        response = vector_service_pb2.GetEmbeddingResponse()
        response.resulting_embedding.values.extend(embedding)

        return response

    def GetSimilarity(self, request, context):
        """
        Computes and returns queries that are similar to the input.
        """
        print("Received request:", request.input_embedding.values[:5], "...")
        assert transformer.embeddings.shape[0] > 0, "No embeddings loaded yet!"
        input_embedding = torch.Tensor(request.input_embedding.values)
        matches, scores = transformer.compare_similarity(input_embedding)
        response = vector_service_pb2.GetSimilarityResponse()
        response.matches.extend(matches)
        response.scores.extend(scores)

        return response

    def LoadQueries(self, request, context):
        """
        Loads the queries into the text transformer.
        """
        print("Received request:", request)
        assert len(request.queries) == len(request.thresholds), "Queries and thresholds must be the same length!"
        transformer.load_queries(request.queries, request.thresholds)
        response = vector_service_pb2.LoadQueryResponse()

        return response


def serve():
    # Start server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vector_service_pb2_grpc.add_VectorServiceServicer_to_server(
        VectorServiceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Listening on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()

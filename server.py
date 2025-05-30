import grpc
import logging
from concurrent import futures
from io import BytesIO

from torchvision.utils import save_image
from PIL import Image

from grpc_interface import inference_pb2
from grpc_interface import inference_pb2_grpc
from hair_swap import HairFast, get_parser


def bytes_to_image(image: bytes) -> Image.Image:
    image = Image.open(BytesIO(image))
    return image


class SwapServer(inference_pb2_grpc.HairSwapServiceServicer):
    def __init__(self):
        self.hair_fast = HairFast(get_parser().parse_args([]))

    def swap(self, request, context):
        # Load the data
        face = bytes_to_image(request.face)
        if request.shape == b'face':
            shape = face
        else:
            shape = bytes_to_image(request.shape)
        if request.color == b'shape':
            color = shape
        else:
            color = bytes_to_image(request.color)

        # Create image
        final_image = self.hair_fast.swap(face, shape, color)

        # Convert to png
        buffer = BytesIO()
        save_image(final_image, buffer, 'png')
        buffer.seek(0)

        return inference_pb2.HairSwapResponse(image=buffer.read())


def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    inference_pb2_grpc.add_HairSwapServiceServicer_to_server(SwapServer(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()

    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
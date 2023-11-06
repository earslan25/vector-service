# Vector Service

This service offers a straightforward implementation for generating vector embeddings and computing similarities between vectors. The functionality is primarily defined using Protocol Buffers (protobuf), making the service interoperable across various languages and platforms.

## Project Structure

Here's a brief overview of the project's directory structure:

- **protos/**: Contains the `.proto` definition files. These define the message types and service definitions.
  - `vector_models.proto`: Contains data type definitions
  - `vector_service.proto`: Contains the service + corresponding request/response messages
  
- **generate_protos.sh**: A shell script to regenerate Python code from `.proto` files whenever they are modified.

- **main.py**: The main entry point of the service.
- **vector_service.py**: Implements the RPC handlers for VectorService.

## Working with Protocol Buffers

Protocol Buffers (protobuf) is a language-neutral, platform-neutral extensible mechanism for serializing structured data. This project uses protobuf to define message formats and service contracts, ensuring a consistent and type-safe way of communication.

If you make any changes to the `.proto` files in the `protos/` directory, you must regenerate the corresponding Python code. 

### Regenerating Python Code for Protobuf

To regenerate the Python code from the `.proto` files, you can use the provided `generate_protos.sh` script:

```bash
./generate_protos.sh
```

Please ensure that you have `protoc`, the Protocol Buffers compiler, installed and that the `grpcio-tools` Python package is available.

## Getting Started

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Ensure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```

3. If you've made changes to the `.proto` files, remember to regenerate the Python code:

```bash
./generate_protos.sh
```

4. Run the application:

```bash
python main.py
```
Credits: Proto structure and proto generation script by @nthnluu as part of Brown University research.

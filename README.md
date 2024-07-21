# llamaindexQ
llamaindex Query

## Overview
llamaindexQ is a project designed to facilitate querying using the llama-index library. It leverages OpenAI's API to create and query a vector store index from documents stored in a directory.

## Features
- Load documents from a directory and create a vector store index.
- Persist the index for future use.
- Query the index using a simple query engine.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llamaindexQ.git
   cd llamaindexQ
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Set up your OpenAI API key:
   ```python
   cp config.json.example config.json
   ```

2. Run the script to create or load the index and perform a query:
   ```bash
   python start.py
   ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

## Contact
For any inquiries, please contact Jianqing Peng 
# Book Recommender

## Project Description
The Book Recommender is a web application that provides book recommendations based on user input. It uses machine learning models to analyze book descriptions and categorize them into different genres and emotional tones. The application is built using Python, Gradio for the web interface, and various machine learning libraries.

## Setup Instructions

### Prerequisites
- Python 3.9 or later
- Docker
- Google Cloud SDK

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/book-recommender.git
   cd book-recommender
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

4. Build and run the Docker container:
   ```bash
   docker build -t book-recommender .
   docker run -p 7860:7860 book-recommender
   ```

## Usage
- Access the application in your web browser at `http://localhost:7860`.
- Enter a book description, select a category and tone, and click "Find recommendations" to get book suggestions.

## Contribution Guidelines
- Fork the repository and create a new branch for your feature or bug fix.
- Submit a pull request with a detailed description of your changes.
- Ensure your code follows the project's coding standards and passes all tests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions or support, please contact [gyulim0606@gmail.com]. 

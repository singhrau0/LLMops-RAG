from dotenv import load_dotenv  # Load variables from the .env file before the app uses OpenAI.
from flask import Flask, render_template, request  # Import Flask tools for routes, templates, and form data.

from src.rag_pipeline.config import ARTIFACT_INDEX, CHUNKS_FILE  # Import artifact paths used by the web app.
from src.rag_pipeline.rag_chain import answer_question  # Import the LangChain RAG function.

load_dotenv(".env")  # Read OPENAI_API_KEY from the real .env file into the program environment.

app = Flask(__name__)  # Create the Flask web application object.

@app.route("/", methods=["GET", "POST"])  # Create one browser page that accepts GET and POST requests.
def home():  # Define the function that controls the home page.
    question = request.form.get("question", "")  # Read the student's question from the HTML form.
    result = None  # Start with no result before the student asks anything.
    if question:  # Run the RAG pipeline only when a question exists.
        result = answer_question(question)  # Retrieve context and ask GPT-4o-mini using LangChain.
    return render_template(  # Send data from Python into the HTML template.
        "index.html",  # Tell Flask which template file to render.
        question=question,  # Pass the question back so it stays visible in the input box.
        result=result,  # Pass the answer, sources, and pipeline steps to the frontend.
        chunks_file=CHUNKS_FILE,  # Show students where chunk artifacts are stored.
        index_file=ARTIFACT_INDEX,  # Show students where vector index artifacts are stored.
    )  # Finish rendering the page.

if __name__ == "__main__":  # Run this block only when we execute python app.py directly.
    app.run(debug=True)  # Start the local Flask development server with auto-reload enabled.

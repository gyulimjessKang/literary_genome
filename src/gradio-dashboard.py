import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("src/data/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "src/image_files/BookCoverNotFound.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("src/data/tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)

    if recommendations.empty:
        return "<p>No recommendations found for your query.</p>"

    # --- 1. CSS for the Modal ---
    # This styles the pop-up window and the overlay behind it.
    modal_css = """
    <style>
        .modal {
            display: none; position: fixed; z-index: 1000; left: 0; top: 0;
            width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.6);
        }
        .modal-content {
            background-color: #fefefe; margin: 10% auto; padding: 25px;
            border: 1px solid #888; width: 60%; border-radius: 10px;
            position: relative;
        }
        /* Add more specific rules to override the theme's text color */
        .modal-content h2, .modal-content h4, .modal-content p {
            color: black !important;
        }
        .close-btn {
            color: #aaa; float: right; font-size: 28px; font-weight: bold;
            position: absolute; top: 10px; right: 20px;
        }
        .close-btn:hover, .close-btn:focus {
            color: black; text-decoration: none; cursor: pointer;
        }
    </style>
    """

    # --- 2. JavaScript for Modal Control ---
    # These functions will show and hide the modals.
    modal_js = """
    <script>
        function showModal(modalId) {
            document.getElementById(modalId).style.display = 'block';
        }
        function closeModal(event, modalId) {
            event.stopPropagation();
            document.getElementById(modalId).style.display = 'none';
        }
    </script>
    """

    # --- 3. HTML for the Book Gallery ---
    gallery_html = "<div style='display: grid; grid-template-columns: repeat(8, 1fr); gap: 20px;'>"
    modals_html = ""

    for _, row in recommendations.iterrows():
        isbn = row['isbn13']
        modal_id = f"modal-{isbn}"
        
        authors = row["authors"]
        if pd.isna(authors):
            authors_str = "Unknown Author"
        else:
            authors_split = str(authors).split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = authors
        
        # Add a clickable book item to the gallery
        gallery_html += f"""
        <div onclick="showModal('{modal_id}')" style="text-align: center; cursor: pointer;">
            <img src="{row['large_thumbnail']}" style="width: 100%; height: 220px; object-fit: cover; border-radius: 5px;">
            <p style="font-size: 12px; margin-top: 8px; font-weight: bold;">{row['title']}</p>
            <p style="font-size: 11px; color: #555;">by {authors_str}</p>
        </div>
        """

        # Create the corresponding hidden modal for this book
        modals_html += f"""
        <div id="{modal_id}" class="modal" onclick="closeModal(event, '{modal_id}')">
            <div class="modal-content">
                <span class="close-btn" onclick="closeModal(event, '{modal_id}')">&times;</span>
                <h2>{row['title']}</h2>
                <h4>by {authors_str}</h4>
                <p>{row['description']}</p>
            </div>
        </div>
        """

    gallery_html += "</div>"

    # --- 4. Combine all parts and return ---
    return modal_css + gallery_html + modals_html


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# JavaScript functions to be injected into the app's header
modal_js = """
<script>
function showModal(modalId) {
    document.getElementById(modalId).style.display = 'block';
}
function closeModal(event, modalId) {
    // Stop the click from propagating to the background div
    event.stopPropagation();
    document.getElementById(modalId).style.display = 'none';
}
</script>
"""

with gr.Blocks(theme = gr.themes.Glass(), head=modal_js) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    # Use a single HTML component to render the gallery manually
    output_html = gr.HTML(value="<p style='color: #888;'>Your recommendations will appear here.</p>")

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output_html
    )


if __name__ == "__main__":
    # Add allowed_paths to give the frontend permission to load local images
    dashboard.launch(share=True, debug=True, allowed_paths=["src/image_files"])
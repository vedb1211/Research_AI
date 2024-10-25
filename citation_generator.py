import streamlit as st
import requests
import xmltodict
from datetime import datetime
import io

# Function to fetch paper details from arXiv
def get_arxiv_data(arxiv_url: str):
    """Fetch data from arXiv using the arXiv API."""
    # Extract arXiv ID from the URL
    arxiv_id = arxiv_url.split('/')[-1]
    
    # Construct the API URL
    api_url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    
    # Make a request to the arXiv API
    response = requests.get(api_url)
    
    # Parse the response as XML
    if response.status_code == 200:
        data = xmltodict.parse(response.content)
        entry = data['feed']['entry']
        return entry
    else:
        return None

# Function to format the citation in APA style
def format_apa_citation(paper_data):
    """Format the arXiv paper data into an APA citation."""
    title = paper_data.get('title', 'No title available').replace("\n", " ")
    authors = paper_data.get('author', [])
    published_date = paper_data.get('published', 'No date available')
    year = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ").year

    if isinstance(authors, list):
        author_names = [f"{author['name'].split()[-1]}, {author['name'][0]}." for author in authors]
        if len(authors) == 1:
            authors_str = author_names[0]
        elif len(authors) == 2:
            authors_str = f"{author_names[0]} & {author_names[1]}"
        else:
            authors_str = ", ".join(author_names[:-1]) + f", & {author_names[-1]}"
    else:
        authors_str = f"{authors['name'].split()[-1]}, {authors['name'][0]}."

    citation = f"{authors_str} ({year}). {title}. arXiv. https://arxiv.org/abs/{paper_data.get('id').split('/')[-1]}"
    return citation

# Function to format the citation in MLA style
def format_mla_citation(paper_data):
    """Format the arXiv paper data into an MLA citation."""
    title = paper_data.get('title', 'No title available').replace("\n", " ")
    authors = paper_data.get('author', [])
    published_date = paper_data.get('published', 'No date available')
    year = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ").year
    
    if isinstance(authors, list):
        author_names = [f"{author['name'].split()[-1]}, {author['name'].split()[0]}" for author in authors]
        authors_str = ", and ".join(author_names)
    else:
        authors_str = f"{authors['name'].split()[-1]}, {authors['name'].split()[0]}"
    
    citation = f"{authors_str}. \"{title}.\" arXiv, {year}, https://arxiv.org/abs/{paper_data.get('id').split('/')[-1]}."
    return citation

# Function to format the citation in Chicago style
def format_chicago_citation(paper_data):
    """Format the arXiv paper data into a Chicago citation."""
    title = paper_data.get('title', 'No title available').replace("\n", " ")
    authors = paper_data.get('author', [])
    published_date = paper_data.get('published', 'No date available')
    year = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ").year
    
    if isinstance(authors, list):
        author_names = [f"{author['name'].split()[-1]}, {author['name'].split()[0]}" for author in authors]
        authors_str = " and ".join(author_names)
    else:
        authors_str = f"{authors['name'].split()[-1]}, {authors['name'].split()[0]}"
    
    citation = f"{authors_str}. \"{title}.\" {year}. arXiv. https://arxiv.org/abs/{paper_data.get('id').split('/')[-1]}."
    return citation

# Function to format the citation in IEEE style
def format_ieee_citation(paper_data):
    """Format the arXiv paper data into an IEEE citation."""
    title = paper_data.get('title', 'No title available').replace("\n", " ")
    authors = paper_data.get('author', [])
    published_date = paper_data.get('published', 'No date available')
    year = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%SZ").year
    
    if isinstance(authors, list):
        author_names = [f"{author['name'][0]}. {author['name'].split()[-1]}" for author in authors]
        authors_str = ", ".join(author_names)
    else:
        authors_str = f"{authors['name'][0]}. {authors['name'].split()[-1]}"
    
    citation = f"{authors_str}, \"{title},\" arXiv, {year}. [Online]. Available: https://arxiv.org/abs/{paper_data.get('id').split('/')[-1]}."
    return citation

# Main function to be called from other scripts
def main():
    st.title("📚 Multi-Style Citation Generator")
    st.write("Generate citations for arXiv papers in APA, MLA, Chicago, or IEEE style.")

    # Input field for arXiv link
    arxiv_url = st.text_input("Enter the arXiv Paper URL:")

    # Dropdown for citation style
    citation_style = st.selectbox("Select Citation Style", ["APA", "MLA", "Chicago", "IEEE"])

    # Generate citation button
    if st.button("Generate Citation"):
        if arxiv_url:
            # Fetch the arXiv paper data
            paper_data = get_arxiv_data(arxiv_url)
            
            if paper_data:
                # Format citation based on selected style
                if citation_style == "APA":
                    citation = format_apa_citation(paper_data)
                elif citation_style == "MLA":
                    citation = format_mla_citation(paper_data)
                elif citation_style == "Chicago":
                    citation = format_chicago_citation(paper_data)
                elif citation_style == "IEEE":
                    citation = format_ieee_citation(paper_data)

                st.subheader(f"{citation_style} Citation:")
                st.write(citation)

                # Download button for citation as .txt file
                buffer = io.StringIO(citation)
                st.download_button(
                    label="Download Citation as TXT",
                    data=buffer.getvalue(),
                    file_name="citation.txt",
                    mime="text/plain"
                )
            else:
                st.error("Could not retrieve data. Please check the arXiv URL.")
        else:
            st.error("Please enter a valid arXiv URL.")

# Check if the file is being run directly or imported
if __name__ == "__main__":
    main()

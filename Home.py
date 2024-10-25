import streamlit as st

# Set the page config
st.set_page_config(page_title="The Best Research Tool Ever", layout="wide")

# Sidebar for navigation
st.sidebar.title("🔍 Navigate")
page = st.sidebar.radio("Go to", ["Home", "Understand Any Research Paper", "Generate Citations", "Search Similar Papers"])

# Home Page
if page == "Home":
    st.title("🔍 The Best Research Tool Ever")
    st.markdown(
        """
        ### Your AI-powered assistant for academic research
        Choose from the options below to make your research journey smoother:
        - **Understand any research paper**: Automatically analyze and summarize a research paper.
        - **Generate citations**: Quickly generate citations in APA, MLA, Chicago, and other formats.
        - **Search similar papers**: Find similar papers based on the content of any research paper.
        """
    )

# Navigate to the "Understand Any Research Paper" page
elif page == "Understand Any Research Paper":
    st.write("🔍 Redirecting to **Understand Any Research Paper**...")
    import understand_paper  # Import the module
    understand_paper.main()  # Call the main function from understand_paper.py

# Navigate to the "Generate Citations" page
elif page == "Generate Citations":
    st.write("📚 Redirecting to **Generate Citations**...")
    import citation_generator  # Import the module
    citation_generator.main()  # Call the main function from citation_generator.py

# Navigate to the "Search Similar Papers" page (Placeholder for future functionality)
elif page == "Search Similar Papers":
    st.write("📄 **Search Similar Papers**...")
    import search_similar_papers  # Import the module
    search_similar_papers.main()  # Call the main function from search_similar_papers.py

